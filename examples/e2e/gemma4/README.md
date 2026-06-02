# Gemma-4 end-to-end demos

One script per officially supported `google/gemma-4-*-it` variant. Every
script uses the same [`Gemma4Recipe`](../../../src/tt_symbiote/models/gemma4/modeling_gemma4.py)
— variants differ only in (a) HF model id, (b) `MeshShape`, and (c)
fabric config. **Phase 8 Wave A** moved five structurally simple Gemma-4
classes onto Tenstorrent silicon (RMSNorm with scale, scaled word
embedding, text MLP, vision MLP, multimodal embedder); the bespoke
attention / RoPE / Per-Layer Embedding pieces stay on CPU until Wave A+1.
The recipe also runs a **budget + MoE gate** in `build_module_dict` that
short-circuits the Wave A swaps for variants whose replicated weight
footprint exceeds 9 GB/chip (31B-it) or that use the MoE text block
(26B-A4B-it) — those variants run end-to-end on CPU until tensor-
parallel sharding and the MoE wrappers land. See "Budget + MoE gating"
below for details.

## Variant matrix

| Script | HF model id | Mesh | Fabric | Weights (BF16) | RAM (recommended) | HW verified | TTNN swaps active? | Notes |
|---|---|---|---|---|---|---|---|---|
| [`run_gemma4_e2b.py`](run_gemma4_e2b.py) | `google/gemma-4-E2B-it` | `(1, 1)` (N150) | `DISABLED` | ~9.6 GB | 16 GB | ✅ | yes (5 Wave A swaps) | Phase 8 Wave A reference reproducer; identifies the dog correctly |
| [`run_gemma4_e4b.py`](run_gemma4_e4b.py) | `google/gemma-4-E4B-it` | `(1, 1)` (N150) | `DISABLED` | ~18 GB | 24 GB | ✅ | yes (5 Wave A swaps) | Same Wave A swaps; verified on N150 |
| [`run_gemma4_31b.py`](run_gemma4_31b.py) | `google/gemma-4-31B-it` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~58 GB | 80 GB | ✅ | no (budget-gated) | T3K target. ~43.5 GB replicated weights exceed 9 GB/chip budget → CPU-only run, semantically correct answer, 0 unexpected fallbacks |
| [`run_gemma4_26b_a4b.py`](run_gemma4_26b_a4b.py) | `google/gemma-4-26B-A4B-it` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~50 GB | 70 GB | ✅ | no (MoE-gated) | MoE; T3K target. `Gemma4TextExperts` + per-head norms incompatible with Wave A wrappers → CPU-only run, semantically correct answer, 0 unexpected fallbacks |

`hw_verified` flag flips to `True` in
[`GEMMA4_TTNN_TUNING`](../../../src/tt_symbiote/models/gemma4/configuration_gemma4.py)
once a script produces a semantically correct answer on the listed
hardware target.

## Budget + MoE gating

`Gemma4Recipe.build_module_dict` calls `_ttnn_swap_is_safe(model)`
before returning the Wave A swap map. The check returns `False` when
either:

- **Replicated weight footprint > 9 GB per chip** (`_TTNN_PER_CHIP_BUDGET_BYTES`).
  Default `ttnn.to_device` replicates weights across every chip in the
  mesh; tensor-parallel sharding is a Wave A+2 follow-up. The
  31B-it variant's Wave A weights alone would replicate ~43.5 GB into
  each T3K chip (~3.6× over the 12 GB DRAM ceiling); the gate forces
  the model onto CPU until shard-aware `TTNNLinear` / `TTNNEmbedding`
  paths land.
- **`text_config.enable_moe_block == True`.** The 26B-A4B-it variant
  ships `Gemma4TextExperts` + `Gemma4TextRouter` (expert sub-FFNs that
  the dense `Gemma4TextMLP` wrapper does not cover) and per-head
  RMSNorms whose `dim` (32 / 96) the current TTNN RMSNorm tile geometry
  rejects. A partial swap produces hundreds of shape-validation
  fallbacks; the gate cleanly avoids that.

When the gate triggers, the recipe emits a `UserWarning` describing
the reason, attaches `_tt_runtime_config["ttnn_swap_skipped"] = True`
plus `ttnn_swap_skipped_reason` / `ttnn_replicated_footprint_bytes`,
and returns `{}` so every module stays as the HF reference module.
The model still runs end-to-end through the TTNN mesh device handle
(opened to satisfy the `set_device` contract) and produces a
semantically correct answer. The runtime JSON for the gated variants
reflects the gate cleanly: `ttnn_swap_skipped == true`,
`modules_swapped == {by_class: {}, by_module: {}}`, `regressions ==
[]`.

Lifting the gate is the natural milestone for Wave A+1 (MoE wrappers)
and Wave A+2 (tensor-parallel sharding). Each can be flipped on
without touching the recipe API.

## License + access

The `google/gemma-4-*-it` checkpoints are gated on Hugging Face Hub.
`from_pretrained` raises `transformers.utils.GatedRepoError` if the
calling token has not accepted the Gemma license on each variant's
model card. Accept once per checkpoint before running.

## Shared shape

Every script in this folder is the same 100-line template:

1. `ttnn.set_fabric_config(...)` + `ttnn.open_mesh_device(...)`
2. `AutoProcessor.from_pretrained(MODEL_ID)`
3. `AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)`
4. `set_device(model, mesh_device, dump_visualization=False)`
5. `processor.apply_chat_template(...)` against
   [`tests/images/test-dog.png`](../../../tests/images/test-dog.png) +
   `"What is this animal in the photo?"`
6. `model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)`
7. `compatibility.report(model)` → printed and written to
   `<script>_coverage.json` next to the script. Phase 8.5 made this
   JSON a runtime-only artefact: gitignored, regenerated each run,
   read locally to confirm `regressions == []` and the expected
   `modules_swapped` / `runtime_observed.successes_by_class`
   populations.
8. Permissive dog-equivalents semantic assertion.

The aggregated textual summary across all variants lives in
[`docs/development/cpu_vs_device_coverage.md`](../../../docs/development/cpu_vs_device_coverage.md).

## Reproducing a clean run

```bash
source /home/<you>/tt_symbiote/.venv/bin/activate
python examples/e2e/gemma4/run_gemma4_e2b.py
```

Expected wall-clock on a warm cache for E2B: ~45 s (TTNN init + cached
weight load + 35-token generation; 5 Wave A modules execute on the
N150 device, the bespoke attention + decoder loop still run on host
CPU pending Wave A+1). The gated variants (31B-it, 26B-A4B-it) run
entirely on host CPU and are bottlenecked by host throughput; expect
a few minutes per token.
