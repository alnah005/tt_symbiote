# Gemma-4 end-to-end demos

One script per officially supported `google/gemma-4-*-it` variant. Every
script uses the same [`Gemma4Recipe`](../../../src/tt_symbiote/models/gemma4/modeling_gemma4.py)
— variants differ only in (a) HF model id, (b) `MeshShape`, and (c)
fabric config. The recipe is **CPU-first**: every Gemma-4 submodule
(vision tower, multimodal embedder, text decoder) currently runs on the
host PyTorch backend; the TTNN mesh is opened so the recipe walker can
attach `model._tt_runtime_config` for downstream wrappers.

## Variant matrix

| Script | HF model id | Mesh | Fabric | Weights (BF16) | RAM (recommended) | HW verified | Notes |
|---|---|---|---|---|---|---|---|
| [`run_gemma4_e2b.py`](run_gemma4_e2b.py) | `google/gemma-4-E2B-it` | `(1, 1)` (N150) | `DISABLED` | ~9.6 GB | 16 GB | ✅ | Phase 7 reference reproducer; identifies the dog correctly |
| [`run_gemma4_e4b.py`](run_gemma4_e4b.py) | `google/gemma-4-E4B-it` | `(1, 1)` (N150) | `DISABLED` | ~18 GB | 24 GB | ⏳ | Structurally supported |
| [`run_gemma4_31b.py`](run_gemma4_31b.py) | `google/gemma-4-31B-it` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~58 GB | 80 GB | ⏳ | T3K target |
| [`run_gemma4_26b_a4b.py`](run_gemma4_26b_a4b.py) | `google/gemma-4-26B-A4B-it` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~50 GB | 70 GB | ⏳ | MoE; T3K target |

`hw_verified` flag flips to `True` in
[`GEMMA4_TTNN_TUNING`](../../../src/tt_symbiote/models/gemma4/configuration_gemma4.py)
once a script produces a semantically correct answer on the listed
hardware target.

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
   `<script>_coverage.json` next to the script.
8. Permissive dog-equivalents semantic assertion.

The per-script `_coverage.json` artefacts aggregate into
[`docs/cpu_vs_device_coverage.md`](../../../docs/cpu_vs_device_coverage.md).

## Reproducing a clean run

```bash
source /home/<you>/tt_symbiote/.venv/bin/activate
python examples/e2e/gemma4/run_gemma4_e2b.py
```

Expected wall-clock on a warm cache for E2B: ~45 s (TTNN init + cached
weight load + 35-token generation, all on host CPU). The other variants
are bottlenecked by host CPU throughput; expect a few minutes per token
for the 31B / 26B-A4B variants.
