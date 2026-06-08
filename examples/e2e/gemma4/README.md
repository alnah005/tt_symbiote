# Gemma-4 end-to-end demos

One script per supported `google/gemma-4-*-it` variant, all sharing the same
[`Gemma4Recipe`](../../../src/tt_symbiote/models/gemma4/modeling_gemma4.py)
(variants differ only in HF model id, `MeshShape`, fabric config). Structurally
simple classes (RMSNorm with scale, scaled word embedding, text MLP, vision MLP,
multimodal embedder) run on device; bespoke attention / RoPE / per-layer-embedding
stay on CPU.

## Variant matrix

| Script | HF model id | Mesh | Fabric | Weights (BF16) | RAM | HW verified | TTNN swaps | Notes |
|---|---|---|---|---|---|---|---|---|
| [`run_gemma4_e2b.py`](run_gemma4_e2b.py) | `google/gemma-4-E2B-it` | `(1, 1)` (N150) | `DISABLED` | ~9.6 GB | 16 GB | ✅ | yes | reference reproducer |
| [`run_gemma4_e4b.py`](run_gemma4_e4b.py) | `google/gemma-4-E4B-it` | `(1, 1)` (N150) | `DISABLED` | ~18 GB | 24 GB | ✅ | yes | same on-device swaps |
| [`run_gemma4_31b.py`](run_gemma4_31b.py) | `google/gemma-4-31B-it` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~58 GB | 80 GB | ✅ | no (budget-gated) | replicated weights exceed per-chip budget → CPU |
| [`run_gemma4_26b_a4b.py`](run_gemma4_26b_a4b.py) | `google/gemma-4-26B-A4B-it` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~50 GB | 70 GB | ✅ | no (MoE-gated) | MoE text block → CPU |

`hw_verified` flips to `True` in
[`GEMMA4_TTNN_TUNING`](../../../src/tt_symbiote/models/gemma4/configuration_gemma4.py)
once a script answers correctly on its hardware target.

## Budget + MoE gate

`Gemma4Recipe.build_module_dict` returns `{}` (everything stays on the HF
reference module), emitting a `UserWarning`, when either: the replicated weight
footprint exceeds the per-chip budget (31B-it → CPU until tensor-parallel
sharding lands), or the MoE text block is enabled (26B-A4B-it → CPU until MoE
wrappers land). See
[`docs/development/cpu_vs_device_coverage.md`](../../../docs/development/cpu_vs_device_coverage.md).

## License + reproduce

The `google/gemma-4-*-it` checkpoints are gated; `from_pretrained` raises
`transformers.utils.GatedRepoError` until the calling token has accepted the
Gemma license on each variant's model card. Scripts follow the
[Shared VLM run shape](../README.md#shared-vlm-run-shape).

```bash
source .venv/bin/activate
python examples/e2e/gemma4/run_gemma4_e2b.py
```
