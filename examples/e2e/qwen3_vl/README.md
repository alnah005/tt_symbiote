# Qwen3-VL end-to-end demos

One script per officially supported dense `Qwen/Qwen3-VL-*-Instruct`
variant. Every script uses the same
[`Qwen3VLRecipe`](../../../src/tt_symbiote/models/qwen3_vl/modeling_qwen3_vl.py)
— variants differ only in (a) HF model id, (b) `MeshShape`, and (c)
fabric config. **Phase 8 Wave B** moved four structurally simple
Qwen3-VL classes onto Tenstorrent silicon (`Qwen3VLTextRMSNorm`,
`Qwen3VLTextMLP`, `Qwen3VLVisionMLP`, `Qwen3VLVisionPatchMerger`); the
bespoke pieces (M-RoPE precompute, Q/K head-norm-aware text attention,
varlen-packed vision SDPA, DeepStack injection, Conv3d patch embed)
stay on CPU until Wave B+1.

The two MoE variants (`Qwen/Qwen3-VL-30B-A3B-Instruct`,
`Qwen/Qwen3-VL-235B-A22B-Instruct`) register under a distinct HF top-
level head (`Qwen3VLMoeForConditionalGeneration`) and will be ported
under a separate `qwen3_vl_moe` recipe in a follow-up commit; they are
intentionally absent from this folder.

## Variant matrix

| Script | HF model id | Mesh | Fabric | Weights (BF16) | RAM (recommended) | HW verified | TTNN swaps active? | Notes |
|---|---|---|---|---|---|---|---|---|
| [`run_qwen3_vl_2b.py`](run_qwen3_vl_2b.py) | `Qwen/Qwen3-VL-2B-Instruct` | `(1, 1)` (N150) | `DISABLED` | ~4 GB | 8 GB | ✅ | yes (4 Wave B swaps) | Originally landed via the [`port-hf-model-to-tt-symbiote`](../../../.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md) skill (CPU-first); Phase 8 Wave B added 4 on-device swaps |
| [`run_qwen3_vl_4b.py`](run_qwen3_vl_4b.py) | `Qwen/Qwen3-VL-4B-Instruct` | `(1, 1)` (N150) | `DISABLED` | ~8 GB | 12 GB | ⏳ | yes (structurally) | Same Wave B swap map; waiting on a hardware run |
| [`run_qwen3_vl_8b.py`](run_qwen3_vl_8b.py) | `Qwen/Qwen3-VL-8B-Instruct` | `(1, 1)` (N150) | `DISABLED` | ~16 GB | 22 GB | ⏳ | yes (structurally) | Same Wave B swap map; waiting on a hardware run |
| [`run_qwen3_vl_32b.py`](run_qwen3_vl_32b.py) | `Qwen/Qwen3-VL-32B-Instruct` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~60 GB | 80 GB | ⏳ | tbd | T3K target. Replicated Wave B weights will likely exceed the Gemma-4 budget gate's 9 GB/chip ceiling; a sibling gate in `Qwen3VLRecipe` is a Wave B+1 follow-up if hardware confirms this |

`hw_verified` flag flips to `True` in
[`QWEN3_VL_TTNN_TUNING`](../../../src/tt_symbiote/models/qwen3_vl/configuration_qwen3_vl.py)
once a script produces a semantically correct answer on the listed
hardware target.

## Shared shape

Every script in this folder is the same 100-line template — identical
to the Gemma-4 demos except for the model id, mesh shape, and fabric
config. See [`../gemma4/README.md`](../gemma4/README.md) "Shared
shape" for the full sequence.

The per-script `_coverage.json` artefacts aggregate into
[`docs/cpu_vs_device_coverage.md`](../../../docs/cpu_vs_device_coverage.md).

## Note on semantic check

The Qwen3-VL family is terse: where Gemma-4 answers *"The animal in the
photo is a **dog**…"*, Qwen3-VL-2B answers *"…the animal is a **Golden
Retriever puppy**…"* without ever using the word "dog". Every script
here therefore accepts any of `{dog, puppy, retriever, labrador,
poodle, terrier, spaniel, shepherd, husky, bulldog}` as a successful
answer. Extend the list if a new variant volunteers a breed that
isn't covered — see
[`reference-vlm.md`](../../../.cursor/skills/port-hf-model-to-tt-symbiote/reference-vlm.md)
"Semantic check" for the rationale.

## Reproducing a clean run

```bash
source /home/<you>/tt_symbiote/.venv/bin/activate
python examples/e2e/qwen3_vl/run_qwen3_vl_2b.py
```

Expected wall-clock on a warm cache for 2B: ~50 s.
