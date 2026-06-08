# Qwen3-VL end-to-end demos

One script per supported dense `Qwen/Qwen3-VL-*-Instruct` variant, all sharing
the same [`Qwen3VLRecipe`](../../../src/tt_symbiote/models/qwen3_vl/modeling_qwen3_vl.py)
(variants differ only in HF model id, `MeshShape`, fabric config). Four
structurally simple classes (`Qwen3VLTextRMSNorm`, `Qwen3VLTextMLP`,
`Qwen3VLVisionMLP`, `Qwen3VLVisionPatchMerger`) run on device; bespoke pieces
(M-RoPE, head-norm-aware text attention, varlen vision SDPA, DeepStack, Conv3d
patch embed) stay on CPU. The two MoE variants (`Qwen3-VL-30B-A3B`,
`Qwen3-VL-235B-A22B`) register under a distinct head
(`Qwen3VLMoeForConditionalGeneration`) and land under a separate `qwen3_vl_moe`
recipe; they are intentionally absent here.

## Variant matrix

| Script | HF model id | Mesh | Fabric | Weights (BF16) | RAM | HW verified | TTNN swaps | Notes |
|---|---|---|---|---|---|---|---|---|
| [`run_qwen3_vl_2b.py`](run_qwen3_vl_2b.py) | `Qwen/Qwen3-VL-2B-Instruct` | `(1, 1)` (N150) | `DISABLED` | ~4 GB | 8 GB | ✅ | yes (4 on-device swaps) | landed via the porting skill |
| [`run_qwen3_vl_4b.py`](run_qwen3_vl_4b.py) | `Qwen/Qwen3-VL-4B-Instruct` | `(1, 1)` (N150) | `DISABLED` | ~8 GB | 12 GB | ⏳ | yes (structurally) | awaiting a hardware run |
| [`run_qwen3_vl_8b.py`](run_qwen3_vl_8b.py) | `Qwen/Qwen3-VL-8B-Instruct` | `(1, 1)` (N150) | `DISABLED` | ~16 GB | 22 GB | ⏳ | yes (structurally) | awaiting a hardware run |
| [`run_qwen3_vl_32b.py`](run_qwen3_vl_32b.py) | `Qwen/Qwen3-VL-32B-Instruct` | `(1, 8)` (T3K) | `FABRIC_1D_RING` | ~60 GB | 80 GB | ⏳ | tbd | T3K; replicated weights may need a budget gate |

`hw_verified` flips to `True` in
[`QWEN3_VL_TTNN_TUNING`](../../../src/tt_symbiote/models/qwen3_vl/configuration_qwen3_vl.py)
once a script answers correctly on its hardware target.

## Semantic check

Qwen3-VL is terse — it may answer *"a Golden Retriever puppy"* without the word
"dog", so each script accepts any of `{dog, puppy, retriever, labrador, poodle,
terrier, spaniel, shepherd, husky, bulldog}` (rationale in
[`reference-vlm.md`](../../../.cursor/skills/port-hf-model-to-tt-symbiote/reference-vlm.md)).

## Reproduce

Scripts follow the [Shared VLM run shape](../README.md#shared-vlm-run-shape).

```bash
source .venv/bin/activate
python examples/e2e/qwen3_vl/run_qwen3_vl_2b.py
```
