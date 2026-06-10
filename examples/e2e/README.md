# End-to-end run scripts

Standalone reproducers that exercise `tt_symbiote` from
`AutoModel*.from_pretrained` through `set_device` to a single forward (or
`model.generate`), one script per supported model *variant* (causal-LM, vision,
VLM). Run a script as-is and you should get a working response on the noted
hardware target; a regression here is a regression in `tt_symbiote`. These
scripts ship with the repo, not with `pip install` (per
[`docs/development/PROJECT_PROPOSAL.md`](../../docs/development/PROJECT_PROPOSAL.md) P10).
Each writes a `<script>_coverage.json` next to itself — a gitignored runtime
artefact, regenerated each run; the aggregated summary lives in
[`docs/development/cpu_vs_device_coverage.md`](../../docs/development/cpu_vs_device_coverage.md).

## Layout

```
examples/e2e/
├── run_ling_mini_2_0.py          ← flat: only 1 variant so far
├── dots_ocr/  run_dots_ocr.py + README.md      ← batched OCR, DP=4/8 (diff image per stream)
├── gemma4/    run_gemma4_{e2b,e4b,31b,26b_a4b}.py + README.md
├── qwen3_vl/  run_qwen3_vl_{2b,4b,8b,32b}.py + README.md
└── resnet/    run_resnet{18,34,50,101,152}.py + README.md
```

## Index (one row per script)

| Variant | Script | Task | Hardware | Status |
|---|---|---|---|---|
| `inclusionAI/Ling-mini-2.0` | [`run_ling_mini_2_0.py`](run_ling_mini_2_0.py) | causal LM | T3K (1×8) | ✅ verified |
| `rednote-hilab/dots.ocr` | [`dots_ocr/run_dots_ocr.py`](dots_ocr/run_dots_ocr.py) | OCR (VLM), batched DP=4/8 | T3K (8×1) / P150x4 (4×1) | ✅ verified (DP=8 T3K) |
| `microsoft/resnet-18` | [`resnet/run_resnet18.py`](resnet/run_resnet18.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `microsoft/resnet-34` | [`resnet/run_resnet34.py`](resnet/run_resnet34.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `microsoft/resnet-50` | [`resnet/run_resnet50.py`](resnet/run_resnet50.py) | image classification | N150 (1×1) | ✅ verified |
| `microsoft/resnet-101` | [`resnet/run_resnet101.py`](resnet/run_resnet101.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `microsoft/resnet-152` | [`resnet/run_resnet152.py`](resnet/run_resnet152.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `google/gemma-4-E2B-it` | [`gemma4/run_gemma4_e2b.py`](gemma4/run_gemma4_e2b.py) | VLM | N150 (1×1) | ✅ verified |
| `google/gemma-4-E4B-it` | [`gemma4/run_gemma4_e4b.py`](gemma4/run_gemma4_e4b.py) | VLM | N150 (1×1) | ✅ verified |
| `google/gemma-4-31B-it` | [`gemma4/run_gemma4_31b.py`](gemma4/run_gemma4_31b.py) | VLM | T3K (1×8) | ✅ verified (CPU via budget gate) |
| `google/gemma-4-26B-A4B-it` | [`gemma4/run_gemma4_26b_a4b.py`](gemma4/run_gemma4_26b_a4b.py) | VLM (MoE) | T3K (1×8) | ✅ verified (CPU via MoE gate) |
| `Qwen/Qwen3-VL-2B-Instruct` | [`qwen3_vl/run_qwen3_vl_2b.py`](qwen3_vl/run_qwen3_vl_2b.py) | VLM | N150 (1×1) | ✅ verified |
| `Qwen/Qwen3-VL-4B-Instruct` | [`qwen3_vl/run_qwen3_vl_4b.py`](qwen3_vl/run_qwen3_vl_4b.py) | VLM | N150 (1×1) | ⏳ structurally supported |
| `Qwen/Qwen3-VL-8B-Instruct` | [`qwen3_vl/run_qwen3_vl_8b.py`](qwen3_vl/run_qwen3_vl_8b.py) | VLM | N150 (1×1) | ⏳ structurally supported |
| `Qwen/Qwen3-VL-32B-Instruct` | [`qwen3_vl/run_qwen3_vl_32b.py`](qwen3_vl/run_qwen3_vl_32b.py) | VLM | T3K (1×8) | ⏳ structurally supported |

For per-family context (license / weight footprints / RAM / notes) see each
folder's `README.md`.

## Shared VLM run shape

The gemma4 and qwen3_vl scripts are vision-language demos and follow one shape:
open mesh → `AutoProcessor.from_pretrained` →
`AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16)` →
`set_device(model, mesh)` → `processor.apply_chat_template(...)` with image +
`"What is this animal in the photo?"` →
`model.generate(..., do_sample=False, use_cache=True)` →
`compatibility.report(model)` written to the gitignored `<script>_coverage.json`
(confirm `regressions == []`) → permissive dog-equivalents semantic check
(`dog`, `puppy`, `retriever`, …) → `ttnn.close_mesh_device(mesh)`. ResNet is
image classification, not a VLM — see [`resnet/README.md`](resnet/README.md).

The VLM scripts read `tests/images/test-dog.png`, which is gitignored (see
[`.gitignore`](../../.gitignore)). Drop any Pillow-readable dog photo there (the
`.png` extension is just the filename they look for); if it is missing the
scripts exit 0 with a `SKIP:` before acquiring a mesh device, so the whole tree
is safe on a fresh clone. VLM demos also need `pip install Pillow pillow-heif`.

## Running

Activate the venv (see [root README → Installation](../../README.md#installation)).
Requires a built tt-metal at `$TT_METAL_HOME` (it provides ttnn).

```bash
source .venv/bin/activate
python examples/e2e/run_ling_mini_2_0.py
```

## Adding a new model

Use the porting skill at
[`.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md`](../../.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md)
(ask your agent to *"port `<HF model id>` to `tt_symbiote`"*). It lands the
recipe under `src/tt_symbiote/models/<name>/`, a smoke test under
`tests/experimental/<name>/`, and a copy of this script under the matching family
folder. Manual contract:
[`docs/development/PROJECT_PROPOSAL.md`](../../docs/development/PROJECT_PROPOSAL.md) §6.
