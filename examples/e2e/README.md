# End-to-end run scripts

This directory tracks **standalone reproducers that exercise `tt_symbiote`
from `AutoModel*.from_pretrained` through `set_device` to a single
forward (or `model.generate`)**, with one script per supported model
*variant*. Causal-LM (`AutoModelForCausalLM` + `generate`), vision
(`AutoModelForImageClassification` + forward), and VLM
(`AutoModelForImageTextToText` + `generate`) flows all live here.

Each script is intended to be:

- **A reproducer.** Run it as-is and you should get a working response
  on the noted hardware target. If it stops working we treat that as a
  regression in `tt_symbiote`.
- **A reference shape.** Use it as the canonical template when adding a
  new model — copy from the matching family folder, swap the model id
  and mesh geometry. Boilerplate (fabric config, mesh open/close,
  teardown) stays identical unless your model truly needs something
  different.
- **Outside the installed package.** Per
  [`docs/internal/PROJECT_PROPOSAL.md`](../../docs/internal/PROJECT_PROPOSAL.md) P10 these scripts
  ship with the repo, not with `pip install tt_symbiote`. They are
  reference material, not installed entry points.

These differ from `examples/chat/HF_chat.py` (an interactive demo): the
files here are deliberately **one-shot, non-interactive smoke runs**.
After every successful run, each script writes a
`<script_name>_coverage.json` next to itself summarising what
*actually* executed on device. Phase 8.5 made these JSONs a **pure
runtime artefact**: they're gitignored, regenerated on every run, and
read locally. The aggregated textual summary lives in
[`docs/cpu_vs_device_coverage.md`](../../docs/cpu_vs_device_coverage.md).

## Layout

One script per HF model id, grouped into a folder per model family. The
exception is single-variant families (Ling today) which stay flat until
a sibling variant lands. The `_coverage.json` artefacts are **not
checked in** (Phase 8.5): each run regenerates them.

```
examples/e2e/
├── README.md                     ← this file
├── run_ling_mini_2_0.py          ← flat: only 1 variant so far
├── run_ling_mini_2_0_coverage.json   (gitignored; written on each run)
├── gemma4/
│   ├── README.md
│   ├── run_gemma4_{e2b,e4b,31b,26b_a4b}.py
│   └── run_gemma4_*_coverage.json    (gitignored)
├── qwen3_vl/
│   ├── README.md
│   ├── run_qwen3_vl_{2b,4b,8b,32b}.py
│   └── run_qwen3_vl_*_coverage.json  (gitignored)
└── resnet/
    ├── README.md
    ├── run_resnet{18,34,50,101,152}.py
    └── run_resnet*_coverage.json     (gitignored)
```

## Index (one row per script)

| Variant | Script | Task | Hardware | Status |
|---|---|---|---|---|
| `inclusionAI/Ling-mini-2.0` | [`run_ling_mini_2_0.py`](run_ling_mini_2_0.py) | causal LM | T3K (1×8) | ✅ verified (Phase 5) |
| `microsoft/resnet-18` | [`resnet/run_resnet18.py`](resnet/run_resnet18.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `microsoft/resnet-34` | [`resnet/run_resnet34.py`](resnet/run_resnet34.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `microsoft/resnet-50` | [`resnet/run_resnet50.py`](resnet/run_resnet50.py) | image classification | N150 (1×1) | ✅ verified (Phase 6) |
| `microsoft/resnet-101` | [`resnet/run_resnet101.py`](resnet/run_resnet101.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `microsoft/resnet-152` | [`resnet/run_resnet152.py`](resnet/run_resnet152.py) | image classification | N150 (1×1) | ⏳ structurally supported |
| `google/gemma-4-E2B-it` | [`gemma4/run_gemma4_e2b.py`](gemma4/run_gemma4_e2b.py) | VLM | N150 (1×1) | ✅ verified (Phase 8 Wave A) |
| `google/gemma-4-E4B-it` | [`gemma4/run_gemma4_e4b.py`](gemma4/run_gemma4_e4b.py) | VLM | N150 (1×1) | ✅ verified (Phase 8 Wave A) |
| `google/gemma-4-31B-it` | [`gemma4/run_gemma4_31b.py`](gemma4/run_gemma4_31b.py) | VLM | T3K (1×8) | ✅ verified (CPU-only via budget gate) |
| `google/gemma-4-26B-A4B-it` | [`gemma4/run_gemma4_26b_a4b.py`](gemma4/run_gemma4_26b_a4b.py) | VLM (MoE) | T3K (1×8) | ✅ verified (CPU-only via MoE gate) |
| `Qwen/Qwen3-VL-2B-Instruct` | [`qwen3_vl/run_qwen3_vl_2b.py`](qwen3_vl/run_qwen3_vl_2b.py) | VLM | N150 (1×1) | ✅ verified (Phase 7 — first via skill; Phase 8 Wave B added 4 TTNN swaps) |
| `Qwen/Qwen3-VL-4B-Instruct` | [`qwen3_vl/run_qwen3_vl_4b.py`](qwen3_vl/run_qwen3_vl_4b.py) | VLM | N150 (1×1) | ⏳ structurally supported |
| `Qwen/Qwen3-VL-8B-Instruct` | [`qwen3_vl/run_qwen3_vl_8b.py`](qwen3_vl/run_qwen3_vl_8b.py) | VLM | N150 (1×1) | ⏳ structurally supported |
| `Qwen/Qwen3-VL-32B-Instruct` | [`qwen3_vl/run_qwen3_vl_32b.py`](qwen3_vl/run_qwen3_vl_32b.py) | VLM | T3K (1×8) | ⏳ structurally supported |

For per-family context (license / weight footprints / RAM
requirements / variant-specific notes), see each folder's `README.md`.

## Adding a new model

**Prefer the porting skill.** The repeatable workflow below is encoded
as a Cursor project-scope skill at
[`.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md`](../../.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md).
Ask your Cursor agent to *"port `<HF model id>` to `tt_symbiote`"* and
the skill produces the recipe, tests, and a copy of the e2e script
under the appropriate family folder. Qwen3-VL-2B was landed this way
with zero design decisions during execution. The manual steps below
are what the skill automates:

1. Land the model's recipe under `src/tt_symbiote/models/<name>/` and a
   pytest-based hardware smoke under `tests/models/<name>/` (per
   [`docs/internal/PROJECT_PROPOSAL.md`](../../docs/internal/PROJECT_PROPOSAL.md) §6).
2. Copy an existing script from the appropriate family folder as a
   starting point.
   - LLM (causal LM): start from [`run_ling_mini_2_0.py`](run_ling_mini_2_0.py).
   - Vision (image classification): start from
     [`resnet/run_resnet50.py`](resnet/run_resnet50.py).
   - VLM (image-text-to-text): start from
     [`gemma4/run_gemma4_e2b.py`](gemma4/run_gemma4_e2b.py) or
     [`qwen3_vl/run_qwen3_vl_2b.py`](qwen3_vl/run_qwen3_vl_2b.py).
     Both cover the CPU-first port pattern (empty `build_module_dict`,
     coverage JSON writeback, permissive dog-equivalents semantic check).
3. Swap the HF model id, mesh geometry (`MeshShape(...)`), fabric
   config, and any per-model knobs:
   - LLM: `kv_cache_kwargs={"max_num_blocks": N}` as a keyword to
     `AutoModelForCausalLM.from_pretrained(...)` for the paged KV
     cache budget. The shape is a model-config decision, so it pairs
     with model loading; `set_device` consumes it implicitly when the
     device is bound. (Pass `kv_cache_kwargs=` to `set_device` only if
     you want to A/B-test budgets against the same loaded model
     without reloading.)
   - Vision: `l1_small_size=...` if the first/heaviest conv needs more
     scratch (see `RESNET_TTNN_TUNING` in
     `src/tt_symbiote/models/resnet/configuration_resnet.py`).
   - VLM: a multimodal `processor.apply_chat_template(...)` instead of
     the LLM `tokenizer.apply_chat_template(...)` (image goes in the
     `content` list).
4. Run it end-to-end on the target hardware. Only flip `hw_verified`
   to `True` after observing a coherent response (LLM: EOS-terminated
   token sequence; vision: a sensible top-1 ImageNet label for a known
   image; VLM: semantically correct answer to a known prompt). For
   CPU-first ports also confirm
   `tt_symbiote.compatibility.report(model)["regressions"]` is empty
   (every observed fallback was declared in `recipe.cpu_fallback`).
5. Add a row to the index table above. Every demo writes a local
   `_coverage.json` next to itself (Phase 8.5: gitignored, regenerated
   on every run — don't commit it). Update
   [`docs/cpu_vs_device_coverage.md`](../../docs/cpu_vs_device_coverage.md)
   if the per-model textual breakdown changed.

## Running

These assume the standalone venv from
[`scripts/bootstrap_venv.sh`](../../scripts/bootstrap_venv.sh) is
already built and activated:

```bash
source /home/<you>/tt_symbiote/.venv/bin/activate
python examples/e2e/run_ling_mini_2_0.py
python examples/e2e/resnet/run_resnet50.py
python examples/e2e/gemma4/run_gemma4_e2b.py
python examples/e2e/qwen3_vl/run_qwen3_vl_2b.py
```

No `tt-metal` source checkout is required.

For the VLM demos you'll also need image-processing extras (only
the first time):

```bash
pip install Pillow pillow-heif
```

(The repo's `tests/images/test-dog.png` is actually AVIF-encoded;
`Pillow` reads it natively on most distributions but `pillow-heif`
is a safe fallback if your PIL build lacks AVIF support.)
