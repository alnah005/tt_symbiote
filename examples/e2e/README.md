# End-to-end run scripts

This directory tracks **standalone reproducers that exercise `tt_symbiote`
from `AutoModel*.from_pretrained` through `set_device` to a single
forward (or `model.generate`)**, with one script per model that has been
verified to produce sensible output on real Tenstorrent hardware. Both
LLM-shaped (`AutoModelForCausalLM` + `generate`) and vision-shaped
(`AutoModelForImageClassification` + forward) flows live here.

Each script is intended to be:

- **A reproducer.** Run it as-is and you should get a working response on
  the noted hardware target. If it stops working we treat that as a
  regression in `tt_symbiote`.
- **A reference shape.** Use it as the canonical template when adding a
  new model — copy, retitle, swap the model ID, mesh geometry, and the
  per-model KV-cache budget; leave the boilerplate (fabric config, mesh
  open/close, teardown) identical unless your model truly needs
  something different.
- **Outside the installed package.** Per
  [`PROJECT_PROPOSAL.md`](../../PROJECT_PROPOSAL.md) P10 these scripts
  ship with the repo, not with `pip install tt_symbiote`. They are
  reference material, not installed entry points.

These differ from `examples/chat/HF_chat.py` (an interactive demo): the
files here are deliberately **one-shot, non-interactive smoke runs**.

## Index

| Model | Script | Task | Hardware verified on | Walkthrough |
|---|---|---|---|---|
| `inclusionAI/Ling-mini-2.0` | [`run_ling_mini_2_0.py`](run_ling_mini_2_0.py) | causal LM | T3K (1x8) | [`docs/ling_mini_2_0_guide.md`](../../docs/ling_mini_2_0_guide.md) |
| `microsoft/resnet-50` | [`run_resnet50.py`](run_resnet50.py) | image classification | N150 / T3K (1x1) | (none yet) |
| `google/gemma-4-E2B-it` | [`run_gemma4_e2b.py`](run_gemma4_e2b.py) | image-text-to-text (VLM) | N150 (1x1), CPU-first | (Phase 7 in [`docs/migration_notes.md`](../../docs/migration_notes.md)) |
| `google/gemma-4-31B-it` | [`run_gemma4_31b.py`](run_gemma4_31b.py) | image-text-to-text (VLM) | structurally supported on T3K (1x8), CPU-first | (Phase 7 in [`docs/migration_notes.md`](../../docs/migration_notes.md)) |
| `Qwen/Qwen3-VL-2B-Instruct` | [`run_qwen3_vl_2b.py`](run_qwen3_vl_2b.py) | image-text-to-text (VLM) | N150 (1x1), CPU-first | (Phase 7 follow-up in [`docs/migration_notes.md`](../../docs/migration_notes.md); first model landed via the [`port-hf-model-to-tt-symbiote`](../../.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md) skill) |

## Adding a new model

**Prefer the porting skill.** As of the Phase 7 follow-up commit the
repeatable workflow below is encoded as a Cursor project-scope skill at
[`.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md`](../../.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md).
Ask your Cursor agent to *"port `<HF model id>` to `tt_symbiote`"* and
the skill produces the recipe, tests, and a copy of the e2e script in
this folder. Qwen3-VL-2B was landed this way with zero design
decisions during execution. The manual steps below are still accurate
for hand-written ports and are what the skill automates:

1. Land the model's recipe under `src/tt_symbiote/models/<name>/` and a
   pytest-based hardware smoke under `tests/models/<name>/` (per
   [`PROJECT_PROPOSAL.md`](../../PROJECT_PROPOSAL.md) §6).
2. Copy an existing script in this folder as a starting point.
   - LLM (causal LM): start from `run_ling_mini_2_0.py`.
   - Vision (image classification): start from `run_resnet50.py`.
   - VLM (image-text-to-text): start from `run_gemma4_e2b.py`. Also
     covers the CPU-first port pattern (empty `build_module_dict` +
     `compatibility.report` at the end of the run).
3. Swap the HF model ID, mesh geometry (`MeshShape(...)`), and any
   per-model knobs:
   - LLM: `kv_cache_kwargs={"max_num_blocks": N}` for the paged KV
     cache budget.
   - Vision: `l1_small_size=...` if the first/heaviest conv needs more
     scratch (see `RESNET_TTNN_TUNING` in
     `src/tt_symbiote/models/resnet/configuration_resnet.py`).
   - VLM: a multimodal `processor.apply_chat_template(...)` instead of
     the LLM `tokenizer.apply_chat_template(...)` (image goes in the
     `content` list); see `GEMMA4_TTNN_TUNING` for the per-variant
     mesh-shape table.
4. Run it end-to-end on the target hardware. Only land a script that
   has been observed to emit a coherent response (LLM: EOS-terminated
   token sequence; vision: a sensible top-1 ImageNet label for a known
   image; VLM: semantically correct answer to a known prompt — e.g.
   the Gemma-4 demo expects `"dog"` in the response for the dog
   photo). For CPU-first ports, also confirm
   `tt_symbiote.compatibility.report(model)["runtime_observed"]["unexpected"]`
   is empty (every CPU fallback was declared in the recipe).
5. Add a row to the table above with the hardware target it was
   verified on, and link a walkthrough (`docs/<model>_guide.md`) if one
   exists.

## Running

These assume the standalone venv from
[`scripts/bootstrap_venv.sh`](../../scripts/bootstrap_venv.sh) is
already built and activated:

```bash
source /home/<you>/tt_symbiote/.venv/bin/activate
python examples/e2e/run_ling_mini_2_0.py
python examples/e2e/run_resnet50.py
python examples/e2e/run_gemma4_e2b.py
```

No `tt-metal` source checkout is required.

For the Gemma-4 demos you'll also need image-processing extras (only
the first time):

```bash
pip install Pillow pillow-heif
```

(The repo's `tests/images/test-dog.png` is actually AVIF-encoded;
`Pillow` reads it natively on most distributions but `pillow-heif`
is a safe fallback if your PIL build lacks AVIF support.)
