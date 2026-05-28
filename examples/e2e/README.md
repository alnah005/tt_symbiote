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

## Adding a new model

1. Land the model's recipe under `src/tt_symbiote/models/<name>/` and a
   pytest-based hardware smoke under `tests/models/<name>/` (per
   [`PROJECT_PROPOSAL.md`](../../PROJECT_PROPOSAL.md) §6).
2. Copy an existing script in this folder as a starting point.
   - LLM (causal LM): start from `run_ling_mini_2_0.py`.
   - Vision (image classification): start from `run_resnet50.py`.
3. Swap the HF model ID, mesh geometry (`MeshShape(...)`), and any
   per-model knobs:
   - LLM: `kv_cache_kwargs={"max_num_blocks": N}` for the paged KV
     cache budget.
   - Vision: `l1_small_size=...` if the first/heaviest conv needs more
     scratch (see `RESNET_TTNN_TUNING` in
     `src/tt_symbiote/models/resnet/configuration_resnet.py`).
4. Run it end-to-end on the target hardware. Only land a script that
   has been observed to emit a coherent response (LLM: EOS-terminated
   token sequence; vision: a sensible top-1 ImageNet label for a known
   image).
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
```

No `tt-metal` source checkout is required.
