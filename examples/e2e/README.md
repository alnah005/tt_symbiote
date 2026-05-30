# End-to-end run scripts

This directory tracks **standalone reproducers that exercise `tt_symbiote`
from `AutoModelForCausalLM.from_pretrained` through `set_device` to
`model.generate`**, with one script per model that has been verified to
produce coherent output on real Tenstorrent hardware.

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

| Model | Script | Hardware verified on | Walkthrough |
|---|---|---|---|
| `inclusionAI/Ling-mini-2.0` | [`run_ling_mini_2_0.py`](run_ling_mini_2_0.py) | T3K (1x8) | [`docs/ling_mini_2_0_guide.md`](../../docs/ling_mini_2_0_guide.md) |

## Adding a new model

1. Land the model's recipe under `src/tt_symbiote/models/<name>/` and a
   pytest-based hardware smoke under `tests/capabilities/<name>/` (per
   [`PROJECT_PROPOSAL.md`](../../PROJECT_PROPOSAL.md) §6).
2. Copy an existing script in this folder as a starting point.
3. Swap the HF model ID, mesh geometry (`MeshShape(...)`), and the
   per-model KV-cache budget (`kv_cache_kwargs={"max_num_blocks": N}`).
4. Run it end-to-end on the target hardware. Only land a script that
   has been observed to emit a coherent, EOS-terminated response.
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
```

No `tt-metal` source checkout is required.
