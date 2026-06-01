"""End-to-end smoke run for ``inclusionAI/Ling-mini-2.0`` on a Tenstorrent T3K.

Loads Ling-mini-2.0 through ``tt_symbiote.AutoModelForCausalLM``, binds the
sub-tree to a 1x8 mesh via :func:`set_device`, and exercises
``model.generate`` against a short chat-templated prompt.

Verified on T3K: produces a coherent, EOS-terminated response in ~4–5 min
(prefill + decode). See ``docs/ling_mini_2_0_guide.md`` for the line-by-line
walkthrough, the underlying recipe contract, and a troubleshooting table.

Usage::

    source .venv/bin/activate        # see scripts/bootstrap_venv.sh
    python examples/e2e/run_ling_mini_2_0.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MESH_DEVICE", "T3K")

import torch
import ttnn
from transformers import AutoTokenizer

from tt_symbiote import AutoModelForCausalLM, compatibility, set_device

# Fabric config must be set BEFORE open_mesh_device in current tt-metal HEAD;
# `fabric_config=` is no longer a kwarg to `open_mesh_device` (it was removed
# in favor of the `ttnn.set_fabric_config(...)` setter). This mirrors how
# tt-metal's pytest `mesh_device` fixture handles it (see
# `tt-metal/conftest.py::set_fabric`).
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),  # T3K is 1x8
    trace_region_size=200_000_000,
    num_command_queues=1,
)

tokenizer = AutoTokenizer.from_pretrained(
    "inclusionAI/Ling-mini-2.0",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0",
    trust_remote_code=True,
    dtype="auto",
)

# Paged KV-cache capacity = block_size (64) * max_num_blocks (512) = 32 768
# tokens of context (prompt + generated). The recipe default is 32 blocks
# (2 048 tokens) which is enough for short demos but truncates long answers.
# 512 blocks costs ~1.3 GB across the mesh (~160 MB / device) on Ling-mini-2.0;
# see `docs/ling_mini_2_0_guide.md` §B.3 for the budget table.
set_device(model, mesh_device, kv_cache_kwargs={"max_num_blocks": 512})
assert hasattr(model, "_tt_kv_cache"), "Phase 5 recipe should have allocated this"

model.eval()
torch.set_grad_enabled(False)

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain the difference between Python and C++ Programming Languages."}],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
inputs.pop("token_type_ids", None)

# `max_new_tokens` is deliberately set well above any expected answer length
# so the stop condition is the model's own EOS token (`<|role_end|>` in
# Ling-mini-2.0's chat template), not this soft cap. Keep this <= the KV-cache
# capacity above minus a budget for the prompt.
out = model.generate(
    **inputs,
    max_new_tokens=30_000,
    use_cache=True,
    past_key_values=model._tt_kv_cache,
)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :]))

# Persist compatibility.report next to the script. Phase 8.5: the
# JSON is a pure runtime observation artefact (gitignored, regenerated
# every run). Ling-mini-2.0 is a full TTNN port — `regressions` should
# stay `[]` and `modules_swapped.by_class` should match the 8 entries
# the recipe ships in `tt_implemented`.
report = compatibility.report(model)
print("\n=== tt_symbiote.compatibility.report(model) ===")
print(json.dumps(report, indent=2))
coverage_path = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
coverage_path.write_text(json.dumps(report, indent=2) + "\n")
print(f"Wrote coverage report to {coverage_path}")

# Match tt-metal's pytest `mesh_device` fixture teardown: close mesh, then
# disable fabric. Without this, the next process that calls
# `ttnn.set_fabric_config(...)` may inherit unstable state.
ttnn.close_mesh_device(mesh_device)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
