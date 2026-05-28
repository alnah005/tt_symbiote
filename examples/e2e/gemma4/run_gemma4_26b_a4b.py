"""End-to-end "what is this animal?" demo for ``google/gemma-4-26B-A4B-it`` on T3K.

Sibling of :mod:`examples.e2e.gemma4.run_gemma4_e2b` for the 26B-A4B MoE
variant of Gemma-4. Same recipe (:class:`Gemma4Recipe`), same chat
template, same image + prompt, same semantic check; mesh shape jumps to
``(1, 8)`` for the T3K target.

Resource notes
--------------

* Weights are ~50 GB in BF16 (26B total parameters, A4B active per
  token via the MoE router). ``from_pretrained`` downloads once; the
  full weight tensor still loads into host RAM under the CPU-first
  port. Allocate **at least 70 GB free RAM** before running.
* HF Hub access to ``google/gemma-4-26B-A4B-it`` requires accepting
  the Gemma license once via the model card.
* CPU forward of a 26B MoE transformer is slow: expect tens of seconds
  to minutes per token on a typical host. The 64-token cap below is a
  soft limit; the model's EOS terminates earlier on the dog question.
* Even on the CPU-first path we open the full T3K mesh — the recipe's
  ``post_register`` reads the ``(1, 8)`` ``mesh_shape`` from
  :data:`GEMMA4_TTNN_TUNING` and attaches it as
  ``model._tt_runtime_config`` so the next-phase TTNN port can
  immediately consume it.

This script is structurally supported (recipe is variant-agnostic) but
not yet hardware-verified; the smaller E2B variant
(:mod:`run_gemma4_e2b`) is the recommended first run.

Usage::

    source .venv/bin/activate        # see scripts/bootstrap_venv.sh
    python examples/e2e/gemma4/run_gemma4_26b_a4b.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MESH_DEVICE", "T3K")

import torch  # noqa: E402
import ttnn  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from tt_symbiote import AutoModelForImageTextToText, compatibility, set_device  # noqa: E402

MODEL_ID = "google/gemma-4-26B-A4B-it"

IMAGE_PATH = Path(__file__).resolve().parents[3] / "tests" / "images" / "test-dog.png"
PROMPT = "What is this animal in the photo?"

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    trace_region_size=200_000_000,
    num_command_queues=1,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
)

set_device(model, mesh_device, dump_visualization=False)
assert hasattr(model, "_tt_runtime_config"), (
    "Gemma4Recipe.post_register should have attached _tt_runtime_config"
)
assert model._tt_runtime_config["mesh_shape"] == (1, 8), (
    f"26B-A4B should target the full T3K mesh; got {model._tt_runtime_config['mesh_shape']}"
)

model.eval()
torch.set_grad_enabled(False)

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: (v.to("cpu") if hasattr(v, "to") else v) for k, v in inputs.items()}

out = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    use_cache=True,
)

prompt_len = inputs["input_ids"].shape[-1]
answer = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()
print(f"Gemma-4 26B-A4B answer: {answer!r}")

report = compatibility.report(model)
print("\n=== tt_symbiote.compatibility.report(model) ===")
print(json.dumps(report, indent=2))

coverage_path = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
coverage_path.write_text(json.dumps(report, indent=2) + "\n")
print(f"Wrote coverage report to {coverage_path}")

ttnn.close_mesh_device(mesh_device)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

_DOG_EQUIVALENTS = (
    "dog", "puppy", "retriever", "labrador", "poodle",
    "terrier", "spaniel", "shepherd", "husky", "bulldog",
)
_lower = answer.lower()
assert any(term in _lower for term in _DOG_EQUIVALENTS), (
    f"Expected the answer to mention a dog or a dog breed. "
    f"Got: {answer!r}."
)
print("\nOK: answer correctly identifies the animal as a dog.")
