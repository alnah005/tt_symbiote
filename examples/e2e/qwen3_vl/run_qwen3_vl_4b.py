"""End-to-end "what is this animal?" demo for ``Qwen/Qwen3-VL-4B-Instruct`` on N150.

Sibling of :mod:`examples.e2e.qwen3_vl.run_qwen3_vl_2b` for the next-size-up
Qwen3-VL dense variant. Same recipe (:class:`Qwen3VLRecipe`), same chat
template, same image + prompt, same semantic check.

Resource notes
--------------

* Weights are ~8 GB in BF16. ``from_pretrained`` downloads on first
  run. Allocate **at least 12 GB free RAM** before running.
* Mesh shape is ``(1, 1)`` — the 4B variant still fits comfortably on
  a single N150 chip in the CPU-first port.

This script is structurally supported (recipe is variant-agnostic) but
not yet hardware-verified; the smaller 2B variant
(:mod:`run_qwen3_vl_2b`) is the recommended first run.

Usage::

    source .venv/bin/activate        # see scripts/bootstrap_venv.sh
    python examples/e2e/qwen3_vl/run_qwen3_vl_4b.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MESH_DEVICE", "N150")

import torch  # noqa: E402
import ttnn  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from tt_symbiote import AutoModelForImageTextToText, compatibility, set_device  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

IMAGE_PATH = Path(__file__).resolve().parents[3] / "tests" / "images" / "test-dog.png"
PROMPT = "What is this animal in the photo?"

ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    trace_region_size=200_000_000,
    num_command_queues=1,
    l1_small_size=245760,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
)

set_device(model, mesh_device, dump_visualization=False)
assert hasattr(model, "_tt_runtime_config"), (
    "Qwen3VLRecipe.post_register should have attached _tt_runtime_config"
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
print(f"Qwen3-VL-4B-Instruct answer: {answer!r}")

report = compatibility.report(model)
print("\n=== tt_symbiote.compatibility.report(model) ===")
print(json.dumps(report, indent=2))

coverage_path = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
coverage_path.write_text(json.dumps(report, indent=2) + "\n")
print(f"Wrote coverage report to {coverage_path}")

ttnn.close_mesh_device(mesh_device)

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
