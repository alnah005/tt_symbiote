"""End-to-end "what is this animal?" demo for ``google/gemma-4-31B-it`` on T3K.

Same VLM recipe and same prompt as :mod:`examples.e2e.run_gemma4_e2b`,
sized for the 31B dense variant on the full T3K (1x8) mesh. Phase 7 is
a **CPU-first port**, so execution runs on the host PyTorch backend
even though a TTNN mesh is opened — the device handle is still used by
the recipe to compute :data:`google/gemma-4-31B-it` -specific
:class:`Gemma4Recipe` runtime config (mesh shape, ``l1_small_size``)
that downstream TTNN wrappers will consume once they land.

Resource notes
--------------

* Weights are ~58 GB in BF16. ``from_pretrained`` downloads once
  (``~/.cache/huggingface/hub/`` will grow by ~60 GB) and then loads
  the full tensor into host RAM. Allocate **at least 80 GB free RAM**
  before running.
* HF hub access to ``google/gemma-4-31B-it`` requires accepting the
  Gemma license once via the model card. ``from_pretrained`` raises a
  clear ``GatedRepoError`` if the token has not accepted it.
* CPU forward of a 31B dense transformer is slow: expect tens of
  seconds to minutes per token on a typical workstation host. The
  64-token generation limit below is a soft cap; the model's EOS
  token usually stops generation much earlier.

This file is the Phase 7 reference reproducer for the 31B variant.
The smaller E2B variant (:mod:`run_gemma4_e2b`) is the recommended
first run; this one validates that the same recipe scales without
modification.

Usage::

    source .venv/bin/activate        # see scripts/bootstrap_venv.sh
    python examples/e2e/gemma4/run_gemma4_31b.py
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

MODEL_ID = "google/gemma-4-31B-it"

IMAGE_PATH = Path(__file__).resolve().parents[3] / "tests" / "images" / "test-dog.png"
PROMPT = "What is this animal in the photo?"


if not IMAGE_PATH.exists():
    # The published repo intentionally ships no binary image (license);
    # see tests/images/.gitignore. Drop any Pillow-readable picture of
    # a dog at IMAGE_PATH (e.g. `cp ~/Pictures/dog.jpg tests/images/test-dog.png`)
    # and rerun. The dog-equivalents assertion at the bottom of this
    # script expects the answer to mention a dog or a dog breed.
    import sys

    print(
        f"SKIP: {__file__} needs an input image at {IMAGE_PATH}. "
        f"The published repo does not ship one. See tests/images/ "
        f"(gitignored) and examples/e2e/README.md for details."
    )
    sys.exit(0)
# Even on the CPU-first path we open the full T3K mesh: the recipe's
# ``post_register`` reads the (1, 8) ``mesh_shape`` from
# :data:`GEMMA4_TTNN_TUNING` and attaches it as ``model._tt_runtime_config``
# so the next-phase TTNN port can immediately consume it.
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
assert hasattr(model, "_tt_runtime_config"), "Gemma4Recipe.post_register should have attached _tt_runtime_config"
assert model._tt_runtime_config["mesh_shape"] == (
    1,
    8,
), f"31B should target the full T3K mesh; got {model._tt_runtime_config['mesh_shape']}"

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
print(f"Gemma-4 31B answer: {answer!r}")

report = compatibility.report(model)
print("\n=== tt_symbiote.compatibility.report(model) ===")
print(json.dumps(report, indent=2))

coverage_path = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
coverage_path.write_text(json.dumps(report, indent=2) + "\n")
print(f"Wrote coverage report to {coverage_path}")

ttnn.close_mesh_device(mesh_device)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

_DOG_EQUIVALENTS = (
    "dog",
    "puppy",
    "retriever",
    "labrador",
    "poodle",
    "terrier",
    "spaniel",
    "shepherd",
    "husky",
    "bulldog",
)
_lower = answer.lower()
assert any(term in _lower for term in _DOG_EQUIVALENTS), (
    f"Expected the answer to mention a dog or a dog breed. " f"Got: {answer!r}."
)
print("\nOK: answer correctly identifies the animal as a dog.")
