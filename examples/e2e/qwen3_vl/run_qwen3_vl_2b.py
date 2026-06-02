"""End-to-end "what is this animal?" demo for ``Qwen/Qwen3-VL-2B-Instruct``.

Loads Qwen3-VL-2B-Instruct through ``tt_symbiote.AutoModelForImageTextToText``,
binds a single-chip TTNN mesh via :func:`set_device`, runs a single
``model.generate`` call against an image + text prompt, and asserts the
decoded answer contains ``"dog"``.

This script is the hardware acceptance gate for the first model ported
via the ``port-hf-model-to-tt-symbiote`` Cursor skill. It mirrors
:mod:`examples.e2e.run_gemma4_e2b` exactly — same image, same prompt,
same semantic assertion — so a regression in *either* the skill output
*or* the underlying ``tt_symbiote`` plumbing shows up immediately by
diffing this script against the Gemma-4 one.

Phase 7 (first Qwen3-VL commit) ships a **CPU-first port**: the
:class:`Qwen3VLRecipe`'s ``build_module_dict`` is intentionally empty,
so all submodules (vision tower, text decoder, top-level composites)
run as the upstream HuggingFace reference implementation on the CPU.
The mesh device is still opened and ``set_device`` still runs — both
because they exercise the documented user-facing API and because
:meth:`Qwen3VLRecipe.post_register` uses the device handle to compute
the per-variant TTNN runtime config (``mesh_shape``, ``l1_small_size``,
etc.) that downstream TTNN wrappers will consume once they land. The
compatibility report printed at the end quantifies exactly which
submodules are still on CPU.

Usage::

    source .venv/bin/activate        # see scripts/bootstrap_venv.sh
    python examples/e2e/qwen3_vl/run_qwen3_vl_2b.py
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

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# The repo-anchored path keeps the script runnable from any CWD: the
# image lives at ``tests/images/test-dog.png`` and this script lives at
# ``examples/e2e/qwen3_vl/`` (two levels deeper, hence ``parents[3]``).
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

# CPU-first phase: ``set_device`` walks the model (no TTNN modules to
# bind), attaches ``model._tt_runtime_config``, and is otherwise a no-op
# for execution. Disable the graph viz — it would render a 28-layer
# decoder + vision tower in one PNG, which is noisy for this run.
set_device(model, mesh_device, dump_visualization=False)
assert hasattr(model, "_tt_runtime_config"), "Qwen3VLRecipe.post_register should have attached _tt_runtime_config"

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
# Keep tensors on CPU — that's where the weights live in the CPU-first port.
inputs = {k: (v.to("cpu") if hasattr(v, "to") else v) for k, v in inputs.items()}

# Deterministic generation so the assertion is reproducible. 64 tokens
# is plenty: a typical answer for "What is this animal?" is well under
# 30 tokens and the EOS token terminates earlier.
out = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    use_cache=True,
)

prompt_len = inputs["input_ids"].shape[-1]
answer = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()
print(f"Qwen3-VL-2B-Instruct answer: {answer!r}")

# Compatibility report drives the docs/supported_models.md "TT-implemented vs
# CPU" status column and the per-script row in docs/development/cpu_vs_device_coverage.md.
# Phase 8.5: any class in ``regressions`` is an actionable signal — it means
# a TTNN wrapper hit the fallback path on a class the recipe expected to
# succeed on. ``modules_swapped`` shows what actually ran on device.
report = compatibility.report(model)
print("\n=== tt_symbiote.compatibility.report(model) ===")
print(json.dumps(report, indent=2))

coverage_path = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
coverage_path.write_text(json.dumps(report, indent=2) + "\n")
print(f"Wrote coverage report to {coverage_path}")

ttnn.close_mesh_device(mesh_device)

# Semantic check: Qwen3-VL-2B-Instruct often skips the generic word
# "dog" and jumps straight to a breed (e.g. "Golden Retriever puppy"
# / "Labrador puppy"). Any of these terms means the model correctly
# identified the animal as a dog — see reference-vlm.md "Semantic
# check" notes. The previous Gemma-4 demo got away with a strict
# ``"dog" in answer`` because that model is more verbose; Qwen3-VL is
# terser.
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
    f"Expected the answer to mention a dog or a dog breed (the photo "
    f"shows a dog). Got: {answer!r}. Re-check the image path "
    f"({IMAGE_PATH}) and the chat-template formatting."
)
print(f"\nOK: answer correctly identifies the animal as a dog.")
