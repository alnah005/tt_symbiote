"""End-to-end smoke run for ``microsoft/resnet-50`` on a Tenstorrent device.

Loads ResNet-50 through ``tt_symbiote.AutoModelForImageClassification``,
binds the network to a single-chip mesh via :func:`set_device`, and runs
a forward pass on a synthetic 224x224 image. The top-1 ImageNet label is
printed (random noise gives an arbitrary-but-valid class; for semantic
verification swap in a real image via the ``AutoImageProcessor`` path
shown below).

This is the Phase 6 reference reproducer for the vision side, mirroring
:mod:`examples.e2e.run_ling_mini_2_0` (Phase 5, LLM side). One file per
model; verified on hardware (zero TTNN fallback warnings; top-1 = 'tiger
cat' on the canonical COCO val cat image) before landing in the
tracking table at ``examples/e2e/README.md``.

Usage::

    source .venv/bin/activate        # see scripts/bootstrap_venv.sh
    python examples/e2e/run_resnet50.py
"""

import os
os.environ.setdefault("MESH_DEVICE", "N150")  # single-chip; works on T3K too

import torch
import ttnn
from tt_symbiote import AutoModelForImageClassification, set_device


# Fabric config must be set BEFORE open_mesh_device in current tt-metal HEAD;
# ResNet runs on a single chip so we use the DISABLED fabric mode (no fabric
# routing needed when there's only one device in the mesh).
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),  # single chip
    trace_region_size=200_000_000,
    num_command_queues=1,
    # The 7x7 stride-2 stem conv on a 224x224 input needs L1 small-region
    # scratch for sliding-window halo data; matches the legacy
    # tt-metal test_resnet50.py setting (`device_params={'l1_small_size':
    # 245760}`).
    l1_small_size=245760,
)

# `torch_dtype=torch.bfloat16` matches the TTNN convs' native dtype so weight
# preprocessing in `set_device` doesn't have to downcast on the host.
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    torch_dtype=torch.bfloat16,
)
set_device(model, mesh_device)
assert hasattr(model, "_tt_runtime_config"), "Phase 6 recipe should have attached this"

model.eval()
torch.set_grad_enabled(False)

# Synthetic input keeps the script offline-runnable. To use a real image:
#
#     from transformers import AutoImageProcessor
#     from PIL import Image
#     processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#     image = Image.open("path/to/photo.jpg")
#     inputs = processor(image, return_tensors="pt")
#     pixel_values = inputs["pixel_values"].to(torch.bfloat16)
pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)

outputs = model(pixel_values=pixel_values)
logits = outputs.logits
predicted_class_idx = int(logits.argmax(dim=-1).item())
label = model.config.id2label.get(predicted_class_idx, f"<class {predicted_class_idx}>")
print(f"ResNet-50 top-1: class {predicted_class_idx} = {label!r}")

# Match tt-metal's pytest `mesh_device` fixture teardown.
ttnn.close_mesh_device(mesh_device)
