"""End-to-end smoke run for ``microsoft/resnet-34`` on a Tenstorrent device.

Sibling of :mod:`examples.e2e.resnet.run_resnet50`. ResNet-34 still uses
the two-conv basic block (:class:`ResNetBasicLayer`) — same as
ResNet-18 but with deeper per-stage repetition counts. The recipe is
shared.

Structurally supported but not yet hardware-verified; flip
``hw_verified`` to ``True`` for ``microsoft/resnet-34`` in
:data:`RESNET_TTNN_TUNING` after a successful run here.

Usage::

    source .venv/bin/activate        # see scripts/bootstrap_venv.sh
    python examples/e2e/resnet/run_resnet34.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MESH_DEVICE", "N150")

import torch
import ttnn

from tt_symbiote import AutoModelForImageClassification, compatibility, set_device

MODEL_ID = "microsoft/resnet-34"

ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    trace_region_size=200_000_000,
    num_command_queues=1,
    l1_small_size=245760,
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
set_device(model, mesh_device)
assert hasattr(model, "_tt_runtime_config"), "ResNet recipe should have attached this"

model.eval()
torch.set_grad_enabled(False)

pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)

outputs = model(pixel_values=pixel_values)
logits = outputs.logits
predicted_class_idx = int(logits.argmax(dim=-1).item())
label = model.config.id2label.get(predicted_class_idx, f"<class {predicted_class_idx}>")
print(f"ResNet-34 top-1: class {predicted_class_idx} = {label!r}")

report = compatibility.report(model)
print("\n=== tt_symbiote.compatibility.report(model) ===")
print(json.dumps(report, indent=2))

coverage_path = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
coverage_path.write_text(json.dumps(report, indent=2) + "\n")
print(f"Wrote coverage report to {coverage_path}")

ttnn.close_mesh_device(mesh_device)
