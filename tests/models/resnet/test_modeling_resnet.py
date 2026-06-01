# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Smoke test for ResNet (Phase 6) end-to-end through the new public API.

Mirrors the Phase 5 reference test
(``tests/models/bailing_moe_v2/test_modeling_bailing_moe_v2.py``) but
for the vision side:

1. :class:`tt_symbiote.AutoModelForImageClassification` loads the HF
   model and applies the registered :class:`ResNetRecipe` (single-dict,
   single-pass module replacement covering the six top-level swaps).
2. :func:`tt_symbiote.set_device` binds every TTNN module to
   ``mesh_device``, subsumes the per-module ``preprocess_weights`` /
   ``move_weights_to_device`` loop. ``make_kv_cache`` is the
   ``@register_recipe`` no-op for ResNet (vision has no KV cache), so
   no ``_tt_kv_cache`` is allocated.
3. A single forward pass on a synthetic ``(1, 3, 224, 224)`` tensor
   asserts the logits shape matches the ImageNet head
   (``config.num_labels == 1000`` for the Microsoft checkpoints).

Numerical PCC checks against an unmodified HF model live in DPL run
mode, not in this smoke test.
"""

import os

import pytest
import torch

import ttnn
from tt_symbiote import AutoModelForImageClassification, set_device


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 245760}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_resnet50(mesh_device):
    """ResNet-50 end-to-end through ``AutoModelForImageClassification``."""

    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        torch_dtype=torch.bfloat16,
    )
    set_device(model, mesh_device)
    assert hasattr(
        model, "_tt_runtime_config"
    ), "ResNetRecipe.post_register should have stashed model._tt_runtime_config"

    model.eval()
    torch.set_grad_enabled(False)

    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
    outputs = model(pixel_values=pixel_values)

    assert hasattr(outputs, "logits"), "ImageClassifierOutput should expose `.logits`"
    assert outputs.logits.shape == (
        1,
        model.config.num_labels,
    ), f"Expected logits shape (1, {model.config.num_labels}); got {tuple(outputs.logits.shape)}"
    assert torch.isfinite(outputs.logits).all(), "logits should be all-finite"
