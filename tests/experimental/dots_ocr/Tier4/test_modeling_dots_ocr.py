# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 full-model PCC tests for rednote-hilab/dots.ocr.

Two test entry points are provided:

* ``test_dots_vision_transformer_smoke``: instantiates the dots_vit tower with
  random weights at the released dims and runs one forward through the full
  encoder + post-norm + PatchMerger, then swaps leaf modules to TTNN and
  re-runs. Runs without downloading the multi-GB checkpoint.

* ``test_dots_ocr_forcausallm_full``: full multimodal forward through
  DotsOCRForCausalLM. Skip-marked by default -- needs the real checkpoint,
  tokenizer, processor, and an actual image grid to exercise the cross-modal
  path. The scaffolding is left in place for the bring-up driver to enable
  once checkpoints land.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import json
import pathlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tqdm import tqdm

from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.modules.ttnn_activation import TTNNGelu
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.modules.ttnn_normalization import TTNNLayerNorm, TTNNLocalRMSNorm
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules

from tests.shared.pcc_utils import assert_pcc

_SHAPES_PATH = pathlib.Path(__file__).parent.parent / "shapes.json"
_SHAPES = json.loads(_SHAPES_PATH.read_text())


def _vision_config_ns():
    v = _SHAPES["vision_config"]
    return SimpleNamespace(
        embed_dim=v["embed_dim"],
        hidden_size=v["hidden_size"],
        intermediate_size=v["intermediate_size"],
        num_hidden_layers=v["num_hidden_layers"],
        num_attention_heads=v["num_attention_heads"],
        num_channels=v["num_channels"],
        patch_size=v["patch_size"],
        temporal_patch_size=v["temporal_patch_size"],
        spatial_merge_size=v["spatial_merge_size"],
        rms_norm_eps=v["rms_norm_eps"],
        use_bias=v["use_bias"],
        post_norm=v["post_norm"],
        init_merger_std=None,
        initializer_range=0.02,
        attn_implementation="sdpa",
        is_causal=False,
        gradient_checkpointing=False,
    )


def _materialize(modules):
    for _, mod in tqdm(modules.items(), desc="ttnn modules"):
        mod.preprocess_weights()
        mod.move_weights_to_device()


# ---------------------------------------------------------------------------
# Vision tower smoke (no checkpoint required)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dots_vision_transformer_smoke(mesh_device):
    """End-to-end vision tower with random weights at dots_vit dims.

    Uses one small grid (t=1, h=14, w=14) so the patch sequence stays bounded.
    The block count is reduced to keep CI runtime reasonable -- the full
    42-layer tower is exercised via the per-block Tier 3 test.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    DotsVisionTransformer = get_class_from_dynamic_module(
        "modeling_dots_vision.DotsVisionTransformer", "rednote-hilab/dots.ocr"
    )
    DotsVisionRMSNorm = get_class_from_dynamic_module("modeling_dots_vision.RMSNorm", "rednote-hilab/dots.ocr")

    cfg = _vision_config_ns()
    cfg.num_hidden_layers = 2  # Smoke test: 2 layers instead of 42
    vision = DotsVisionTransformer(cfg).to(torch.bfloat16)
    vision.eval()
    torch.set_grad_enabled(False)

    # Grid: 1 frame, 14x14 patches. patch_size=14 means image is 196x196 px.
    grid_thw = torch.tensor([[1, 14, 14]], dtype=torch.int64)
    n_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])
    pixel_values = TorchTTNNTensor(
        torch.randn(
            n_patches,
            cfg.num_channels,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
            dtype=torch.bfloat16,
        )
    )

    torch_out = vision(pixel_values, grid_thw)

    swap_map = {
        nn.Linear: TTNNLinear,
        nn.GELU: TTNNGelu,
        nn.LayerNorm: TTNNLayerNorm,
        DotsVisionRMSNorm: TTNNLocalRMSNorm,
    }
    modules = register_modules(vision, swap_map, model_config=None)
    set_device(vision, mesh_device)
    _materialize(modules)

    ttnn_out = vision(pixel_values, grid_thw)
    assert_pcc(ttnn_out, torch_out, threshold=0.97, msg="vision.DotsVisionTransformer")


# ---------------------------------------------------------------------------
# Full multimodal model (skip-marked until checkpoint + image fixtures land)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Full DotsOCRForCausalLM forward requires the real ~3B checkpoint "
        "plus image inputs (pixel_values, image_grid_thw, img_mask). Enable "
        "from the model-bringup driver once the multimodal fixture is wired up."
    )
)
def test_dots_ocr_forcausallm_full(mesh_device):
    """End-to-end multimodal forward (currently skip-marked).

    When enabling, follow this pattern (mirrors tests/experimental/olmo3):

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "rednote-hilab/dots.ocr", trust_remote_code=True
        ).to(torch.bfloat16)

        swap_map = {nn.Linear: TTNNLinear, nn.SiLU: TTNNSilu, ...}
        modules = register_modules(model, swap_map, model_config=None)
        set_device(model, mesh_device)
        for _, mod in modules.items():
            mod.preprocess_weights()
            mod.move_weights_to_device()

        outputs = model(input_ids=..., pixel_values=..., image_grid_thw=...)
    """
    pytest.skip("multimodal fixture not yet wired up")
