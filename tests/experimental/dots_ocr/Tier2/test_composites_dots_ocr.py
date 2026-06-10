# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 (composite) PCC / structural tests for dots.ocr.

  * MLP (fused SwiGLU gate+up, row-sharded down) vs HF MLP -- decode and
    prefill activation paths.
  * One vision-tower block (norm -> attn(2D RoPE) -> norm -> MLP) structural
    pass, mirroring the proven tt-metal reference vision-block driving.

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier2/test_composites_dots_ocr.py -x -s \
        --override-ini="addopts="
"""

import pytest
import torch

import ttnn
from tt_symbiote.models.dots_ocr import TTNNDotsOCRVisionTower
from tt_symbiote.models.dots_ocr.dots_ocr_mlp import TTNNDotsOCRMLP
from tt_symbiote.utils.device_management import set_device

from ..dots_ocr_helpers import (
    assert_pcc,
    canonical_to_torch,
    dots_ocr_device_params,
    resolve_mesh_device_shape,
    resolve_model_path,
)

DOTS_OCR_LOCAL_PATH = resolve_model_path()


def _hf_model_1layer(vision_blocks=None):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    cfg.num_hidden_layers = 1
    if vision_blocks is not None:
        vc = getattr(cfg, "vision_config", None)
        if vc is not None:
            for attr in ("num_hidden_layers", "num_layers", "depth"):
                if hasattr(vc, attr):
                    setattr(vc, attr, vision_blocks)
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
@pytest.mark.parametrize("seq_len,memcfg_name", [(1, "L1"), (128, "DRAM")], ids=["decode", "prefill"])
def test_dots_ocr_mlp_pcc(mesh_device, seq_len, memcfg_name):
    """TTNNDotsOCRMLP vs HF Qwen2MLP for decode (seq=1) and prefill (seq=128)."""
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    hf_model = _hf_model_1layer()
    hidden_size = hf_model.config.hidden_size
    torch_mlp = hf_model.model.layers[0].mlp

    x = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    torch_out = torch_mlp(x)  # reference BEFORE fuse/transpose weight mutation

    mlp = TTNNDotsOCRMLP.from_torch(torch_mlp)
    mlp._unique_name = "model.layers.0.mlp"
    set_device(mlp, mesh_device)

    memcfg = ttnn.L1_MEMORY_CONFIG if memcfg_name == "L1" else ttnn.DRAM_MEMORY_CONFIG
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=memcfg)
    out_tt = mlp.forward(x_tt)
    ttnn.synchronize_device(mesh_device)

    out_torch = canonical_to_torch(out_tt, mesh_device).to(torch.bfloat16).reshape(torch_out.shape)
    # bf16 + BFP4/BFP8 fused-SwiGLU drift -> 0.98 bar.
    assert_pcc(out_torch, torch_out, threshold=0.98, msg=f"DotsOCRMLP[{memcfg_name}]")


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_vision_block_structural(mesh_device):
    """One dots.ocr vision block: norm -> attn(2D RoPE) -> norm -> MLP, shape gate.

    Uses the production vision-tower perf bucket (grid [1, 88, 128] -> 11264
    patch tokens) that the reference harness exercises, so the tuned vision
    program configs engage exactly as in the full tower.
    """
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    hf_model = _hf_model_1layer(vision_blocks=1)
    blocks = getattr(hf_model.vision_tower, "blocks", getattr(hf_model.vision_tower, "layers", None))
    assert blocks is not None, "dots.ocr vision tower should expose blocks/layers"

    vision_tower = TTNNDotsOCRVisionTower.from_torch(hf_model.vision_tower, hf_model.config)
    vision_tower._unique_name = "vision_tower"
    vision_tower.override_children_module_names()
    set_device(vision_tower, mesh_device)
    assert len(vision_tower.blocks) == 1

    grid_thw = torch.tensor([[1, 88, 128]], dtype=torch.int32)
    seq_len = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])
    hidden_size = int(vision_tower.hidden_size)
    token_shape = [1, 1, seq_len, hidden_size]

    hidden_states = ttnn.from_torch(
        torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    rot_mats, _ = vision_tower.rope.build_padded(grid_thw, seq_len, seq_len)
    out = vision_tower.blocks[0].forward(
        hidden_states,
        rot_mats=rot_mats,
        cu_seqlens=None,
        attention_logical_seq_len=seq_len,
    )
    ttnn.synchronize_device(mesh_device)

    assert isinstance(out, ttnn.Tensor)
    assert list(out.shape) == token_shape, f"vision block output shape: expected {token_shape}, got {list(out.shape)}"
