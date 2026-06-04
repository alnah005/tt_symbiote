# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 op-level PCC tests for rednote-hilab/dots.ocr (model_type=dots_ocr).

Each test instantiates a single leaf module with the real dots.ocr dimensions
(read from shapes.json), runs PyTorch reference and TTNN forward on the same
input, and asserts PCC >= the bring-up threshold. Tests use synthetic weights
to avoid downloading the full checkpoint; the dims match the released model.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import json
import math
import pathlib

import pytest
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.modules.ttnn_activation import TTNNGelu, TTNNSilu
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.modules.ttnn_normalization import (
    TTNNLayerNorm,
    TTNNLocalRMSNorm,
    TTNNRMSNorm,
)
from tt_symbiote.utils.device_management import set_device

from tests.capabilities.pcc_utils import assert_pcc

_SHAPES_PATH = pathlib.Path(__file__).parent / "shapes.json"
_SHAPES = json.loads(_SHAPES_PATH.read_text())
_TEXT = _SHAPES["module_shapes"]["text"]
_VISION = _SHAPES["module_shapes"]["vision"]


class _DotsVisionRMSNorm(nn.Module):
    """Reproduction of modeling_dots_vision.RMSNorm (exposes ``eps``, not ``variance_epsilon``)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match the dots vision norm exactly: cast to float for the rsqrt, scale by weight.
        output = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return output * self.weight


def _make_input(shape, dtype=torch.bfloat16):
    return TorchTTNNTensor(torch.randn(*shape, dtype=dtype))


# ---------------------------------------------------------------------------
# Linear projections (text backbone -- Qwen2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
def test_text_linear(mesh_device, name):
    """PCC test for each text-backbone Linear projection at the real model dims."""
    spec = _TEXT[name]
    in_features = spec["weight"][1]
    out_features = spec["weight"][0]
    has_bias = spec["bias"] is not None

    torch_linear = nn.Linear(in_features, out_features, bias=has_bias).to(torch.bfloat16)
    torch_linear.eval()
    torch.set_grad_enabled(False)

    inputs = _make_input(spec["input"])
    torch_out = torch_linear(inputs)

    ttnn_linear = TTNNLinear.from_torch(torch_linear)
    set_device(ttnn_linear, mesh_device)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()
    ttnn_out = ttnn_linear(inputs)

    assert_pcc(ttnn_out, torch_out, threshold=0.999, msg=f"text.{name}")


# ---------------------------------------------------------------------------
# Linear projections (vision tower -- dots_vit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["qkv", "proj", "fc1", "fc2", "fc3", "merger_mlp_0", "merger_mlp_2"],
)
def test_vision_linear(mesh_device, name):
    """PCC test for each vision-tower Linear at the real model dims."""
    spec = _VISION[name]
    in_features = spec["weight"][1]
    out_features = spec["weight"][0]
    has_bias = spec["bias"] is not None

    torch_linear = nn.Linear(in_features, out_features, bias=has_bias).to(torch.bfloat16)
    torch_linear.eval()
    torch.set_grad_enabled(False)

    inputs = _make_input(spec["input"])
    torch_out = torch_linear(inputs)

    ttnn_linear = TTNNLinear.from_torch(torch_linear)
    set_device(ttnn_linear, mesh_device)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()
    ttnn_out = ttnn_linear(inputs)

    assert_pcc(ttnn_out, torch_out, threshold=0.999, msg=f"vision.{name}")


# ---------------------------------------------------------------------------
# RMSNorm -- text backbone (Qwen2RMSNorm uses ``variance_epsilon``)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["input_layernorm", "post_attention_layernorm", "norm"],
)
def test_text_rms_norm(mesh_device, name):
    spec = _TEXT[name]
    hidden_size = spec["weight"][0]
    eps = spec["eps"]

    torch_norm = Qwen2RMSNorm(hidden_size, eps=eps).to(torch.bfloat16)
    torch_norm.weight.data = torch.randn_like(torch_norm.weight.data)
    torch_norm.eval()
    torch.set_grad_enabled(False)

    inputs = _make_input(spec["input"])
    torch_out = torch_norm(inputs)

    ttnn_norm = TTNNRMSNorm.from_torch(torch_norm)
    set_device(ttnn_norm, mesh_device)
    ttnn_norm.preprocess_weights()
    ttnn_norm.move_weights_to_device()
    ttnn_out = ttnn_norm(inputs)

    assert_pcc(ttnn_out, torch_out, threshold=0.999, msg=f"text.{name}")


# ---------------------------------------------------------------------------
# RMSNorm -- vision tower (exposes ``eps``, not ``variance_epsilon``)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["norm1", "norm2", "patch_embed_norm"],
)
def test_vision_rms_norm(mesh_device, name):
    spec = _VISION[name]
    hidden_size = spec["weight"][0]
    eps = spec["eps"]

    torch_norm = _DotsVisionRMSNorm(hidden_size, eps=eps).to(torch.bfloat16)
    torch_norm.weight.data = torch.randn_like(torch_norm.weight.data)
    torch_norm.eval()
    torch.set_grad_enabled(False)

    inputs = _make_input(spec["input"])
    torch_out = torch_norm(inputs)

    ttnn_norm = TTNNLocalRMSNorm.from_torch(torch_norm)
    set_device(ttnn_norm, mesh_device)
    ttnn_norm.preprocess_weights()
    ttnn_norm.move_weights_to_device()
    ttnn_out = ttnn_norm(inputs)

    assert_pcc(ttnn_out, torch_out, threshold=0.99, msg=f"vision.{name}")


# ---------------------------------------------------------------------------
# LayerNorm -- vision PatchMerger.ln_q
# ---------------------------------------------------------------------------


def test_vision_layer_norm(mesh_device):
    spec = _VISION["ln_q"]
    hidden_size = spec["weight"][0]

    torch_norm = nn.LayerNorm(hidden_size, eps=1e-6).to(torch.bfloat16)
    torch_norm.weight.data = torch.randn_like(torch_norm.weight.data)
    torch_norm.bias.data = torch.randn_like(torch_norm.bias.data)
    torch_norm.eval()
    torch.set_grad_enabled(False)

    inputs = _make_input(spec["input"])
    torch_out = torch_norm(inputs)

    ttnn_norm = TTNNLayerNorm.from_torch(torch_norm)
    set_device(ttnn_norm, mesh_device)
    ttnn_norm.preprocess_weights()
    ttnn_norm.move_weights_to_device()
    ttnn_out = ttnn_norm(inputs)

    assert_pcc(ttnn_out, torch_out, threshold=0.999, msg="vision.ln_q")


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def test_silu_activation(mesh_device):
    """TTNNSilu applied on a shape matching Qwen2MLP intermediate."""
    spec = _TEXT["gate_proj"]  # output shape after gate_proj: [1, 128, 8960]
    out_shape = [spec["input"][0], spec["input"][1], spec["weight"][0]]

    torch_silu = nn.SiLU().to(torch.bfloat16)
    torch.set_grad_enabled(False)
    inputs = _make_input(out_shape)
    torch_out = torch_silu(inputs)

    ttnn_silu = TTNNSilu()
    set_device(ttnn_silu, mesh_device)
    ttnn_out = ttnn_silu(inputs)

    assert_pcc(ttnn_out, torch_out, threshold=0.999, msg="text.mlp.act_fn")


def test_gelu_activation(mesh_device):
    """TTNNGelu applied on a shape matching PatchMerger MLP intermediate."""
    spec = _VISION["merger_mlp_0"]
    out_shape = [spec["input"][0], spec["weight"][0]]

    torch_gelu = nn.GELU().to(torch.bfloat16)
    torch.set_grad_enabled(False)
    inputs = _make_input(out_shape)
    torch_out = torch_gelu(inputs)

    ttnn_gelu = TTNNGelu()
    set_device(ttnn_gelu, mesh_device)
    ttnn_out = ttnn_gelu(inputs)

    assert_pcc(ttnn_out, torch_out, threshold=0.999, msg="vision.merger.mlp.gelu")
