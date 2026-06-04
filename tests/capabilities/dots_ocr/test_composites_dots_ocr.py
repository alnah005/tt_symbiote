# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 composite-module PCC tests for rednote-hilab/dots.ocr.

Composites are exercised by building the real PyTorch module (Qwen2MLP for the
text backbone; DotsSwiGLUFFN and PatchMerger for the dots_vit tower), running
its torch forward, then swapping leaf modules (Linear / SiLU / GELU / LayerNorm)
via ``register_modules`` and re-running. PCC is checked against the torch
reference.

Attention composites are intentionally NOT exercised at this tier -- no
Qwen2-specific or dots_vit-specific TTNN attention integration exists yet;
attention coverage shows up at the decoder-layer tier (Tier 3) where the
inner Linears are swapped in place.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import json
import pathlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.modules.ttnn_activation import TTNNGelu, TTNNSilu
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.modules.ttnn_normalization import TTNNLayerNorm
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules

from tests.capabilities.pcc_utils import assert_pcc

_SHAPES_PATH = pathlib.Path(__file__).parent / "shapes.json"
_SHAPES = json.loads(_SHAPES_PATH.read_text())


def _vision_config() -> SimpleNamespace:
    """Lightweight stand-in for DotsVisionConfig with the real model dims."""
    v = _SHAPES["vision_config"]
    return SimpleNamespace(
        embed_dim=v["embed_dim"],
        hidden_size=v["hidden_size"],
        intermediate_size=v["intermediate_size"],
        num_attention_heads=v["num_attention_heads"],
        num_channels=v["num_channels"],
        patch_size=v["patch_size"],
        temporal_patch_size=v["temporal_patch_size"],
        spatial_merge_size=v["spatial_merge_size"],
        rms_norm_eps=v["rms_norm_eps"],
        use_bias=v["use_bias"],
        post_norm=v["post_norm"],
        init_merger_std=None,  # Skip explicit init -- random weights are fine for PCC.
    )


def _text_config():
    """Stripped Qwen2-style config built from shapes.json (avoids HF checkpoint download)."""
    t = _SHAPES["config"]
    return SimpleNamespace(
        hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"],
        hidden_act=t["hidden_act"],
        mlp_bias=False,
    )


def _materialize_ttnn_modules(modules):
    for _, mod in tqdm(modules.items(), desc="ttnn modules"):
        mod.preprocess_weights()
        mod.move_weights_to_device()


# ---------------------------------------------------------------------------
# Text backbone: Qwen2MLP (gate_proj / up_proj / down_proj + SiLU)
# ---------------------------------------------------------------------------


def test_text_qwen2_mlp(mesh_device):
    """Qwen2MLP at dots.ocr text dims: hidden=1536, intermediate=8960."""
    cfg = _text_config()
    mlp = Qwen2MLP(cfg).to(torch.bfloat16)
    mlp.eval()
    torch.set_grad_enabled(False)

    inputs = TorchTTNNTensor(torch.randn(1, 128, cfg.hidden_size, dtype=torch.bfloat16))
    torch_out = mlp(inputs)

    swap_map = {nn.Linear: TTNNLinear, nn.SiLU: TTNNSilu}
    modules = register_modules(mlp, swap_map, model_config=None)
    set_device(mlp, mesh_device)
    _materialize_ttnn_modules(modules)

    ttnn_out = mlp(inputs)
    assert_pcc(ttnn_out, torch_out, threshold=0.99, msg="text.Qwen2MLP")


# ---------------------------------------------------------------------------
# Vision tower: DotsSwiGLUFFN (fc1, fc3, F.silu, fc2)
# ---------------------------------------------------------------------------


def test_vision_swiglu_ffn(mesh_device):
    """DotsSwiGLUFFN at dots_vit dims: embed=1536, intermediate=4224.

    NOTE: The original ``DotsSwiGLUFFN.forward`` uses ``F.silu`` (functional),
    which ``register_modules`` cannot swap. The Linears are swapped; the SiLU
    activation stays in torch. PCC should still meet the composite threshold
    because the float SiLU is exact between the two paths.
    """
    pytest.importorskip("transformers")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    DotsSwiGLUFFN = get_class_from_dynamic_module(
        "modeling_dots_vision.DotsSwiGLUFFN", "rednote-hilab/dots.ocr"
    )
    cfg = _vision_config()
    ffn = DotsSwiGLUFFN(cfg).to(torch.bfloat16)
    ffn.eval()
    torch.set_grad_enabled(False)

    inputs = TorchTTNNTensor(torch.randn(256, cfg.embed_dim, dtype=torch.bfloat16))
    torch_out = ffn(inputs)

    swap_map = {nn.Linear: TTNNLinear}
    modules = register_modules(ffn, swap_map, model_config=None)
    set_device(ffn, mesh_device)
    _materialize_ttnn_modules(modules)

    ttnn_out = ffn(inputs)
    assert_pcc(ttnn_out, torch_out, threshold=0.99, msg="vision.DotsSwiGLUFFN")


# ---------------------------------------------------------------------------
# Vision tower: PatchMerger (LayerNorm + nn.Linear + nn.GELU + nn.Linear)
# ---------------------------------------------------------------------------


def test_vision_patch_merger(mesh_device):
    """PatchMerger at dots_vit dims: context_dim=1536, hidden_size=6144, dim=1536."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    PatchMerger = get_class_from_dynamic_module(
        "modeling_dots_vision.PatchMerger", "rednote-hilab/dots.ocr"
    )
    v = _SHAPES["vision_config"]
    merger = PatchMerger(
        dim=v["hidden_size"],
        context_dim=v["embed_dim"],
        spatial_merge_size=v["spatial_merge_size"],
        pre_norm="layernorm",
    ).to(torch.bfloat16)
    merger.eval()
    torch.set_grad_enabled(False)

    # PatchMerger flattens via .view(-1, hidden_size) in its forward, so the
    # input is shaped as a stack of (spatial_merge_size ** 2) patches.
    sm = v["spatial_merge_size"]
    n_tiles = 64
    seq_len = n_tiles * (sm * sm)
    inputs = TorchTTNNTensor(torch.randn(seq_len, v["embed_dim"], dtype=torch.bfloat16))
    torch_out = merger(inputs)

    swap_map = {nn.Linear: TTNNLinear, nn.GELU: TTNNGelu, nn.LayerNorm: TTNNLayerNorm}
    modules = register_modules(merger, swap_map, model_config=None)
    set_device(merger, mesh_device)
    _materialize_ttnn_modules(modules)

    ttnn_out = merger(inputs)
    assert_pcc(ttnn_out, torch_out, threshold=0.99, msg="vision.PatchMerger")
