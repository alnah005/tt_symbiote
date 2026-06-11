# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier3 (decoder/encoder-layer) PCC test for diffusion_gemma.

Exercises the full dual-FFN sandwich layer against its HF torch reference:
  input_layernorm -> self_attn -> post_attention_layernorm -> + residual ;
  then a MLP branch (pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm_1)
  and a MoE branch (pre_feedforward_layernorm_2 -> router/experts -> post_feedforward_layernorm_2)
  from the post-attn residual, summed -> post_feedforward_layernorm -> + residual -> * layer_scalar.

Validated on P150x4 (pcc 0.998 at reduced expert dims). To keep the MoE fast the
expert dims are reduced while hidden/heads/head_dim/rope stay REAL (so attention
runs normally) -- the composition logic (norms, residuals, MLP+MoE combine,
layer_scalar) is what this tier validates; the individual modules are validated
full-scale in Tier2.

Device: P150x4 mesh via the ttnn-plugin ``mesh_device`` fixture.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
import ttnn

from tests.shared.pcc_utils import assert_pcc
from tt_symbiote.models.diffusion_gemma.modeling_diffusion_gemma import (
    TTNNDiffusionGemmaDecoderTextLayer,
    TTNNDiffusionGemmaEncoderTextLayer,
)
from tt_symbiote.utils.device_management import set_device

_SHAPES = json.loads((Path(__file__).parent.parent / "shapes.json").read_text())
_CFG = _SHAPES["model_config"]
_HF_ID = "google/diffusiongemma-26B-A4B-it"
_DEVICE_PARAMS = [{"mesh_shape": (1, 4)}]
_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "post_feedforward_layernorm_1",
    "post_feedforward_layernorm_2",
    "pre_feedforward_layernorm_2",
)


def _from(t, md):
    if t.dtype == torch.float32:
        t = t.to(torch.bfloat16)
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
    )


def _to(t, md, ref_shape):
    g = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()
    n = 1
    for d in ref_shape:
        n *= d
    return g.flatten()[:n].reshape(ref_shape)


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "cls,hf_name",
    [
        (TTNNDiffusionGemmaEncoderTextLayer, "DiffusionGemmaEncoderTextLayer"),
        (TTNNDiffusionGemmaDecoderTextLayer, "DiffusionGemmaDecoderTextLayer"),
    ],
)
def test_text_layer(mesh_device, pcc_threshold, cls, hf_name):
    """Full layer composition (reduced expert dims; real attention/rope)."""
    from transformers import AutoConfig
    from transformers.models.diffusion_gemma import modeling_diffusion_gemma as M
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaTextRotaryEmbedding,
    )

    tc = copy.deepcopy(AutoConfig.from_pretrained(_HF_ID).text_config)
    tc.num_experts = 8
    tc.moe_intermediate_size = 64  # fast MoE; hidden/heads/rope stay real
    H, S, li = tc.hidden_size, 32, 0  # layer 0 = sliding_attention

    torch.manual_seed(5)
    ref_layer = getattr(M, hf_name)(tc, layer_idx=li).eval()
    with torch.no_grad():
        ref_layer.experts.gate_up_proj.normal_(0, 0.02)
        ref_layer.experts.down_proj.normal_(0, 0.02)
        ref_layer.router.proj.weight.normal_(0, 1.0)
        ref_layer.self_attn.q_norm.weight.copy_(torch.randn(ref_layer.self_attn.head_dim) * 0.1 + 1)
        ref_layer.self_attn.k_norm.weight.copy_(torch.randn(ref_layer.self_attn.head_dim) * 0.1 + 1)
        for nm in _NORMS:
            getattr(ref_layer, nm).weight.copy_(torch.randn(H) * 0.05 + 1)
        ref_layer.layer_scalar.copy_(torch.tensor([1.3]))

    hidden = torch.randn(1, S, H)
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = DiffusionGemmaTextRotaryEmbedding(tc)(hidden, pos, layer_type=tc.layer_types[li])
    # Layer attention is is_causal=True (matches the full model); HF eager adds the mask.
    causal = torch.triu(torch.full((1, 1, S, S), float("-inf")), diagonal=1)
    ref = ref_layer(hidden, position_embeddings=(cos, sin), attention_mask=causal)

    tt = cls.from_torch(ref_layer)
    set_device(tt, mesh_device)
    out = tt.forward(_from(hidden, mesh_device), (_from(cos, mesh_device), _from(sin, mesh_device)))
    got = _to(out, mesh_device, ref.shape)
    assert_pcc(got, ref, threshold=pcc_threshold, msg=hf_name)
