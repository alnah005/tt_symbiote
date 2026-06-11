# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier2 (composite-module) PCC tests for diffusion_gemma.

Exercises the implemented TTNN module classes against their HF torch reference
modules (random-initialized -- we test op composition, not trained weights). All
validated on P150x4 Blackhole hardware (see profiling/validate_tier12.py +
_probe_attn.py + _probe_moe.py):
  * TTNNDiffusionGemmaRMSNorm    <- DiffusionGemmaRMSNorm (with_scale True & False)
  * TTNNDiffusionGemmaTextMLP    <- DiffusionGemmaText4MLP
  * TTNNDiffusionGemmaTextRouter <- DiffusionGemmaTextRouter
  * TTNNDiffusionGemma{Encoder,Decoder}TextAttention (sliding + full layer types)
  * TTNNDiffusionGemmaTextExperts <- DiffusionGemmaTextExperts (dense-equivalent MoE)

Device: P150x4 mesh via the ttnn-plugin ``mesh_device`` fixture.
"""

import json
from pathlib import Path

import pytest
import torch
import ttnn

from tests.shared.pcc_utils import assert_pcc
from tt_symbiote.models.diffusion_gemma.modeling_diffusion_gemma import (
    TTNNDiffusionGemmaDecoderTextAttention,
    TTNNDiffusionGemmaEncoderTextAttention,
    TTNNDiffusionGemmaRMSNorm,
    TTNNDiffusionGemmaTextExperts,
    TTNNDiffusionGemmaTextMLP,
    TTNNDiffusionGemmaTextRouter,
)
from tt_symbiote.utils.device_management import set_device

_SHAPES = json.loads((Path(__file__).parent.parent / "shapes.json").read_text())
_CFG = _SHAPES["model_config"]
_HF_ID = "google/diffusiongemma-26B-A4B-it"
_DEVICE_PARAMS = [{"mesh_shape": (1, 4)}]


@pytest.fixture(scope="module")
def text_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(_HF_ID).text_config


def _from_torch(t, mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    if t.dtype == torch.float32 and dtype == ttnn.bfloat16:
        t = t.to(torch.bfloat16)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_torch(t, mesh_device, ref_shape):
    """Read a module/op output back to torch, shaped like the ref.

    Handles TorchTTNNTensor (module __call__ wrapper), plain torch.Tensor, and
    raw ttnn.Tensor. Inputs are replicated across the mesh, so a concat readback
    yields N identical copies -> take the first ref_shape-volume elements.
    """
    from tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(t, TorchTTNNTensor):
        t.elem = None
        got = t.to_torch
    elif isinstance(t, torch.Tensor):
        got = t
    else:
        got = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    got = got.float()
    n = 1
    for d in ref_shape:
        n *= d
    return got.flatten()[:n].reshape(ref_shape)


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("with_scale", [True, False])
def test_rmsnorm_module(mesh_device, pcc_threshold, with_scale):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaRMSNorm,
    )

    torch.manual_seed(0)
    dim = _CFG["hidden_size"]
    ref_mod = DiffusionGemmaRMSNorm(dim, eps=_CFG["rms_norm_eps"], with_scale=with_scale).eval()
    if with_scale:
        with torch.no_grad():
            ref_mod.weight.copy_(torch.randn(dim))

    x = torch.randn(1, 128, dim)
    ref = ref_mod(x)

    tt = TTNNDiffusionGemmaRMSNorm.from_torch(ref_mod)
    set_device(tt, mesh_device)
    got = _to_torch(tt.forward(_from_torch(x, mesh_device)), mesh_device, ref.shape)

    assert_pcc(got, ref, threshold=pcc_threshold, msg=f"RMSNorm(with_scale={with_scale})")


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_text_mlp_module(mesh_device, pcc_threshold, text_config):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaText4MLP,
    )

    torch.manual_seed(1)
    ref_mod = DiffusionGemmaText4MLP(text_config, layer_idx=0).eval()
    x = torch.randn(1, 128, _CFG["hidden_size"])
    ref = ref_mod(x)

    tt = TTNNDiffusionGemmaTextMLP.from_torch(ref_mod)
    set_device(tt, mesh_device)
    got = _to_torch(tt.forward(_from_torch(x, mesh_device)), mesh_device, ref.shape)

    assert_pcc(got, ref, threshold=pcc_threshold, msg="TextMLP")


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_router_module(mesh_device, pcc_threshold, text_config):
    """Router: validate softmax probabilities + top-k weight compute via PCC.

    Validated on hardware (P150x4): probs pcc 0.9992, weights pcc 0.9983.
    Two precision effects of bf16 topk over 128 experts are handled explicitly:
      * the expert *set* picked on-device agrees with fp32 for ~84% of tokens
        (the rest are near-tied logits -- expected bf16 behavior, not a bug), so
        weights are compared only on matched tokens; and
      * ttnn.topk and torch.topk order the k experts differently within a token,
        so the matched-token weights are compared as a SORTED multiset.
    """
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaTextRouter,
    )

    torch.manual_seed(2)
    ref_mod = DiffusionGemmaTextRouter(text_config).eval()
    with torch.no_grad():
        # Larger proj variance -> peaked softmax over 128 experts, so PCC is
        # meaningful (a near-uniform distribution makes both the probs PCC and
        # the selected experts degenerate/unstable).
        ref_mod.proj.weight.copy_(torch.randn_like(ref_mod.proj.weight))
        ref_mod.scale.copy_(torch.randn_like(ref_mod.scale))
        ref_mod.per_expert_scale.copy_(torch.rand_like(ref_mod.per_expert_scale) + 0.5)

    x = torch.randn(1, 128, _CFG["hidden_size"])
    ref_probs, ref_weights, ref_index = ref_mod(x)

    tt = TTNNDiffusionGemmaTextRouter.from_torch(ref_mod)
    set_device(tt, mesh_device)
    probs, weights, index = tt.forward(_from_torch(x, mesh_device))

    got_probs = _to_torch(probs, mesh_device, ref_probs.shape)
    assert_pcc(got_probs, ref_probs, threshold=pcc_threshold, msg="Router probabilities")

    got_idx = _to_torch(index, mesh_device, ref_weights.shape).round().long()
    got_w = _to_torch(weights, mesh_device, ref_weights.shape)
    agree = (got_idx.sort(-1).values == ref_index.sort(-1).values).all(-1)
    assert agree.float().mean().item() > 0.5, "router expert-set agreement unexpectedly low"
    m = agree.unsqueeze(-1).expand_as(got_w)
    assert_pcc(
        got_w.sort(-1).values[m],
        ref_weights.sort(-1).values[m],
        threshold=pcc_threshold,
        msg="Router top-k weights (matched tokens, sorted)",
    )


# ---------------------------------------------------------------------------
# Attention (encoder + decoder; sliding & full layer types) and the dense MoE.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("cls", [TTNNDiffusionGemmaEncoderTextAttention, TTNNDiffusionGemmaDecoderTextAttention])
@pytest.mark.parametrize("layer_idx", [0, 5])  # 0=sliding (hd256,kv8,v_proj); 5=full (hd512,kv2,v=k)
def test_attention_module(mesh_device, pcc_threshold, text_config, cls, layer_idx):
    """Per-head-normed + RoPE'd GQA attention. Encoder is_causal=True (the full
    model always applies a causal/sliding mask), so the HF ref gets a causal
    additive mask and TTNN SDPA runs is_causal=True (sliding_window > S here)."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaEncoderTextAttention,
        DiffusionGemmaTextRotaryEmbedding,
    )

    torch.manual_seed(10 + layer_idx)
    ref_mod = DiffusionGemmaEncoderTextAttention(text_config, layer_idx=layer_idx).eval()
    with torch.no_grad():  # exercise non-trivial per-head q/k norms
        ref_mod.q_norm.weight.copy_(torch.randn_like(ref_mod.q_norm.weight) * 0.1 + 1)
        ref_mod.k_norm.weight.copy_(torch.randn_like(ref_mod.k_norm.weight) * 0.1 + 1)

    S = 128
    hidden = torch.randn(1, S, _CFG["hidden_size"])
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = DiffusionGemmaTextRotaryEmbedding(text_config)(
        hidden, pos, layer_type=text_config.layer_types[layer_idx]
    )
    # Causal additive mask (matches SDPA is_causal=True); HF eager adds it.
    causal = torch.triu(torch.full((1, 1, S, S), float("-inf")), diagonal=1)
    ref, _ = ref_mod(hidden, position_embeddings=(cos, sin), attention_mask=causal)

    tt = cls.from_torch(ref_mod)
    set_device(tt, mesh_device)
    out = tt.forward(
        _from_torch(hidden, mesh_device),
        (_from_torch(cos, mesh_device), _from_torch(sin, mesh_device)),
    )
    got = _to_torch(out, mesh_device, ref.shape)
    assert_pcc(got, ref, threshold=pcc_threshold, msg=f"{cls.__name__} L{layer_idx}")


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_experts_module(mesh_device, pcc_threshold, text_config):
    """128-expert SwiGLU MoE via ttnn.sparse_matmul (gemma4 prefill pattern) with
    expert-chunking to fit the Blackhole core grid. Identical router (index,
    weights) are fed to both torch and TTNN to isolate the expert compute.
    Validated reduced-scale pcc 0.9949; full-128-expert scale uses gcd-based
    expert chunks (G=2 -> 64 chunks), slower but functionally equivalent."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaTextExperts,
        DiffusionGemmaTextRouter,
    )

    torch.manual_seed(7)
    ref_mod = DiffusionGemmaTextExperts(text_config).eval()
    with torch.no_grad():  # HF __init__ uses torch.empty -> must initialize
        ref_mod.gate_up_proj.normal_(0, 0.02)
        ref_mod.down_proj.normal_(0, 0.02)

    router = DiffusionGemmaTextRouter(text_config).eval()
    with torch.no_grad():
        router.proj.weight.copy_(torch.randn_like(router.proj.weight))
    hidden = torch.randn(128, _CFG["hidden_size"])
    _, top_w, top_i = router(hidden)

    ref = ref_mod(hidden, top_i, top_w)

    tt = TTNNDiffusionGemmaTextExperts.from_torch(ref_mod)
    set_device(tt, mesh_device)
    out = tt.forward(
        _from_torch(hidden, mesh_device),
        _from_torch(top_i.to(torch.int32), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32),
        _from_torch(top_w, mesh_device),
    )
    got = _to_torch(out, mesh_device, ref.shape)
    assert_pcc(got, ref, threshold=pcc_threshold, msg="MoE experts (dense)")
