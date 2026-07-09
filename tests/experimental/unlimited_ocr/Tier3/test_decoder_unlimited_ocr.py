# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier3 decoder-layer PCC test for baidu/Unlimited-OCR on Blackhole (P150).

The dense (layer-0) DeepSeek-V2 decoder layer is:
  input_layernorm(RMSNorm) -> self_attn(Llama MHA) -> post_attention_layernorm
  -> mlp(dense SwiGLU). Pre-norm residual.

BLOCKED this iteration: the layer depends on TTNNUnlimitedOcrLlamaMHA (RoPE+SDPA),
which is not yet greened, so the layer forward raises NotImplementedError. Marked
xfail; the RMSNorm and dense-MLP sub-composites it composes are already GREEN in
Tier1/Tier2.
"""

import json
import math
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import ttnn

from tt_symbiote.utils.device_management import set_device
from tests.shared.pcc_utils import assert_pcc

SHAPES = json.loads((Path(__file__).parent.parent / "shapes.json").read_text())
T3 = SHAPES["tier_shapes"]["tier3"]
LM = SHAPES["config"]["language_model"]
PCC = 0.99


def _run_no_fallback(tt, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tt(*args, **kwargs)
        fell_back = any("fallback" in str(w.message).lower() for w in caught)
    assert not fell_back, "TTNN forward fell back to torch (PCC would be a false 1.0)"
    return out


def _rope_cos_sin(seq, head_dim, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


def _rotate_half(t, d):
    return torch.cat([-t[..., d // 2:], t[..., :d // 2]], dim=-1)


# Reuse the shared `device` fixture from tests/experimental/unlimited_ocr/conftest.py
# (function-scoped, honors device_params; avoids the manual open/close hang).
pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        v = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return self.weight * (x.to(torch.float32) * torch.rsqrt(v + self.variance_epsilon)).to(x.dtype)


class _MLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    def __init__(self, hidden, heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.layer_idx = 0

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)
        q = q * cos + _rotate_half(q, D) * sin
        k = k * cos + _rotate_half(k, D) * sin
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
        attn = torch.softmax(attn + mask, dim=-1) @ v
        return self.o_proj(attn.transpose(1, 2).reshape(B, S, -1))


class _DecoderLayer(nn.Module):
    """Faithful dense (layer-0) DeepSeek-V2 decoder layer (pre-norm residual)."""

    def __init__(self, h, heads, head_dim, i):
        super().__init__()
        self.input_layernorm = _RMSNorm(h)
        self.self_attn = _Attn(h, heads, head_dim)
        self.post_attention_layernorm = _RMSNorm(h)
        self.mlp = _MLP(h, i)
        self.layer_idx = 0

    def forward(self, x, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))


def test_decoder_layer_dense(device):
    """Dense (layer-0) decoder layer: RMSNorm + Llama MHA (prefill) + SwiGLU MLP."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrDecoderLayer,
    )

    cfg = T3["decoder_layer_dense"]
    hidden = cfg["hidden"]
    heads, head_dim, inter = LM["num_attention_heads"], LM["head_dim"], LM["intermediate_size"]
    seq = cfg["input"][1]
    layer = _DecoderLayer(hidden, heads, head_dim, inter).eval()
    for p in layer.parameters():
        if p.dim() > 1:
            p.data *= 0.1
    x = torch.randn(1, seq, hidden) * 0.1
    cos, sin = _rope_cos_sin(seq, head_dim)
    ref = layer(x, cos, sin)

    tt = TTNNUnlimitedOcrDecoderLayer.from_torch(layer, layer_idx=0)
    set_device(tt, device)
    out = _run_no_fallback(tt, x, position_embeddings=(cos, sin))
    assert_pcc(out, ref, threshold=PCC, msg="TTNNUnlimitedOcrDecoderLayer (dense L0)")


class _MoEGate(nn.Module):
    """Faithful MoEGate stand-in (softmax scoring, greedy top-k)."""

    def __init__(self, n_experts, hidden, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))
        self.top_k = top_k
        self.n_routed_experts = n_experts
        self.norm_topk_prob = False
        self.routed_scaling_factor = 1.0

    def forward(self, hidden_states):
        hs = hidden_states.view(-1, hidden_states.shape[-1])
        scores = torch.nn.functional.linear(hs.float(), self.weight.float()).softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        return topk_idx, topk_weight * self.routed_scaling_factor


class _DeepseekV2MoE(nn.Module):
    """Faithful DeepseekV2MoE stand-in (softmax gate + greedy top-k + shared experts)."""

    def __init__(self, hidden, moe_inter, n_experts, top_k, n_shared):
        super().__init__()
        self.num_experts_per_tok = top_k
        self.gate = _MoEGate(n_experts, hidden, top_k)
        self.experts = nn.ModuleList([_MLP(hidden, moe_inter) for _ in range(n_experts)])
        self.shared_experts = _MLP(hidden, moe_inter * n_shared)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        x = hidden_states.view(-1, hidden_states.shape[-1])
        y = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(topk_idx.shape[1]):
                y[t] += topk_weight[t, j] * self.experts[topk_idx[t, j].item()](x[t])
        return y.view(*orig_shape) + self.shared_experts(identity)


class _MoEDecoderLayer(nn.Module):
    """DeepSeek-V2 decoder layer with a MoE mlp (layers 1..11)."""

    def __init__(self, h, heads, head_dim):
        super().__init__()
        self.input_layernorm = _RMSNorm(h)
        self.self_attn = _Attn(h, heads, head_dim)
        self.self_attn.layer_idx = 1
        self.post_attention_layernorm = _RMSNorm(h)
        self.mlp = _DeepseekV2MoE(h, LM["moe_intermediate_size"], LM["n_routed_experts"],
                                  LM["num_experts_per_tok"], LM["n_shared_experts"])
        self.layer_idx = 1

    def forward(self, x, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))


def test_decoder_layer_moe(device):
    """MoE (layers 1..11) decoder layer: RMSNorm + Llama MHA (prefill) + DeepSeek MoE."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrDecoderLayer,
    )

    cfg = T3["decoder_layer_dense"]
    hidden = cfg["hidden"]
    heads, head_dim = LM["num_attention_heads"], LM["head_dim"]
    seq = cfg["input"][1]
    layer = _MoEDecoderLayer(hidden, heads, head_dim).eval()
    for p in layer.parameters():
        if p.dim() > 1:
            p.data *= 0.1
    x = torch.randn(1, seq, hidden) * 0.1
    cos, sin = _rope_cos_sin(seq, head_dim)
    ref = layer(x, cos, sin)

    tt = TTNNUnlimitedOcrDecoderLayer.from_torch(layer, layer_idx=1)
    set_device(tt, device)
    out = _run_no_fallback(tt, x, position_embeddings=(cos, sin))
    assert_pcc(out, ref, threshold=PCC, msg="TTNNUnlimitedOcrDecoderLayer (MoE L1)")


# =============================================================================
# Vision block layers (SAM + CLIP) -- GREEN on P150. Faithful deepencoder.py stand-ins.
# =============================================================================
import torch.nn.functional as F  # noqa: E402


def _quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class _RefClipAttn(nn.Module):
    def __init__(self, hidden=1024, heads=16):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x):
        b, n, _ = x.shape
        xqkv = self.qkv_proj(x).view(b, n, 3, self.num_heads, self.head_dim)
        xq, xk, xv = torch.split(xqkv, 1, dim=2)
        xq = xq.squeeze(2).permute(0, 2, 1, 3)
        xk = xk.squeeze(2).permute(0, 2, 1, 3)
        xv = xv.squeeze(2).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
        return self.out_proj(o.permute(0, 2, 1, 3).reshape(b, n, -1))


class _RefClipFFN(nn.Module):
    def __init__(self, dim=1024, hid=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(_quick_gelu(self.fc1(x)))


class _RefClipBlock(nn.Module):
    def __init__(self, dim=1024, heads=16, hid=4096, eps=1e-5):
        super().__init__()
        self.self_attn = _RefClipAttn(dim, heads)
        self.mlp = _RefClipFFN(dim, hid)
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


def _get_rel_pos(q_size, k_size, rel_pos):
    q = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    rc = (q - k) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[rc.long()]


def _add_decomposed_rel_pos(q, rph, rpw, qs, ks):
    qh, qw = qs
    kh, kw = ks
    Rh = _get_rel_pos(qh, kh, rph)
    Rw = _get_rel_pos(qw, kw, rpw)
    B, _, dim = q.shape
    rq = q.reshape(B, qh, qw, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", rq, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", rq, Rw)
    return (rel_h.unsqueeze(-1).reshape(B, qh * qw, kh, 1),
            rel_w.unsqueeze(-2).reshape(B, qh * qw, 1, kw))


class _RefSamAttn(nn.Module):
    def __init__(self, dim=768, heads=12, input_size=(14, 14)):
        super().__init__()
        self.num_heads = heads
        hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = True
        self.rel_pos_h = nn.Parameter(torch.randn(2 * input_size[0] - 1, hd) * 0.1)
        self.rel_pos_w = nn.Parameter(torch.randn(2 * input_size[1] - 1, hd) * 0.1)

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        rel_h, rel_w = _add_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)
        rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
        rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
        ab = (rel_h + rel_w).view(B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ab)
        return self.proj(x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1))


class _RefMLPBlock(nn.Module):
    def __init__(self, dim=768, mlp=3072):
        super().__init__()
        self.lin1 = nn.Linear(dim, mlp)
        self.lin2 = nn.Linear(mlp, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


def _win_part(x, ws):
    B, H, W, C = x.shape
    ph = (ws - H % ws) % ws
    pw = (ws - W % ws) % ws
    if ph > 0 or pw > 0:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C), (Hp, Wp)


def _win_unpart(w, ws, pad, hw):
    Hp, Wp = pad
    H, W = hw
    B = w.shape[0] // (Hp * Wp // ws // ws)
    x = w.view(B, Hp // ws, Wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :] if (Hp > H or Wp > W) else x


class _RefSamBlock(nn.Module):
    def __init__(self, dim=768, heads=12, ws=14, input_size=(64, 64), eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.attn = _RefSamAttn(dim, heads, input_size if ws == 0 else (ws, ws))
        self.mlp = _RefMLPBlock(dim, dim * 4)
        self.window_size = ws

    def forward(self, x):
        sc = x
        x = self.norm1(x)
        H, W = x.shape[1], x.shape[2]
        if self.window_size > 0:
            x, pad = _win_part(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = _win_unpart(x, self.window_size, pad, (H, W))
        x = sc + x
        return x + self.mlp(self.norm2(x))


def test_clip_block(device):
    """CLIP NoTPTransformerBlock: LN->SDPA attn->+res->LN->quick_gelu FFN->+res."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrClipBlock

    blk = _RefClipBlock().eval()
    for p in blk.parameters():
        p.data *= 0.3
    x = torch.randn(1, 257, 1024) * 0.3
    tt = TTNNUnlimitedOcrClipBlock.from_torch(blk)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, blk(x), threshold=PCC, msg="TTNNUnlimitedOcrClipBlock")


def test_sam_block_window(device):
    """SAM windowed Block: window_partition -> rel-pos attn -> unpartition + GELU MLP."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrSamBlock

    blk = _RefSamBlock(ws=14).eval()
    for p in blk.parameters():
        p.data *= 0.3
    x = torch.randn(1, 64, 64, 768) * 0.3
    tt = TTNNUnlimitedOcrSamBlock.from_torch(blk)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, blk(x), threshold=PCC, msg="TTNNUnlimitedOcrSamBlock (window)")


def test_sam_block_global(device):
    """SAM global Block (window_size=0): full 64x64 rel-pos attn + GELU MLP."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrSamBlock

    blk = _RefSamBlock(ws=0).eval()
    for p in blk.parameters():
        p.data *= 0.3
    x = torch.randn(1, 64, 64, 768) * 0.3
    tt = TTNNUnlimitedOcrSamBlock.from_torch(blk)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, blk(x), threshold=PCC, msg="TTNNUnlimitedOcrSamBlock (global)")
