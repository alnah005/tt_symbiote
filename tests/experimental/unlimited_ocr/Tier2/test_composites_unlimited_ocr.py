# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier2 composite PCC tests for baidu/Unlimited-OCR on Blackhole (P150).

GREEN (real TTNN, no fallback):
  * TTNNUnlimitedOcrDeepseekMLP  -- dense SwiGLU MLP (down(silu(gate(x))*up(x)))
  * TTNNUnlimitedOcrMlpProjector -- vision->text Linear(2048->1280)

BLOCKED / deferred (xfail, documented): MoE routing, Llama MHA (RoPE+SDPA),
SAM/CLIP attention. See per-test skip/xfail reasons and op_map.json.
"""

import json
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn

from tt_symbiote.utils.device_management import set_device
from tests.shared.pcc_utils import assert_pcc

SHAPES = json.loads((Path(__file__).parent.parent / "shapes.json").read_text())
T2 = SHAPES["tier_shapes"]["tier2"]
PCC = 0.99


# Reuse the shared `device` fixture from tests/experimental/unlimited_ocr/conftest.py
# (function-scoped, honors device_params; avoids the manual open/close hang).
pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)


def _run_no_fallback(tt, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tt(*args, **kwargs)
        fell_back = any("fallback" in str(w.message).lower() for w in caught)
    assert not fell_back, "TTNN forward fell back to torch (PCC would be a false 1.0)"
    return out


class _DeepseekV2MLP(nn.Module):
    """Faithful stand-in for the remote-code DeepseekV2MLP (SwiGLU)."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_deepseek_mlp(device):
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrDeepseekMLP,
    )

    cfg = T2["deepseek_mlp"]
    mlp = _DeepseekV2MLP(cfg["hidden"], cfg["intermediate"]).eval()
    x = torch.randn(*cfg["input"]) * 0.1
    tt = TTNNUnlimitedOcrDeepseekMLP.from_torch(mlp)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, mlp(x), threshold=PCC, msg="TTNNUnlimitedOcrDeepseekMLP")


def test_mlp_projector(device):
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrMlpProjector,
    )

    cfg = T2["mlp_projector"]
    proj = nn.Linear(cfg["in"], cfg["out"]).eval()
    x = torch.randn(*cfg["input"]) * 0.1
    tt = TTNNUnlimitedOcrMlpProjector.from_torch(proj)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, proj(x), threshold=PCC, msg="TTNNUnlimitedOcrMlpProjector")


class _MoEGate(nn.Module):
    """Faithful stand-in for the remote-code MoEGate (softmax scoring, greedy top-k)."""

    def __init__(self, n_experts, hidden, top_k, norm_topk_prob=False, routed_scaling_factor=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))
        self.top_k = top_k
        self.n_routed_experts = n_experts
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = "softmax"

    def forward(self, hidden_states):
        h = hidden_states.shape[-1]
        hs = hidden_states.view(-1, h)
        logits = F.linear(hs.float(), self.weight.float(), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class _DeepseekV2MoE(nn.Module):
    """Faithful stand-in for the remote-code DeepseekV2MoE (softmax + greedy top-k)."""

    def __init__(self, cfg):
        super().__init__()
        h, mi = cfg["hidden"], cfg["moe_intermediate"]
        n, k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
        self.num_experts_per_tok = k
        self.gate = _MoEGate(n, h, k)
        self.experts = nn.ModuleList([_DeepseekV2MLP(h, mi) for _ in range(n)])
        self.shared_experts = _DeepseekV2MLP(h, mi * cfg["n_shared_experts"])

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        x = hidden_states.view(-1, hidden_states.shape[-1])
        y = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(topk_idx.shape[1]):
                e = topk_idx[t, j].item()
                y[t] += topk_weight[t, j] * self.experts[e](x[t])
        y = y.view(*orig_shape)
        return y + self.shared_experts(identity)


def test_moe(device):
    """DeepSeek-V2 MoE: fp32 softmax gate -> greedy top-6 -> routed + shared experts."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrMoE

    cfg = dict(T2["moe"], n_shared_experts=2)
    moe = _DeepseekV2MoE(cfg).eval()
    for p in moe.parameters():
        if p.dim() > 1:
            p.data *= 0.1
    x = torch.randn(*cfg["input"]) * 0.1
    ref = moe(x)

    tt = TTNNUnlimitedOcrMoE.from_torch(moe)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, ref, threshold=PCC, msg="TTNNUnlimitedOcrMoE")


def _rotate_half(t, d):
    return torch.cat([-t[..., d // 2:], t[..., :d // 2]], dim=-1)


def _llama_rope_cos_sin(seq, head_dim, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]  # [1,1,S,D]


class _LlamaMHA(nn.Module):
    """Faithful HF-Llama-style MHA (prefill / full causal attention)."""

    def __init__(self, hidden, heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x, cos, sin):
        import math

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


def test_llama_mha_prefill(device):
    """Prefill-only PCC for the plain Llama MHA (RoPE + causal SDPA + o_proj).

    Decode / ring-buffer sliding-window (win=128) KV cache is NOT covered here.
    """
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrLlamaMHA

    cfg = T2["llama_mha"]
    hidden, heads, head_dim = cfg["hidden"], cfg["heads"], cfg["head_dim"]
    seq = cfg["input"][1]
    mha = _LlamaMHA(hidden, heads, head_dim).eval()
    for p in mha.parameters():
        p.data *= 0.1
    x = torch.randn(1, seq, hidden) * 0.1
    cos, sin = _llama_rope_cos_sin(seq, head_dim)
    ref = mha(x, cos, sin)

    tt = TTNNUnlimitedOcrLlamaMHA.from_torch(mha, layer_idx=0)
    set_device(tt, device)
    out = _run_no_fallback(tt, x, position_embeddings=(cos, sin))
    assert_pcc(out, ref, threshold=PCC, msg="TTNNUnlimitedOcrLlamaMHA (prefill)")


# =============================================================================
# Vision composites (SAM + CLIP) -- GREEN on P150. Faithful stand-ins mirror the
# remote-code deepencoder.py forwards exactly.
# =============================================================================
def _quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class _RefClipAttn(nn.Module):
    """NoTPAttention: fused qkv_proj, 16 heads, plain non-causal SDPA, out_proj."""

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
    """SAM Attention with decomposed rel-pos bias -> SDPA(attn_mask=rel_h+rel_w)."""

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


def test_clip_attention(device):
    """CLIP NoTPAttention: fused qkv, 16-head non-causal SDPA, out_proj."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrClipAttention

    attn = _RefClipAttn().eval()
    for p in attn.parameters():
        p.data *= 0.1
    x = torch.randn(1, 257, 1024) * 0.5
    tt = TTNNUnlimitedOcrClipAttention.from_torch(attn)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, attn(x), threshold=PCC, msg="TTNNUnlimitedOcrClipAttention")


def test_clip_ffn(device):
    """CLIP NoTPFeedForward: fc2(quick_gelu(fc1(x)))."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrClipFFN

    ff = _RefClipFFN().eval()
    x = torch.randn(1, 257, 1024) * 0.5
    tt = TTNNUnlimitedOcrClipFFN.from_torch(ff)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, ff(x), threshold=PCC, msg="TTNNUnlimitedOcrClipFFN")


def test_sam_attention_window(device):
    """SAM windowed attention (rel-pos einsum + SDPA mask), window grid 14x14, batched windows."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrSamAttention

    attn = _RefSamAttn(input_size=(14, 14)).eval()
    for p in attn.parameters():
        p.data *= 0.5
    x = torch.randn(2, 14, 14, 768) * 0.5  # Bn=2 windows
    tt = TTNNUnlimitedOcrSamAttention.from_torch(attn)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, attn(x), threshold=PCC, msg="TTNNUnlimitedOcrSamAttention (window 14x14)")


def test_sam_attention_global(device):
    """SAM global attention (rel-pos einsum + SDPA mask), full 64x64 grid (seq 4096)."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrSamAttention

    attn = _RefSamAttn(input_size=(64, 64)).eval()
    for p in attn.parameters():
        p.data *= 0.5
    x = torch.randn(1, 64, 64, 768) * 0.5
    tt = TTNNUnlimitedOcrSamAttention.from_torch(attn)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, attn(x), threshold=PCC, msg="TTNNUnlimitedOcrSamAttention (global 64x64)")
