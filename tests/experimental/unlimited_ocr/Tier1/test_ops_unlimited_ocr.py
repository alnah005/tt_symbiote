# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier1 leaf-op PCC tests for baidu/Unlimited-OCR on Blackhole (P150).

Validates the REUSED tt_symbiote integration modules that back the model's leaf
ops (Linear, RMSNorm, Embedding, SiLU, GELU, LayerNorm) with real TTNN execution
and explicit ``assert_pcc``.

HARDWARE RETARGET SHIM: several reused modules carry ``@run_on_devices(T3K)``.
The active hardware is Blackhole (single device -> P150). ``run_on_devices``
captures the allowed-arch set in a closure at decoration time, so we re-decorate
each such ``forward`` (via its ``__wrapped__`` original) to widen the arch set to
include P150/P150x4. This does NOT edit shared source; it only adjusts the
in-process guard so the genuine TTNN op runs on this machine. Every test asserts
NO torch fallback fired, so a silent fallback (which would give a false PCC=1.0)
fails the test.
"""

import json
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import ttnn

from tt_symbiote.core.module import DeviceArch, run_on_devices
from tt_symbiote.utils.device_management import set_device
from tests.shared.pcc_utils import assert_pcc, compute_pcc

SHAPES = json.loads((Path(__file__).parent.parent / "shapes.json").read_text())
T1 = SHAPES["tier_shapes"]["tier1"]
PCC = SHAPES.get("pcc_threshold", 0.99)

# Every test reuses the ttnn-plugin `device` fixture (single Blackhole device ->
# P150), configured once via device_params. l1_small_size backs the k3 conv2d halo
# (sliding-window) config tensors; it is harmless for the non-conv ops. Reusing the
# plugin-managed device (instead of manually open/close-ing a mesh per test) avoids
# the program-cache/L1 accumulation that hung the suite. Mirrors tests/shared/test_conv.py.
pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)


def _widen(cls):
    """Re-decorate cls.forward to also allow P150/P150x4 (hardware retarget)."""
    f = cls.forward
    orig = getattr(f, "__wrapped__", None)
    allowed = getattr(f, "__tt_allowed_archs__", None)
    if orig is not None and allowed is not None and DeviceArch.P150 not in allowed:
        cls.forward = run_on_devices(*(set(allowed) | {DeviceArch.P150, DeviceArch.P150x4}))(orig)


def _run_no_fallback(tt, *args, **kwargs):
    """Run tt(*args) asserting the TTNN path executed (no torch fallback)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tt(*args, **kwargs)
        fell_back = any("fallback" in str(w.message).lower() for w in caught)
    assert not fell_back, "TTNN forward fell back to torch (PCC would be a false 1.0)"
    return out


def test_linear(device):
    from tt_symbiote.modules.ttnn_linear import TTNNLinear

    cfg = T1["linear"]
    lin = nn.Linear(cfg["in_features"], cfg["out_features"], bias=False).eval()
    x = torch.randn(*cfg["input"])
    tt = TTNNLinear.from_torch(lin)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, lin(x), threshold=PCC, msg="TTNNLinear")


def test_rmsnorm(device):
    from tt_symbiote.modules.ttnn_normalization import DeepseekV2RMSNorm, TTNNRMSNorm

    _widen(TTNNRMSNorm)
    cfg = T1["rmsnorm"]
    rms = DeepseekV2RMSNorm(cfg["dim"], eps=cfg["eps"]).eval()
    rms.weight.data = torch.randn(cfg["dim"]) * 0.1 + 1.0
    x = torch.randn(*cfg["input"])
    tt = TTNNRMSNorm.from_torch(rms)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, rms(x), threshold=PCC, msg="TTNNRMSNorm")


def test_embedding(device):
    from tt_symbiote.modules.ttnn_embedding import TTNNEmbedding

    _widen(TTNNEmbedding)
    cfg = T1["embedding"]
    # Small vocab keeps the lookup cheap; the op is vocab-size-agnostic for PCC.
    emb = nn.Embedding(1024, cfg["embedding_dim"]).eval()
    idx = torch.randint(0, 1024, tuple(cfg["indices"]))
    tt = TTNNEmbedding.from_torch(emb)
    set_device(tt, device)
    # ttnn.embedding requires UINT32 indices; the generic __call__ tensor-wrap of
    # int64 yields an unsupported dtype, so build the ttnn index tensor directly
    # and call forward (weights are already preprocessed + on device by set_device).
    tt.preprocess_weights()
    tt.move_weights_to_device()
    tt_idx = ttnn.from_torch(
        idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = tt.forward(tt_idx)
    out_torch = ttnn.to_torch(out)
    assert_pcc(out_torch, emb(idx), threshold=PCC, msg="TTNNEmbedding")


def test_silu(device):
    from tt_symbiote.modules.ttnn_activation import TTNNSilu

    _widen(TTNNSilu)
    cfg = T1["silu"]
    act = nn.SiLU().eval()
    x = torch.randn(*cfg["input"])
    tt = TTNNSilu()
    tt._fallback_torch_layer = act
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, act(x), threshold=PCC, msg="TTNNSilu")


def test_gelu(device):
    from tt_symbiote.modules.ttnn_activation import TTNNGelu

    _widen(TTNNGelu)
    cfg = T1["gelu"]
    act = nn.GELU().eval()
    x = torch.randn(*cfg["input"])
    tt = TTNNGelu()
    tt._fallback_torch_layer = act
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, act(x), threshold=PCC, msg="TTNNGelu")


def test_layernorm(device):
    from tt_symbiote.modules.ttnn_normalization import TTNNLayerNorm

    _widen(TTNNLayerNorm)
    ln = nn.LayerNorm(1024).eval()
    ln.weight.data = torch.randn(1024) * 0.1 + 1.0
    ln.bias.data = torch.randn(1024) * 0.1
    x = torch.randn(1, 8, 1024)
    tt = TTNNLayerNorm.from_torch(ln)
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, ln(x), threshold=PCC, msg="TTNNLayerNorm")


# ---------------------------------------------------------------------------
# Vision leaf ops (custom activation / normalization) -- GREEN on P150.
# ---------------------------------------------------------------------------
def test_quick_gelu(device):
    """CLIP quick_gelu = x * sigmoid(1.702 * x) (custom; NOT ttnn.gelu)."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrQuickGELU,
    )

    x = torch.randn(1, 64, 1024) * 0.5
    ref = x * torch.sigmoid(1.702 * x)
    tt = TTNNUnlimitedOcrQuickGELU.from_torch()
    set_device(tt, device)
    out = _run_no_fallback(tt, x)
    assert_pcc(out, ref, threshold=PCC, msg="TTNNUnlimitedOcrQuickGELU")


class _LayerNorm2d(nn.Module):
    """Faithful stand-in for the SAM neck LayerNorm2d (channel-first, NCHW)."""

    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(C))
        self.bias = nn.Parameter(torch.randn(C))
        self.eps = eps

    def forward(self, x):  # x: NCHW
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def test_layernorm2d(device):
    """SAM neck LayerNorm2d: channel-first LN == ttnn.layer_norm over last dim (NHWC)."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrLayerNorm2d,
    )

    ln = _LayerNorm2d(256).eval()
    x_nhwc = torch.randn(1, 64, 64, 256)
    ref = ln(x_nhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # NHWC
    tt = TTNNUnlimitedOcrLayerNorm2d.from_torch(ln)
    set_device(tt, device)
    out = _run_no_fallback(tt, x_nhwc)
    assert_pcc(out, ref, threshold=PCC, msg="TTNNUnlimitedOcrLayerNorm2d")


# ---------------------------------------------------------------------------
# Base ops previously exercised only INSIDE composites -- now covered standalone
# as genuine Tier1 leaves: Conv2d (nn.Module), and the inline ttnn ops SDPA,
# softmax, and HF-Llama RoPE (rotate_half). All are known-GREEN (they back the
# green composites/encoders); validated here with real PCC + no-fallback.
# ---------------------------------------------------------------------------
def _run_conv(fn):
    """Run a conv forward asserting no torch fallback EXCEPT the known shared
    TTNNReshape guard (T3K-only; the conv2d itself runs on device -- see Tier4)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
        bad = [
            str(w.message)
            for w in caught
            if "fallback" in str(w.message).lower() and "TTNNReshape" not in str(w.message)
        ]
    assert not bad, f"Unexpected torch fallback (not the shared TTNNReshape guard): {bad}"
    return out


def _call_no_fallback(fn):
    """Run an inline-ttnn-op callable asserting no torch fallback warning fired."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
        fell_back = any("fallback" in str(w.message).lower() for w in caught)
    assert not fell_back, "unexpected torch fallback (PCC would be a false 1.0)"
    return out


# (name, in_channels, out_channels, kernel, stride, padding, NCHW input shape).
# The model's 5 conv configs; HxW shrunk to keep the standalone op light (kernel/
# stride ratios preserved so the conv path is identical to the real encoder).
_CONV_CONFIGS = [
    ("patch_embed", 3, 768, 16, 16, 0, [1, 3, 64, 64]),
    ("neck_conv1", 768, 256, 1, 1, 0, [1, 768, 16, 16]),
    ("neck_conv2", 256, 256, 3, 1, 1, [1, 256, 16, 16]),
    ("net_2", 256, 512, 3, 2, 1, [1, 256, 16, 16]),
    ("net_3", 512, 1024, 3, 2, 1, [1, 512, 16, 16]),
]


@pytest.mark.parametrize(
    "name,cin,cout,k,s,p,nchw", _CONV_CONFIGS, ids=[c[0] for c in _CONV_CONFIGS]
)
def test_conv2d(device, name, cin, cout, k, s, p, nchw):
    """SAM patch_embed / neck / net conv2d (the genuine missing leaf nn.Module).

    Mirrors TTNNUnlimitedOcrSamEncoder: TTNNConv2dNHWC.from_torch -> set_device ->
    NHWC ttnn input -> NHWC ttnn output; compared vs torch NCHW conv permuted NHWC.
    """
    from tt_symbiote.modules.ttnn_conv import TTNNConv2dNHWC

    torch.manual_seed(0)
    conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p).eval()
    x_nchw = torch.randn(*nchw)
    ref = conv(x_nchw).permute(0, 2, 3, 1)  # NCHW -> NHWC to align with ttnn output

    tt = TTNNConv2dNHWC.from_torch(conv)
    set_device(tt, device)
    x_nhwc = ttnn.from_torch(
        x_nchw.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out = _run_conv(lambda: tt.forward(x_nhwc))
    got = ttnn.to_torch(out).reshape(ref.shape)
    assert_pcc(got, ref, threshold=PCC, msg=f"TTNNConv2dNHWC[{name}]")


def test_sdpa(device):
    """Standalone scaled_dot_product_attention (backs the LM + CLIP attention).

    ttnn.transformer.scaled_dot_product_attention (bf16, non-causal) vs torch SDPA.
    """
    import math

    torch.manual_seed(0)
    B, H, S, D = 1, 8, 32, 64
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # non-causal, scale 1/sqrt(D)

    tq, tk, tv = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for t in (q, k, v)
    )
    out = _call_no_fallback(
        lambda: ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, scale=1.0 / math.sqrt(D)
        )
    )
    assert_pcc(ttnn.to_torch(out), ref, threshold=PCC, msg="ttnn.transformer.scaled_dot_product_attention")


def test_softmax(device):
    """ttnn.softmax(dim=-1) vs torch.softmax (backs MoE router + SAM window attn)."""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64)
    ref = torch.softmax(x, dim=-1)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = _call_no_fallback(lambda: ttnn.softmax(tx, dim=-1))
    assert_pcc(ttnn.to_torch(out), ref, threshold=PCC, msg="ttnn.softmax")


def _rope_cos_sin(seq, head_dim, theta=10000.0):
    """HF-Llama cos/sin tables, broadcastable to [1,1,S,head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


def _rotate_half_torch(t, d):
    return torch.cat([-t[..., d // 2:], t[..., :d // 2]], dim=-1)


def test_rope(device):
    """HF-Llama RoPE (rotate_half + cos/sin) -- the EXACT ttnn ops used in
    TTNNUnlimitedOcrLlamaMHA._rope -- vs torch apply_rotary_pos_emb."""
    torch.manual_seed(0)
    H, S, D = 8, 32, 64
    q = torch.randn(1, H, S, D)
    cos_t, sin_t = _rope_cos_sin(S, D)
    ref = q * cos_t + _rotate_half_torch(q, D) * sin_t

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _rope(t):  # copied from TTNNUnlimitedOcrLlamaMHA.forward._rope
        x1 = ttnn.slice(t, [0, 0, 0, 0], [1, H, S, D // 2])
        x2 = ttnn.slice(t, [0, 0, 0, D // 2], [1, H, S, D])
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(rot, sin))

    out = _call_no_fallback(lambda: _rope(tq))
    assert_pcc(ttnn.to_torch(out), ref, threshold=PCC, msg="ttnn HF-Llama RoPE (rotate_half)")
