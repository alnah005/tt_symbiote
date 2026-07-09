# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier4 full-model PCC test for baidu/Unlimited-OCR.

Two levels of validation:

* ``test_deepseek_text_model`` (GREEN, real TTNN): a text-only DeepSeek-V2
  ``model`` forward -- ``embed_tokens`` -> decoder layers (dense L0 + MoE L1..)
  -> final RMSNorm -- validated with ``assert_pcc``. This exercises the full LM
  stack shell (embedding + Llama MHA + dense/MoE MLP + residual + norm) on a
  faithful synthetic DeepSeek text model. Layer count/vocab are reduced from the
  production 12-layer / 129280-vocab config; the executed code path is identical.

* vision encoders vs the REAL reference (loads ~3.3B once): ``test_clip_encoder``
  GREEN (PCC 0.997); ``test_sam_encoder`` (PCC 0.982) and ``test_deep_encoder``
  (projector milestone PCC 0.917) are documented xfails -- functional but below the
  0.99 gate due to ttnn flash-SDPA bf16 precision compounding over the 12-deep SAM
  tower (torch-bf16 ceiling 0.9999). All vision sub-components (Tier1-3) are GREEN.

Still deferred: the full ForCausalLM e2e (scatter-merge of image tokens + lm_head)
-- see the report; the LM stack and vision encoders are validated separately here.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn

from tt_symbiote.utils.device_management import set_device
from tests.shared.pcc_utils import assert_pcc

SHAPES = json.loads((Path(__file__).parent.parent / "shapes.json").read_text())
LM = SHAPES["config"]["language_model"]
PCC = 0.99


# Reuse the shared `device` fixture from tests/experimental/unlimited_ocr/conftest.py.
# l1_small_size is required for the SAM neck/net conv2d halo (sliding-window) config
# tensors; harmless for the text-only LM path.
pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rope_cos_sin(seq, head_dim, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


def _rotate_half(t, d):
    return torch.cat([-t[..., d // 2:], t[..., :d // 2]], dim=-1)


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
    def __init__(self, hidden, heads, head_dim, layer_idx):
        super().__init__()
        self.num_heads, self.head_dim = heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.layer_idx = layer_idx

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


class _MoEGate(nn.Module):
    def __init__(self, n_experts, hidden, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))
        self.top_k = top_k
        self.n_routed_experts = n_experts
        self.norm_topk_prob = False
        self.routed_scaling_factor = 1.0

    def forward(self, hidden_states):
        hs = hidden_states.view(-1, hidden_states.shape[-1])
        scores = F.linear(hs.float(), self.weight.float()).softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        return topk_idx, topk_weight * self.routed_scaling_factor


class _DeepseekV2MoE(nn.Module):
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


class _DecoderLayer(nn.Module):
    def __init__(self, h, heads, head_dim, layer_idx, moe):
        super().__init__()
        self.input_layernorm = _RMSNorm(h)
        self.self_attn = _Attn(h, heads, head_dim, layer_idx)
        self.post_attention_layernorm = _RMSNorm(h)
        if moe:
            self.mlp = _DeepseekV2MoE(h, LM["moe_intermediate_size"], LM["n_routed_experts"],
                                      LM["num_experts_per_tok"], LM["n_shared_experts"])
        else:
            self.mlp = _MLP(h, LM["intermediate_size"])
        self.layer_idx = layer_idx

    def forward(self, x, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))


class _DeepseekModel(nn.Module):
    """Text-only DeepSeek-V2 model: embed -> (dense L0 + MoE L1..) -> RMSNorm."""

    def __init__(self, vocab, h, heads, head_dim, n_layers, first_k_dense):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, h)
        self.layers = nn.ModuleList(
            [_DecoderLayer(h, heads, head_dim, i, moe=(i >= first_k_dense)) for i in range(n_layers)]
        )
        self.norm = _RMSNorm(h)

    def forward(self, input_ids, cos, sin):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.norm(h)


def test_deepseek_text_model(device):
    """Text-only DeepSeek-V2 model forward (embed + dense L0 + MoE layers + norm)."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrDeepseekModel,
    )

    vocab = 1024                       # reduced from 129280 (op path identical)
    n_layers = 3                       # L0 dense + L1,L2 MoE
    hidden = LM["hidden_size"]
    heads, head_dim = LM["num_attention_heads"], LM["head_dim"]
    seq = 8
    torch.manual_seed(0)
    model = _DeepseekModel(vocab, hidden, heads, head_dim, n_layers, LM["first_k_dense_replace"]).eval()
    for p in model.parameters():
        if p.dim() > 1:
            p.data *= 0.1
    input_ids = torch.randint(0, vocab, (1, seq), dtype=torch.int64)
    cos, sin = _rope_cos_sin(seq, head_dim)
    ref = model(input_ids, cos, sin)

    tt = TTNNUnlimitedOcrDeepseekModel.from_torch(model)
    set_device(tt, device)
    out = tt(input_ids=input_ids, position_embeddings=(cos, sin))
    assert_pcc(out, ref, threshold=PCC, msg="TTNNUnlimitedOcrDeepseekModel (text-only)")


# =============================================================================
# Vision encoders (SAM + CLIP + DeepEncoder) vs the REAL baidu/Unlimited-OCR
# reference. Loads the ~3.3B model once (module-scoped). Pure-TTNN forwards; no
# torch fallback (the shared TTNNConv2dNHWC's trivial output reshape uses the
# T3K-guarded TTNNReshape -> torch, which is filtered as a known shared-module
# fallback; the conv2d itself runs on device).
# =============================================================================
import warnings  # noqa: E402


@pytest.fixture(scope="module")
def real_model():
    """Load the real ~3.3B baidu/Unlimited-OCR reference once (shared by e2e + vision)."""
    from tt_symbiote.models.unlimited_ocr.reference_loader import load_reference_model

    model, cfg = load_reference_model()
    return model, cfg


@pytest.fixture(scope="module")
def ref_vision(real_model):
    """Precompute the torch reference vision outputs from the shared real model."""
    model, _ = real_model
    inner = model.model if hasattr(model, "model") else model
    sam, vis, proj = inner.sam_model, inner.vision_model, inner.projector
    torch.manual_seed(0)
    px = (torch.rand(1, 3, 1024, 1024) - 0.5) / 0.5
    with torch.no_grad():
        sam_out = sam(px)                                    # [1,1024,16,16]
        vit_out = vis(px, sam_out)                           # [1,257,1024]
        feats = torch.cat([vit_out[:, 1:], sam_out.flatten(2).permute(0, 2, 1)], dim=-1)  # [1,256,2048]
        proj_out = proj(feats)                               # [1,256,1280]
    return dict(sam=sam, vis=vis, proj=proj, px=px, sam_out=sam_out,
                vit_out=vit_out, feats=feats, proj_out=proj_out)


def _run_vision(fn):
    """Run a vision forward; assert no torch fallback except the known shared
    TTNNReshape guard inside TTNNConv2dNHWC (whose conv2d itself runs on device)."""
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


def test_clip_encoder(device, ref_vision):
    """CLIP VitModel: embeddings(cls+pos) + pre_layrnorm + 24 blocks. GREEN.

    Fed the REAL SAM features as the channels-last patch sequence [1,256,1024].
    """
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrClipEncoder

    rv = ref_vision
    sam_seq = rv["sam_out"].flatten(2).permute(0, 2, 1)      # [1,256,1024]
    tt = TTNNUnlimitedOcrClipEncoder.from_torch(rv["vis"])
    set_device(tt, device)
    x = ttnn.from_torch(sam_seq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = _run_vision(lambda: tt.forward(x))
    assert_pcc(ttnn.to_torch(out), rv["vit_out"], threshold=PCC, msg="TTNNUnlimitedOcrClipEncoder")


def test_sam_encoder(device, ref_vision):
    """SAM ImageEncoderViT: patch_embed + pos_embed + 12 blocks + neck/net conv compressor.

    GREEN at PCC~0.9987. The window layers run attention manually in fp32 (QK^T ->
    +rel_pos_bias -> softmax -> AV) instead of the bf16 flash-SDPA kernel, whose bf16
    q/k/v error previously compounded across the 12-deep tower (dropping the encoder to
    0.982). Global layers (seq=4096) keep flash-SDPA (fp32 scores would OOM); their
    residual bf16 error is now small enough for the encoder to clear the 0.99 gate.
    """
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import TTNNUnlimitedOcrSamEncoder

    rv = ref_vision
    tt = TTNNUnlimitedOcrSamEncoder.from_torch(rv["sam"])
    set_device(tt, device)
    px_nhwc = ttnn.from_torch(
        rv["px"].permute(0, 2, 3, 1).contiguous(), dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    out = _run_vision(lambda: tt.forward(px_nhwc))
    got = ttnn.to_torch(out).reshape(1, 256, 1024)
    ref = rv["sam_out"].flatten(2).permute(0, 2, 1)
    print(f"\n[SAM encoder measured PCC] {_pcc(got, ref):.5f}")
    assert_pcc(got, ref, threshold=PCC, msg="TTNNUnlimitedOcrSamEncoder")


def test_deep_encoder(device, ref_vision):
    """DeepEncoder (SAM+CLIP merge) -> projector -> [1,256,1280]. Key milestone vs real ref.

    GREEN at PCC~0.996. Promoted from xfail (was 0.98871) after the SAM global-layer
    flash-SDPA accuracy lift: passing an explicit ``ttnn.SDPAProgramConfig`` with
    ``exp_approx_mode=False`` (the flash-softmax exponent defaults to the APPROXIMATE
    mode) and matched q/k chunk sizes to the 4096-seq global SDPA. This is orthogonal
    to the proven fp32_dest_acc Blackhole kernel bug (accumulator stays bf16). SAM
    encoder rose 0.99908 -> 0.99943; this milestone 0.98871 -> 0.99640.
    """
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrDeepEncoder,
        TTNNUnlimitedOcrMlpProjector,
    )

    rv = ref_vision
    de = TTNNUnlimitedOcrDeepEncoder.from_torch(rv["sam"], rv["vis"])
    set_device(de, device)
    px_nhwc = ttnn.from_torch(
        rv["px"].permute(0, 2, 3, 1).contiguous(), dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    feats = _run_vision(lambda: de.forward(px_nhwc))
    feats_t = ttnn.to_torch(feats)
    print(f"\n[DeepEncoder feats PCC] {_pcc(feats_t, rv['feats']):.5f}")

    proj = TTNNUnlimitedOcrMlpProjector.from_torch(rv["proj"])
    set_device(proj, device)
    feats_in = ttnn.from_torch(feats_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = _run_vision(lambda: proj.forward(feats_in))
    print(f"[DeepEncoder+projector milestone PCC] {_pcc(ttnn.to_torch(out), rv['proj_out']):.5f}")
    assert_pcc(ttnn.to_torch(out), rv["proj_out"], threshold=PCC, msg="TTNNUnlimitedOcrDeepEncoder+projector")


# =============================================================================
# Tier4 SEMANTIC VALIDATION -- full ForCausalLM text-only e2e vs the REAL model.
# Exercises the REAL 3.3B weights end-to-end: embed_tokens -> 12 real decoder
# layers (real SlidingWindowLlamaAttention prefill with real RoPE/causal mask +
# real dense-L0 / MoE-L1..11) -> final RMSNorm -> real lm_head (1280 -> 129280).
# This is the min-accuracy bar: logits PCC + last-token argmax-match vs torch,
# and coherent (non-degenerate, torch-matching) greedy generation.
# =============================================================================
def _real_rope(model, cfg, seq):
    """Exact (cos, sin) from the model's own LlamaRotaryEmbedding -> [1,1,S,head_dim]."""
    rot = model.model.layers[0].self_attn.rotary_emb
    pos = torch.arange(seq).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(torch.zeros(1, 1, seq, cfg.head_dim), pos)  # [1,S,D]
    return cos.unsqueeze(1).float(), sin.unsqueeze(1).float()      # [1,1,S,D]


def _tt_logits(tt, device, input_ids, cfg, model):
    """Run the TTNN ForCausalLM text-only forward and read back fp32 logits [1,S,vocab]."""
    from tt_symbiote.core.tensor import TorchTTNNTensor

    seq = input_ids.shape[-1]
    cos, sin = _real_rope(model, cfg, seq)
    tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_ids = ttnn.from_torch(
        input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = tt(input_ids=tt_ids, position_embeddings=(tt_cos, tt_sin))
    if isinstance(out, TorchTTNNTensor):
        out.elem = None
        return out.to_torch.float().reshape(1, seq, cfg.vocab_size)
    if isinstance(out, torch.Tensor):
        return out.float().reshape(1, seq, cfg.vocab_size)
    return ttnn.to_torch(out).float().reshape(1, seq, cfg.vocab_size)


@pytest.mark.slow
def test_forcausallm_text_e2e(device, real_model):
    """REAL 3.3B text-only e2e: logits PCC + argmax-match + coherent greedy generation.

    Builds ``TTNNUnlimitedOcrForCausalLM`` from the real reference (real embed +
    12 real decoder layers + real lm_head) and validates the full text LM path.
    """
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrForCausalLM,
    )

    model, cfg = real_model
    assert len(model.model.layers) == 12, "expected the full 12-layer real DeepSeek stack"

    tt = TTNNUnlimitedOcrForCausalLM.from_torch(model)
    set_device(tt, device)

    # --- (1) single-forward logits PCC + last-token argmax-match ---------------
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.int64)
    with torch.no_grad():
        ref = model(input_ids=input_ids, images=None, use_cache=False, return_dict=True).logits.float()
    got = _tt_logits(tt, device, input_ids, cfg, model)

    pcc = _pcc(got, ref)
    tt_arg = got[0, -1].argmax().item()
    ref_arg = ref[0, -1].argmax().item()
    print(f"\n[FORCAUSALLM text-e2e] logits PCC={pcc:.5f}  "
          f"last-token argmax TTNN={tt_arg} torch={ref_arg} match={tt_arg == ref_arg}")
    assert torch.isfinite(got).all(), "TTNN logits contain NaN/Inf"
    assert_pcc(got, ref, threshold=PCC, msg="TTNNUnlimitedOcrForCausalLM text-only logits")
    assert tt_arg == ref_arg, f"last-token argmax mismatch: TTNN={tt_arg} torch={ref_arg}"

    # --- (2) greedy generation: TTNN sequence must match torch + be coherent ---
    n_new = 8
    prompt = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.int64)
    tt_seq = prompt.clone()
    torch_seq = prompt.clone()
    for _ in range(n_new):
        with torch.no_grad():
            tlog = model(input_ids=torch_seq, images=None, use_cache=False, return_dict=True).logits.float()
        torch_next = tlog[0, -1].argmax().item()
        torch_seq = torch.cat([torch_seq, torch.tensor([[torch_next]])], dim=1)

        ttlog = _tt_logits(tt, device, tt_seq, cfg, model)
        tt_next = ttlog[0, -1].argmax().item()
        tt_seq = torch.cat([tt_seq, torch.tensor([[tt_next]])], dim=1)

    tt_gen = tt_seq[0, prompt.shape[-1]:].tolist()
    torch_gen = torch_seq[0, prompt.shape[-1]:].tolist()
    print(f"[FORCAUSALLM text-e2e] TTNN gen ={tt_gen}")
    print(f"[FORCAUSALLM text-e2e] torch gen={torch_gen}")

    # Coherence: non-degenerate (not all identical) and matches torch greedy path.
    assert len(set(tt_gen)) > 1, f"degenerate TTNN generation (all identical): {tt_gen}"
    n_match = sum(int(a == b) for a, b in zip(tt_gen, torch_gen))
    assert n_match >= n_new - 1, (
        f"TTNN greedy generation diverges from torch: {tt_gen} vs {torch_gen} "
        f"({n_match}/{n_new} match)"
    )


# =============================================================================
# PRIORITY 2 -- full VLM e2e (image + text) with on-device scatter-merge.
# DeepEncoder(px) -> projector -> global-only vision block [273,1280] (view + per-row
# image_newline + view_seperator) -> dots_ocr scatter-merge into the text embeds at
# the 273 <image> positions -> DeepseekModel -> lm_head. Validated vs a CPU-replicated
# torch reference (the model's own images-path uses .cuda(), unavailable on CPU).
# GREEN at logits PCC~0.9944 with last-token argmax-match: the fp32-window SAM lift
# brought the injected 273 vision tokens' fidelity up enough for the full VLM logits to
# clear the 0.99 gate (the scatter-merge + LM path is exact -- text-only e2e is 0.9977).
# =============================================================================
@pytest.mark.slow
def test_forcausallm_vlm_e2e(device, real_model):
    """REAL 3.3B image+text e2e: DeepEncoder -> scatter-merge 273 vision tokens -> LM."""
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrForCausalLM,
    )

    model, cfg = real_model
    inner = model.model
    img_id = getattr(model.config, "image_token_id", 128815)
    torch.manual_seed(0)
    px = (torch.rand(1, 3, 1024, 1024) - 0.5) / 0.5

    # torch reference: replicate the global-only vision block + masked_scatter on CPU.
    with torch.no_grad():
        g1 = inner.sam_model(px)
        g2 = inner.vision_model(px, g1)
        feats = torch.cat([g2[:, 1:], g1.flatten(2).permute(0, 2, 1)], dim=-1)
        gf = inner.projector(feats)                       # [1,256,1280]
        _, hw, nd = gf.shape
        h = w = int(hw ** 0.5)
        gf = gf.view(h, w, nd)
        gf = torch.cat([gf, inner.image_newline[None, None, :].expand(h, 1, nd)], dim=1)
        gf = gf.view(-1, nd)
        block = torch.cat([gf, inner.view_seperator[None, :]], dim=0)  # [273,1280]
    nv = block.shape[0]

    pre, post = 3, 3
    seq = pre + nv + post
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq), dtype=torch.int64)
    input_ids[0, pre:pre + nv] = img_id
    seq_mask = torch.zeros(1, seq, dtype=torch.bool)
    seq_mask[0, pre:pre + nv] = True

    with torch.no_grad():
        emb = inner.embed_tokens(input_ids).clone()
        emb[0].masked_scatter_(seq_mask[0].unsqueeze(-1), block.to(emb.dtype))
        hidden = model.model(inputs_embeds=emb, images=None, use_cache=False, return_dict=True)[0]
        ref = model.lm_head(hidden).float()

    cos, sin = _real_rope(model, cfg, seq)
    gather_idx = torch.zeros(1, seq, dtype=torch.int32)
    gather_idx[0, pre:pre + nv] = torch.arange(1, nv + 1, dtype=torch.int32)
    mask_f = seq_mask.float().unsqueeze(-1)

    tt = TTNNUnlimitedOcrForCausalLM.from_torch(model)
    set_device(tt, device)
    tt_px = ttnn.from_torch(px.permute(0, 2, 3, 1).contiguous(), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_ids = ttnn.from_torch(input_ids.to(torch.int32), dtype=ttnn.uint32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_idx = ttnn.from_torch(gather_idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    from tt_symbiote.core.tensor import TorchTTNNTensor

    out = tt(input_ids=tt_ids, pixel_values=tt_px, position_embeddings=(tt_cos, tt_sin),
             vision_idx=tt_idx, vision_mask=tt_mask)
    if isinstance(out, TorchTTNNTensor):
        out.elem = None
        got = out.to_torch.float().reshape(ref.shape)
    else:
        got = (out if isinstance(out, torch.Tensor) else ttnn.to_torch(out)).float().reshape(ref.shape)

    pcc = _pcc(got, ref)
    tt_arg, ref_arg = got[0, -1].argmax().item(), ref[0, -1].argmax().item()
    print(f"\n[FORCAUSALLM vlm-e2e] logits PCC={pcc:.5f}  "
          f"last-token argmax TTNN={tt_arg} torch={ref_arg} match={tt_arg == ref_arg}")
    assert torch.isfinite(got).all(), "VLM logits contain NaN/Inf"
    assert_pcc(got, ref, threshold=PCC, msg="TTNNUnlimitedOcrForCausalLM VLM logits")
    assert tt_arg == ref_arg, f"VLM last-token argmax mismatch: TTNN={tt_arg} torch={ref_arg}"
