# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 (component) PCC tests for the pi0.5 TTNN port.

Each test compares a ``tt_symbiote`` TTNN component against the pi0_5 PyTorch
golden reference on IDENTICAL random weights -- so these run on a single
Blackhole (P150) WITHOUT the gated ``pi05_base`` checkpoint. The ``device``
fixture is provided by the ttnn pytest plugin.

Run:
    TT_METAL_HOME=... pytest tests/capabilities/pi05/test_ops_pi05.py -xvs
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from tt_symbiote.models.pi05.configuration_pi05 import GemmaConfig, SigLIPConfig, SuffixConfig
from tt_symbiote.utils.device_management import set_device

from ..pi05_helpers import SEED, require_reference
from ..pi05_helpers import assert_pcc

PCC = 0.99

# 3-camera deployment prefill length: 3 cameras x 256 SigLIP patches + 128 language
# tokens = 896 (tile-aligned, 28 tiles). The VLM-path component tests run at this
# real seq so regressions on the 896-token path (e.g. sharded-grid sizing) are
# caught at Tier 1. The action expert (suffix) and per-camera SigLIP blocks are
# camera-count-invariant and keep their action/patch-sized seqs.
_N_CAMERAS = 3
_VLM_SEQ = _N_CAMERAS * 256 + 128


# --------------------------------------------------------------------------- MLP
@pytest.mark.parametrize(
    "config_fn,name",
    [(GemmaConfig.gemma_2b, "vlm_2b"), (GemmaConfig.gemma_300m, "expert_300m")],
    ids=["vlm_2b", "expert_300m"],
)
def test_gemma_mlp(dev, config_fn, name):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import GemmaMLP
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05GemmaMLP

    torch.manual_seed(SEED)
    cfg = config_fn()
    weights = {
        "mlp.gate_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.up_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.down_proj.weight": torch.randn(cfg.width, cfg.mlp_dim) * 0.02,
    }
    ref = GemmaMLP(cfg, weights)
    # VLM runs the 3-camera prefill (800 tokens); the expert MLP is action-sized.
    seq = _VLM_SEQ if name == "vlm_2b" else 64
    x = torch.randn(1, seq, cfg.width)
    out_ref = ref.forward(x)

    tt = TTNNPi05GemmaMLP.from_torch(ref, cfg)
    set_device(tt, dev)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt = tt.forward(x_tt)
    assert_pcc(out_tt, out_ref, threshold=PCC, msg=f"GemmaMLP[{name}]")


def _attn_weights(cfg):
    return {
        "self_attn.q_proj.weight": torch.randn(cfg.num_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.k_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.v_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.o_proj.weight": torch.randn(cfg.width, cfg.num_heads * cfg.head_dim) * 0.02,
    }


def _mlp_weights(cfg):
    return {
        "mlp.gate_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.up_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.down_proj.weight": torch.randn(cfg.width, cfg.mlp_dim) * 0.02,
    }


def _meta_rope_sliced(dev, head_dim, seq, base):
    from tt_symbiote.models.pi05.modeling_pi05_common import precompute_freqs_cis_meta

    cos, sin = precompute_freqs_cis_meta(head_dim, max(seq, 64), dev, base)
    cos = ttnn.slice(cos, [0, 0, 0, 0], [1, 1, seq, head_dim])
    sin = ttnn.slice(sin, [0, 0, 0, 0], [1, 1, seq, head_dim])
    return cos, sin


# ----------------------------------------------------------------- Attention
def test_gemma_attention(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import GemmaAttention, precompute_freqs_cis
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05GemmaAttention

    torch.manual_seed(SEED)
    cfg = GemmaConfig.gemma_2b()
    seq = _VLM_SEQ  # 3-camera VLM prefill (800 tokens)
    w = _attn_weights(cfg)
    ref = GemmaAttention(cfg, w, 0)
    x = torch.randn(1, seq, cfg.width) * 0.5
    cos_t, sin_t = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len if hasattr(cfg, "max_seq_len") else 2048, cfg.rope_base)
    out_ref, _ = ref.forward(x, cos_t, sin_t)

    tt = TTNNPi05GemmaAttention.from_torch(ref, cfg)
    set_device(tt, dev)
    cos_s, sin_s = _meta_rope_sliced(dev, cfg.head_dim, seq, cfg.rope_base)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt, _ = tt.forward(x_tt, cos_s, sin_s)
    assert_pcc(out_tt, out_ref, threshold=PCC, msg="GemmaAttention[vlm_2b]")


# ---------------------------------------------------------------- Gemma block
def test_gemma_block(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import GemmaBlock, precompute_freqs_cis
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05GemmaBlock

    torch.manual_seed(SEED)
    cfg = GemmaConfig.gemma_2b()
    seq = _VLM_SEQ  # 3-camera VLM prefill (800 tokens)
    w = {
        "input_layernorm.weight": torch.randn(cfg.width) * 0.02,
        "post_attention_layernorm.weight": torch.randn(cfg.width) * 0.02,
        **_attn_weights(cfg),
        **_mlp_weights(cfg),
    }
    ref = GemmaBlock(cfg, w, 0)
    x = torch.randn(1, seq, cfg.width) * 0.5
    cos_t, sin_t = precompute_freqs_cis(cfg.head_dim, 2048, cfg.rope_base)
    out_ref, _ = ref.forward(x, cos_t, sin_t)

    tt = TTNNPi05GemmaBlock.from_torch(ref, cfg)
    set_device(tt, dev)
    cos_s, sin_s = _meta_rope_sliced(dev, cfg.head_dim, seq, cfg.rope_base)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt, _ = tt.forward(x_tt, cos_s, sin_s)
    assert_pcc(out_tt, out_ref, threshold=PCC, msg="GemmaBlock[vlm_2b]")


# --------------------------------------------------------------- AdaRMS block
def test_adarms_block(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock, precompute_freqs_cis
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05AdaRMSGemmaBlock

    torch.manual_seed(SEED)
    cfg = GemmaConfig.gemma_300m()
    seq = 64  # tile-aligned (real action_horizon=50 is padded to 64 in the full model)
    w = {
        "input_layernorm.dense.weight": torch.randn(3 * cfg.width, cfg.width) * 0.02,
        "input_layernorm.dense.bias": torch.randn(3 * cfg.width) * 0.02,
        "post_attention_layernorm.dense.weight": torch.randn(3 * cfg.width, cfg.width) * 0.02,
        "post_attention_layernorm.dense.bias": torch.randn(3 * cfg.width) * 0.02,
        **_attn_weights(cfg),
        **_mlp_weights(cfg),
    }
    ref = AdaRMSGemmaBlock(cfg, w, 0)
    x = torch.randn(1, seq, cfg.width) * 0.5
    cond = torch.randn(1, cfg.width) * 0.5
    cos_t, sin_t = precompute_freqs_cis(cfg.head_dim, 2048, cfg.rope_base)
    out_ref, _ = ref.forward(x, cos_t, sin_t, cond)

    tt = TTNNPi05AdaRMSGemmaBlock.from_torch(ref, cfg)
    set_device(tt, dev)
    cos_s, sin_s = _meta_rope_sliced(dev, cfg.head_dim, seq, cfg.rope_base)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cond_tt = ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt, _ = tt.forward(x_tt, cos_s, sin_s, cond_tt)
    assert_pcc(out_tt, out_ref, threshold=PCC, msg="AdaRMSGemmaBlock[expert_300m]")


# -------------------------------------------------------------------- Suffix
def test_suffix_embed_actions_and_project(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
    from tt_symbiote.models.pi05.modeling_pi05_suffix import TTNNPi05SuffixEmbedding

    torch.manual_seed(SEED)
    cfg = SuffixConfig()  # action_dim=32, expert_width=1024, action_horizon=50, pi05=True
    weights = {
        "action_in_proj.weight": torch.randn(cfg.expert_width, cfg.action_dim) * 0.02,
        "action_in_proj.bias": torch.randn(cfg.expert_width) * 0.02,
        "action_out_proj.weight": torch.randn(cfg.action_dim, cfg.expert_width) * 0.02,
        "action_out_proj.bias": torch.randn(cfg.action_dim) * 0.02,
        "time_mlp_in.weight": torch.randn(cfg.expert_width, cfg.expert_width) * 0.02,
        "time_mlp_in.bias": torch.randn(cfg.expert_width) * 0.02,
        "time_mlp_out.weight": torch.randn(cfg.expert_width, cfg.expert_width) * 0.02,
        "time_mlp_out.bias": torch.randn(cfg.expert_width) * 0.02,
    }
    ref = Pi0_5SuffixEmbedding(cfg, weights)
    tt = TTNNPi05SuffixEmbedding.from_torch(ref, cfg)
    set_device(tt, dev)

    # action_in_proj
    actions = torch.randn(1, 64, cfg.action_dim) * 0.5
    out_ref = ref.embed_actions(actions)
    a_tt = ttnn.from_torch(actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt = tt.embed_actions(a_tt)
    assert_pcc(out_tt, out_ref, threshold=PCC, msg="Suffix.embed_actions")

    # action_out_proj
    expert_out = torch.randn(1, 64, cfg.expert_width) * 0.5
    proj_ref = ref.project_output(expert_out)
    e_tt = ttnn.from_torch(expert_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    proj_tt = tt.project_output(e_tt)
    assert_pcc(proj_tt, proj_ref, threshold=PCC, msg="Suffix.project_output")


def test_suffix_adarms_cond(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
    from tt_symbiote.models.pi05.modeling_pi05_suffix import TTNNPi05SuffixEmbedding

    torch.manual_seed(SEED)
    cfg = SuffixConfig()
    weights = {
        "action_in_proj.weight": torch.randn(cfg.expert_width, cfg.action_dim) * 0.02,
        "action_in_proj.bias": torch.randn(cfg.expert_width) * 0.02,
        "action_out_proj.weight": torch.randn(cfg.action_dim, cfg.expert_width) * 0.02,
        "action_out_proj.bias": torch.randn(cfg.action_dim) * 0.02,
        "time_mlp_in.weight": torch.randn(cfg.expert_width, cfg.expert_width) * 0.02,
        "time_mlp_in.bias": torch.randn(cfg.expert_width) * 0.02,
        "time_mlp_out.weight": torch.randn(cfg.expert_width, cfg.expert_width) * 0.02,
        "time_mlp_out.bias": torch.randn(cfg.expert_width) * 0.02,
    }
    ref = Pi0_5SuffixEmbedding(cfg, weights)
    cond_ref = ref.embed_timestep_adarms(torch.tensor([0.5]))

    tt = TTNNPi05SuffixEmbedding.from_torch(ref, cfg)
    set_device(tt, dev)
    cond_tt = tt.embed_adarms_cond(torch.tensor([0.5]))
    # sincos timestep embedding is the most bf16-sensitive op; allow 0.98.
    assert_pcc(cond_tt, cond_ref, threshold=0.98, msg="Suffix.embed_adarms_cond")


# -------------------------------------------------------------------- SigLIP
def _siglip_attn_mlp_ln_weights(cfg):
    h, inter = cfg.hidden_size, cfg.intermediate_size
    return {
        "layer_norm1.weight": torch.randn(h) * 0.02 + 1.0,
        "layer_norm1.bias": torch.randn(h) * 0.02,
        "layer_norm2.weight": torch.randn(h) * 0.02 + 1.0,
        "layer_norm2.bias": torch.randn(h) * 0.02,
        "self_attn.q_proj.weight": torch.randn(h, h) * 0.02,
        "self_attn.q_proj.bias": torch.randn(h) * 0.02,
        "self_attn.k_proj.weight": torch.randn(h, h) * 0.02,
        "self_attn.k_proj.bias": torch.randn(h) * 0.02,
        "self_attn.v_proj.weight": torch.randn(h, h) * 0.02,
        "self_attn.v_proj.bias": torch.randn(h) * 0.02,
        "self_attn.out_proj.weight": torch.randn(h, h) * 0.02,
        "self_attn.out_proj.bias": torch.randn(h) * 0.02,
        "mlp.fc1.weight": torch.randn(inter, h) * 0.02,
        "mlp.fc1.bias": torch.randn(inter) * 0.02,
        "mlp.fc2.weight": torch.randn(h, inter) * 0.02,
        "mlp.fc2.bias": torch.randn(h) * 0.02,
    }


def test_siglip_mlp(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_siglip import SigLIPMLP
    from tt_symbiote.models.pi05.modeling_pi05_siglip import TTNNPi05SigLIPMLP

    torch.manual_seed(SEED)
    cfg = SigLIPConfig()
    w = _siglip_attn_mlp_ln_weights(cfg)
    ref = SigLIPMLP(cfg, w)
    x = torch.randn(1, cfg.num_patches, cfg.hidden_size) * 0.5
    out_ref = ref.forward(x)

    tt = TTNNPi05SigLIPMLP.from_torch(ref, cfg)
    set_device(tt, dev)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt = tt.forward(x_tt)
    assert_pcc(out_tt, out_ref, threshold=PCC, msg="SigLIPMLP")


def test_siglip_block(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_siglip import SigLIPBlock
    from tt_symbiote.models.pi05.modeling_pi05_siglip import TTNNPi05SigLIPBlock

    torch.manual_seed(SEED)
    cfg = SigLIPConfig()
    w = _siglip_attn_mlp_ln_weights(cfg)
    ref = SigLIPBlock(cfg, w)
    x = torch.randn(1, cfg.num_patches, cfg.hidden_size) * 0.5
    out_ref = ref.forward(x)

    tt = TTNNPi05SigLIPBlock.from_torch(ref, cfg)
    set_device(tt, dev)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt = tt.forward(x_tt)
    assert_pcc(out_tt, out_ref, threshold=PCC, msg="SigLIPBlock")
