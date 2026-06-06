# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-TTNNModule tracy profiler for pi0.5 (bottom-up).

Instantiates each pi0.5 TTNNModule with random weights at its realistic shape,
warms up (compile) OUTSIDE the signpost, then runs N iterations INSIDE a
per-module ``signpost`` so tt-perf-report / CSV parsing yields the real
DEVICE KERNEL DURATION per module and per op.

    MESH_DEVICE=P150 TT_METAL_HOME=... PI05_REFERENCE_ROOT=... \
      python_env/bin/python -m tracy -p -r -v --op-support-count 100000 \
      tests/capabilities/pi05/profile_modules.py
"""

import os
import sys

_REF = os.environ.get("PI05_REFERENCE_ROOT", "/home/ttuser/salnahari/pi0_5_ref")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

import torch
import ttnn
from tracy import signpost

from tt_symbiote.models.pi05.configuration_pi05 import GemmaConfig, SigLIPConfig, SuffixConfig
from tt_symbiote.models.pi05.modeling_pi05_common import precompute_freqs_cis_meta
from tt_symbiote.utils.device_management import set_device

N = 10


def _rope(dev, head_dim, seq, base=10000.0):
    cos, sin = precompute_freqs_cis_meta(head_dim, max(seq, 64), dev, base)
    return ttnn.slice(cos, [0, 0, 0, 0], [1, 1, seq, head_dim]), ttnn.slice(sin, [0, 0, 0, 0], [1, 1, seq, head_dim])


def run(dev):
    from models.experimental.pi0_5.reference import torch_gemma as RG, torch_siglip as RS, torch_suffix as RSf

    def prof(label, fn):
        fn()
        ttnn.synchronize_device(dev)  # warmup/compile OUTSIDE signpost
        signpost(header=f"MOD_{label}")
        for _ in range(N):
            fn()
        ttnn.synchronize_device(dev)

    # ---------- Gemma leaf modules: expert (seq64) + VLM (seq288) ----------
    from tt_symbiote.models.pi05.modeling_pi05_gemma import (
        TTNNPi05GemmaMLP, TTNNPi05GemmaAttention, TTNNPi05GemmaBlock, TTNNPi05AdaRMSGemmaBlock,
    )
    # VLM at the 3-camera prefill (3*256 patches + 32 lang = 800); expert action-sized (64).
    for tag, cfg, seq in [("expert", GemmaConfig.gemma_300m(), 64), ("vlm", GemmaConfig.gemma_2b(), 800)]:
        torch.manual_seed(0)
        w = {
            "input_layernorm.weight": torch.randn(cfg.width) * 0.02,
            "post_attention_layernorm.weight": torch.randn(cfg.width) * 0.02,
            "self_attn.q_proj.weight": torch.randn(cfg.num_heads * cfg.head_dim, cfg.width) * 0.02,
            "self_attn.k_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
            "self_attn.v_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
            "self_attn.o_proj.weight": torch.randn(cfg.width, cfg.num_heads * cfg.head_dim) * 0.02,
            "mlp.gate_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
            "mlp.up_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
            "mlp.down_proj.weight": torch.randn(cfg.width, cfg.mlp_dim) * 0.02,
        }
        x = torch.randn(1, seq, cfg.width) * 0.5
        cos, sin = _rope(dev, cfg.head_dim, seq, cfg.rope_base)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        mlp = TTNNPi05GemmaMLP.from_torch(RG.GemmaMLP(cfg, w), cfg); set_device(mlp, dev)
        prof(f"GemmaMLP_{tag}", lambda: mlp.forward(x_tt))
        attn = TTNNPi05GemmaAttention.from_torch(RG.GemmaAttention(cfg, w, 0), cfg); set_device(attn, dev)
        prof(f"GemmaAttn_{tag}", lambda: attn.forward(x_tt, cos, sin)[0])
        blk = TTNNPi05GemmaBlock.from_torch(RG.GemmaBlock(cfg, w, 0), cfg); set_device(blk, dev)
        prof(f"GemmaBlock_{tag}", lambda: blk.forward(x_tt, cos, sin)[0])

    # ---------- AdaRMS expert block (seq64) ----------
    cfg = GemmaConfig.gemma_300m(); seq = 64
    torch.manual_seed(0)
    wa = {
        "input_layernorm.dense.weight": torch.randn(3 * cfg.width, cfg.width) * 0.02,
        "input_layernorm.dense.bias": torch.randn(3 * cfg.width) * 0.02,
        "post_attention_layernorm.dense.weight": torch.randn(3 * cfg.width, cfg.width) * 0.02,
        "post_attention_layernorm.dense.bias": torch.randn(3 * cfg.width) * 0.02,
        "self_attn.q_proj.weight": torch.randn(cfg.num_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.k_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.v_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.o_proj.weight": torch.randn(cfg.width, cfg.num_heads * cfg.head_dim) * 0.02,
        "mlp.gate_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.up_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.down_proj.weight": torch.randn(cfg.width, cfg.mlp_dim) * 0.02,
    }
    x = torch.randn(1, seq, cfg.width) * 0.5
    cond = torch.randn(1, cfg.width) * 0.5
    cos, sin = _rope(dev, cfg.head_dim, seq, cfg.rope_base)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cond_tt = ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ab = TTNNPi05AdaRMSGemmaBlock.from_torch(RG.AdaRMSGemmaBlock(cfg, wa, 0), cfg); set_device(ab, dev)
    prof("AdaRMSBlock_expert", lambda: ab.forward(x_tt, cos, sin, cond_tt)[0])

    # ---------- SigLIP modules (seq256) ----------
    from tt_symbiote.models.pi05.modeling_pi05_siglip import (
        TTNNPi05SigLIPMLP, TTNNPi05SigLIPAttention, TTNNPi05SigLIPBlock, TTNNPi05MultiModalProjector,
    )
    sc = SigLIPConfig(); h, inter = sc.hidden_size, sc.intermediate_size
    torch.manual_seed(0)
    ws = {
        "layer_norm1.weight": torch.randn(h) * 0.02 + 1, "layer_norm1.bias": torch.randn(h) * 0.02,
        "layer_norm2.weight": torch.randn(h) * 0.02 + 1, "layer_norm2.bias": torch.randn(h) * 0.02,
        "self_attn.q_proj.weight": torch.randn(h, h) * 0.02, "self_attn.q_proj.bias": torch.randn(h) * 0.02,
        "self_attn.k_proj.weight": torch.randn(h, h) * 0.02, "self_attn.k_proj.bias": torch.randn(h) * 0.02,
        "self_attn.v_proj.weight": torch.randn(h, h) * 0.02, "self_attn.v_proj.bias": torch.randn(h) * 0.02,
        "self_attn.out_proj.weight": torch.randn(h, h) * 0.02, "self_attn.out_proj.bias": torch.randn(h) * 0.02,
        "mlp.fc1.weight": torch.randn(inter, h) * 0.02, "mlp.fc1.bias": torch.randn(inter) * 0.02,
        "mlp.fc2.weight": torch.randn(h, inter) * 0.02, "mlp.fc2.bias": torch.randn(h) * 0.02,
    }
    xs = ttnn.from_torch(torch.randn(1, sc.num_patches, h) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    smlp = TTNNPi05SigLIPMLP.from_torch(RS.SigLIPMLP(sc, ws), sc); set_device(smlp, dev)
    prof("SigLIP_MLP", lambda: smlp.forward(xs))
    sattn = TTNNPi05SigLIPAttention.from_torch(RS.SigLIPAttention(sc, ws), sc); set_device(sattn, dev)
    prof("SigLIP_Attn", lambda: sattn.forward(xs))
    sblk = TTNNPi05SigLIPBlock.from_torch(RS.SigLIPBlock(sc, ws), sc); set_device(sblk, dev)
    prof("SigLIP_Block", lambda: sblk.forward(xs))
    proj_w = {"linear.weight": torch.randn(2048, h) * 0.02, "linear.bias": torch.randn(2048) * 0.02}
    proj = TTNNPi05MultiModalProjector.from_torch(RS.MultiModalProjector(proj_w)); set_device(proj, dev)
    prof("MMProjector", lambda: proj.forward(xs))

    # ---------- Suffix (action_in/out, adarms_cond) ----------
    from tt_symbiote.models.pi05.modeling_pi05_suffix import TTNNPi05SuffixEmbedding
    su = SuffixConfig()
    torch.manual_seed(0)
    wsf = {
        "action_in_proj.weight": torch.randn(su.expert_width, su.action_dim) * 0.02, "action_in_proj.bias": torch.randn(su.expert_width) * 0.02,
        "action_out_proj.weight": torch.randn(su.action_dim, su.expert_width) * 0.02, "action_out_proj.bias": torch.randn(su.action_dim) * 0.02,
        "time_mlp_in.weight": torch.randn(su.expert_width, su.expert_width) * 0.02, "time_mlp_in.bias": torch.randn(su.expert_width) * 0.02,
        "time_mlp_out.weight": torch.randn(su.expert_width, su.expert_width) * 0.02, "time_mlp_out.bias": torch.randn(su.expert_width) * 0.02,
    }
    suf = TTNNPi05SuffixEmbedding.from_torch(RSf.Pi0_5SuffixEmbedding(su, wsf), su); set_device(suf, dev)
    act = ttnn.from_torch(torch.randn(1, 64, su.action_dim) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    eo = ttnn.from_torch(torch.randn(1, 64, su.expert_width) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    prof("Suffix_action_in", lambda: suf.embed_actions(act))
    prof("Suffix_action_out", lambda: suf.project_output(eo))
    prof("Suffix_adarms_cond", lambda: suf.embed_adarms_cond(suf._ts(0.9) if hasattr(suf, "_ts") else __import__("torch").tensor([0.9])))
    signpost(header="MOD_END")
    print("per-module profile done")


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=134_217_728)
    try:
        run(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
