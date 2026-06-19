# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Replay wall-clock latency for the fused 5-step denoise loop trace (one trace per submesh).

Production config: 1024-token VLM prefix, 5-step euler denoise, 30 timed replays. Weights-on-L1
and the SDPA grid-clamp are model defaults (denoise_block.py). The mesh is opened/closed by the
``denoise_parent_mesh`` pytest fixture (conftest.py); this test only builds, captures, and times.

replay() wall time = host dispatch + on-device exec + a single synchronize + action readback.
"""
from __future__ import annotations

import os
import statistics
import time

import torch

import ttnn

from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
from tt_symbiote.models.pipelined_pi05.denoise_pipeline import (
    TTNNPi05DenoiseExpertBlock,
    build_denoise_loop_pipeline,
    euler_schedule,
    perf_action_horizon,
    perf_suffix_len,
)

from ..pi05_helpers import SEED, require_reference

_PREFIX_LEN = 1024  # production VLM prefix length
_ACTION_DIM = 32
_N_STEPS = 5
_N_REPLAYS = 30
_LATENCY_CEILING_MS = 40.0  # generous regression bound; the real number is printed


def _expert_block_w(W, mlp_dim, head_dim, num_heads, num_kv_heads):
    qkv_out, kv_out = num_heads * head_dim, num_kv_heads * head_dim
    return {
        "input_layernorm.dense.weight": torch.randn(3 * W, W) * 0.02,
        "input_layernorm.dense.bias": torch.randn(3 * W) * 0.02,
        "post_attention_layernorm.dense.weight": torch.randn(3 * W, W) * 0.02,
        "post_attention_layernorm.dense.bias": torch.randn(3 * W) * 0.02,
        "self_attn.q_proj.weight": torch.randn(qkv_out, W) * 0.02,
        "self_attn.k_proj.weight": torch.randn(kv_out, W) * 0.02,
        "self_attn.v_proj.weight": torch.randn(kv_out, W) * 0.02,
        "self_attn.o_proj.weight": torch.randn(W, qkv_out) * 0.02,
        "mlp.gate_proj.weight": torch.randn(mlp_dim, W) * 0.02,
        "mlp.up_proj.weight": torch.randn(mlp_dim, W) * 0.02,
        "mlp.down_proj.weight": torch.randn(W, mlp_dim) * 0.02,
    }


def _suffix_w(W, action_dim):
    return {
        "action_in_proj.weight": torch.randn(W, action_dim) * 0.02,
        "action_in_proj.bias": torch.randn(W) * 0.02,
        "action_out_proj.weight": torch.randn(action_dim, W) * 0.02,
        "action_out_proj.bias": torch.randn(action_dim) * 0.02,
        "time_mlp_in.weight": torch.randn(W, W) * 0.02,
        "time_mlp_in.bias": torch.randn(W) * 0.02,
        "time_mlp_out.weight": torch.randn(W, W) * 0.02,
        "time_mlp_out.bias": torch.randn(W) * 0.02,
    }


def _build_inputs(config, ah, suffix_len, n_steps):
    from models.experimental.pi0_5.common.configs import GemmaConfig as RefGemmaConfig
    from models.experimental.pi0_5.common.configs import SuffixConfig as RefSuffixConfig
    from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock, apply_rotary_emb, precompute_freqs_cis
    from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding

    ec = config.expert_config
    W, head_dim, num_kv_heads = ec.width, ec.head_dim, ec.num_kv_heads
    ref_ec = RefGemmaConfig.gemma_300m()
    ref_scfg = RefSuffixConfig(action_dim=_ACTION_DIM, action_horizon=ah, expert_width=W, pi05=True)

    torch.manual_seed(SEED)
    bw = [_expert_block_w(W, ec.mlp_dim, head_dim, ec.num_heads, num_kv_heads) for _ in range(18)]
    ref_blocks = [AdaRMSGemmaBlock(ref_ec, bw[i], i) for i in range(18)]
    ref_suffix = Pi0_5SuffixEmbedding(ref_scfg, _suffix_w(W, _ACTION_DIM))
    final_mod_w = torch.randn(3 * W, W) * 0.02
    final_mod_b = torch.randn(3 * W) * 0.02

    torch.manual_seed(SEED + 100)
    x_t = torch.randn(1, suffix_len, _ACTION_DIM) * 0.5
    x_t[:, ah:, :] = 0.0
    timesteps, _ = euler_schedule(n_steps)
    conds = [ref_suffix.embed_timestep_adarms(torch.tensor([timesteps[i]])) for i in range(n_steps)]

    cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, base=ec.rope_base)
    pid_pre = torch.arange(_PREFIX_LEN).unsqueeze(0)
    mask = torch.zeros(1, 1, suffix_len, _PREFIX_LEN + suffix_len)
    mask[:, :, :, _PREFIX_LEN + ah :] = -1e4

    torch.manual_seed(SEED + 200)
    prefix_kv = []
    for _ in range(18):
        k = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        v = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        k_roped, _ = apply_rotary_emb(k, k.clone(), cos, sin, position_ids=pid_pre)
        prefix_kv.append((k_roped, v))
    return ref_blocks, final_mod_w, final_mod_b, ref_suffix, x_t, conds, prefix_kv, mask


def test_replay_walltime(denoise_parent_mesh):
    require_reference()
    assert os.environ.get("MESH_DEVICE") == "P150"
    ah = perf_action_horizon()
    suffix_len = perf_suffix_len(ah)
    config = Pi0_5ModelConfig()
    ref_blocks, fmw, fmb, ref_suffix, x_t, conds, prefix_kv, mask = _build_inputs(config, ah, suffix_len, _N_STEPS)

    drv = None
    try:
        os.environ["TT_SYMBIOTE_RUN_MODE"] = "TRACED"
        drv = build_denoise_loop_pipeline(
            ref_blocks,
            fmw,
            fmb,
            ref_suffix,
            config,
            config.suffix_config,
            denoise_parent_mesh,
            adarms_cond_per_step=conds,
            prefix_kv_cache=prefix_kv,
            prefix_len=_PREFIX_LEN,
            suffix_len=suffix_len,
            attention_mask_torch=mask,
            position_offset=_PREFIX_LEN,
            num_steps=_N_STEPS,
            action_horizon=ah,
            splits=(5, 5, 4, 4),
            block_cls=TTNNPi05DenoiseExpertBlock,
            use_concat_kv=True,
            drain="all",
        )
        drv.stream_euler(x_t, capture=True)  # capture the fused per-submesh loop trace
        for _ in range(3):
            drv.replay()  # warm-up / stabilize

        times = []
        for _ in range(_N_REPLAYS):
            t0 = time.perf_counter()
            drv.replay()  # replay_loop + single synchronize + action readback
            times.append((time.perf_counter() - t0) * 1e3)
        md = statistics.median(times)
        print(
            f"[replay-walltime prefix={_PREFIX_LEN} N={_N_STEPS} M={suffix_len // 32}] "
            f"min={min(times):.3f} median={md:.3f} mean={statistics.mean(times):.3f} ms "
            f"({md / _N_STEPS:.3f} ms/step) over {_N_REPLAYS} reps"
        )
        assert md < _LATENCY_CEILING_MS, f"5-step replay {md:.2f} ms exceeds {_LATENCY_CEILING_MS} ms ceiling"
    finally:
        os.environ.pop("TT_SYMBIOTE_RUN_MODE", None)
        if drv is not None:
            drv.close()
