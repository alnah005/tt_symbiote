# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

os.environ.setdefault("MESH_DEVICE", "P150")

import pi05_helpers
import torch

import ttnn
from tracy import signpost

from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
from tt_symbiote.models.pipelined_pi05.denoise_pipeline import (
    TTNNPi05DenoiseExpertBlock,
    build_denoise_loop_pipeline,
    build_denoise_pipeline,
    euler_schedule,
    perf_action_horizon,
    perf_suffix_len,
)

SEED = 42
_PREFIX_LEN = int(os.environ.get("PI05_PREFIX_LEN", "256"))
_ACTION_DIM = 32


def _expert_block_w(W, mlp_dim, head_dim, num_heads, num_kv_heads):
    qkv_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim
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


def _build_inputs(config, ah, suffix_len):
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
    adarms_cond = ref_suffix.embed_timestep_adarms(torch.tensor([0.5]))

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
    return ref_blocks, final_mod_w, final_mod_b, ref_suffix, x_t, adarms_cond, prefix_kv, mask, ref_suffix


def _open_parent():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=134_217_728)


def _close_parent(parent):
    for submesh in parent.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    mode = os.environ.get("PI05_PROFILE_MODE", "per_step")
    ah = perf_action_horizon()
    suffix_len = perf_suffix_len(ah)
    n_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", 5))
    config = Pi0_5ModelConfig()
    suffix_config = config.suffix_config
    ref_blocks, fmw, fmb, ref_suffix, x_t, adarms_cond, prefix_kv, mask, _ = _build_inputs(config, ah, suffix_len)

    parent = _open_parent()
    obj = None
    try:
        if mode == "per_step":
            os.environ.setdefault("TT_SYMBIOTE_RUN_MODE", "NORMAL")
            pipe = obj = build_denoise_pipeline(
                ref_blocks,
                fmw,
                fmb,
                ref_suffix,
                config,
                suffix_config,
                parent,
                adarms_cond_torch=adarms_cond,
                prefix_kv_cache=prefix_kv,
                prefix_len=_PREFIX_LEN,
                suffix_len=suffix_len,
                attention_mask_torch=mask,
                position_offset=_PREFIX_LEN,
                splits=(5, 5, 4, 4),
                block_cls=TTNNPi05DenoiseExpertBlock,
                use_concat_kv=True,
            )
            x_dev = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=pipe.meshes[0])
            pipe(x_dev)
            ttnn.synchronize_device(pipe.meshes[-1])
            signpost(header=f"denoise_perstep_ah{ah}_M{suffix_len // 32}")
            out = pipe(x_dev)
            ttnn.synchronize_device(pipe.meshes[-1])
            signpost(header="denoise_perstep_end")
            print(f"[per_step ah={ah} M={suffix_len // 32}] out shape", tuple(out.shape))
        else:
            os.environ["TT_SYMBIOTE_RUN_MODE"] = "TRACED"
            timesteps, _ = euler_schedule(n_steps)
            conds = [ref_suffix.embed_timestep_adarms(torch.tensor([timesteps[i]])) for i in range(n_steps)]
            drv = obj = build_denoise_loop_pipeline(
                ref_blocks,
                fmw,
                fmb,
                ref_suffix,
                config,
                suffix_config,
                parent,
                adarms_cond_per_step=conds,
                prefix_kv_cache=prefix_kv,
                prefix_len=_PREFIX_LEN,
                suffix_len=suffix_len,
                attention_mask_torch=mask,
                position_offset=_PREFIX_LEN,
                num_steps=n_steps,
                action_horizon=ah,
                splits=(5, 5, 4, 4),
                block_cls=TTNNPi05DenoiseExpertBlock,
                use_concat_kv=True,
                drain="all",
            )
            drv.stream_euler(x_t, capture=True)
            signpost(header=f"denoise_streamed_N{n_steps}_ah{ah}")
            out = drv.replay()
            signpost(header="denoise_streamed_end")
            print(f"[streamed N={n_steps} ah={ah}] out shape", tuple(out.shape))
    finally:
        if obj is not None:
            obj.close()
        _close_parent(parent)


if __name__ == "__main__":
    main()
