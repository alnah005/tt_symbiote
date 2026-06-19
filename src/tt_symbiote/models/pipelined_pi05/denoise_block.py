# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import Optional, Tuple

import ttnn

from tt_symbiote.core.module import DeviceArch, run_on_devices
from tt_symbiote.core.run_config import trace_enabled
from tt_symbiote.models.pi05.modeling_pi05_bs import matmul_pcfg, sdpa_program_config
from tt_symbiote.models.pi05.modeling_pi05_common import get_sdpa_compute_kernel_config
from tt_symbiote.models.pi05.modeling_pi05_gemma import (
    TTNNPi05AdaRMSGemmaBlock,
    TTNNPi05GemmaAttention,
    TTNNPi05GemmaMLP,
)

TT_METAL_COMMIT = "7d1b555a394b7b5413444fc95e5c2d4a54ec197a"
_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


_DENOISE_TUNE_TABLE_BASE = {
    (64, 32): (120, 32),
    (128, 32): (24, 32),
}
_DENOISE_TUNE_TABLE_V2 = {
    (64, 32): (120, 32),
    (128, 32): (24, 32),
    (32, 80): (64, 8),
}


def _denoise_tuned_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, *, activation=None):
    if m_tiles != 1 or not _flag("PI0_DENOISE_MM_TUNE"):
        return None
    table = _DENOISE_TUNE_TABLE_V2 if _flag("PI0_MM_SWEEP_V2") else _DENOISE_TUNE_TABLE_BASE
    override = table.get((k_tiles, n_tiles))
    if override is None:
        return None
    num_cores, in0_bw = override
    if k_tiles % in0_bw != 0:
        return None
    per_core_N = (n_tiles + num_cores - 1) // num_cores if n_tiles % num_cores else n_tiles // num_cores
    eff_budget = 4
    out_sw = min(per_core_N, eff_budget)
    while out_sw > 1 and per_core_N % out_sw != 0:
        out_sw -= 1
    out_sh = max(1, eff_budget // out_sw)
    out_sh = min(m_tiles, out_sh)
    while out_sh > 1 and m_tiles % out_sh != 0:
        out_sh -= 1
    cfg_gx = min(grid_x, num_cores)
    cfg_gy = (num_cores + cfg_gx - 1) // cfg_gx
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cfg_gx, cfg_gy),
        in0_block_w=in0_bw,
        out_subblock_h=out_sh,
        out_subblock_w=out_sw,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=True,
    )


def _expert_lofi_ck():
    if not _flag("PI0_EXPERT_MM_LOFI"):
        return None
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _denoise_sdpa_pcfg(q_seq, kv_seq, grid_x, grid_y):
    q_chunk = 32
    k_force = os.environ.get("PI0_SDPA_DENOISE_K_FORCE", "").strip()
    k_chunk = None
    kv_aligned = ((kv_seq + 31) // 32) * 32
    if k_force:
        kf = int(k_force)
        if kv_aligned % kf == 0:
            k_chunk = kf
    if k_chunk is None:
        for cand in (256, 128, 96, 64, 32):
            if kv_aligned % cand == 0:
                k_chunk = cand
                break
        k_chunk = k_chunk or 32
    return sdpa_program_config(q_seq, kv_seq, grid_x, grid_y, q_chunk=q_chunk, k_chunk=k_chunk)


class TTNNPi05DenoiseExpertAttention(TTNNPi05GemmaAttention):
    def _proj(self, x, w, dtype, m_t, grid, ck):
        k_t, n_t = w.shape[-2] // 32, w.shape[-1] // 32
        pc = _denoise_tuned_pcfg(m_t, k_t, n_t, grid.x, grid.y) or matmul_pcfg(
            m_t, k_t, n_t, grid.x, grid.y, in0_block_w=8
        )
        if pc is not None:
            return ttnn.linear(x, w, dtype=dtype, memory_config=_L1, program_config=pc, compute_kernel_config=ck)
        cg = ttnn.CoreGrid(y=1, x=grid.x)
        return ttnn.linear(x, w, dtype=dtype, memory_config=_L1, core_grid=cg, compute_kernel_config=ck)

    @run_on_devices(DeviceArch.P150)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        if len(hidden_states.shape) == 3:
            b, s, _ = hidden_states.shape
            hidden_states = ttnn.reshape(hidden_states, (b, 1, s, hidden_states.shape[-1]))
        else:
            b, _, s, _ = hidden_states.shape

        _g = self.device.compute_with_storage_grid_size()
        _expert_ck = _expert_lofi_ck()

        m_t = s // 32
        qkv = self._proj(hidden_states, self.tt_wqkv, ttnn.bfloat8_b, m_t, _g, _expert_ck)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=_L1,
        )
        ttnn.deallocate(qkv)

        q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=_L1)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=_L1)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ttnn.concat([past_k, k], dim=2, memory_config=_L1)
            v = ttnn.concat([past_v, v], dim=2, memory_config=_L1)
        new_cache = (k, v) if use_cache else None

        kv_seq = k.shape[-2]
        _sdpa_kwargs = {"memory_config": _L1}
        # Clamp the SDPA grid to its work units (num_heads * q_chunks); the small-q shape
        # otherwise over-parallelizes the full device grid.
        _sdpa_cores = min(_g.x, self.num_heads * ((q.shape[-2] + 31) // 32))
        _spc = _denoise_sdpa_pcfg(q.shape[-2], kv_seq, _sdpa_cores, 1)
        if _spc is not None:
            _sdpa_kwargs["program_config"] = _spc
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=get_sdpa_compute_kernel_config(),
            **_sdpa_kwargs,
        )
        ttnn.deallocate(q)

        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=_L1)
        out = self._proj(attn_out, self.tt_o, ttnn.bfloat16, m_t, _g, _expert_ck)
        ttnn.deallocate(attn_out)
        if len(out.shape) == 4:
            out = ttnn.reshape(out, (b, s, out.shape[-1]))
        return out, new_cache


class TTNNPi05DenoiseExpertMLP(TTNNPi05GemmaMLP):
    @run_on_devices(DeviceArch.P150)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not _flag("PI0_DENOISE_MLP_WIDE"):
            return super().forward(x)
        mt = x.shape[-2] // 32
        k_gu = self.tt_gate.shape[-2] // 32
        n_gu = self.tt_gate.shape[-1] // 32
        k_dn = self.tt_down.shape[-2] // 32
        n_dn = self.tt_down.shape[-1] // 32
        gate_pc = matmul_pcfg(mt, k_gu, n_gu, 12, 10, activation=(ttnn.UnaryOpType.GELU, True))
        up_pc = matmul_pcfg(mt, k_gu, n_gu, 12, 10)
        down_pc = matmul_pcfg(mt, k_dn, n_dn, 12, 10)
        if gate_pc is None or up_pc is None or down_pc is None:
            return super().forward(x)
        gate = ttnn.linear(x, self.tt_gate, dtype=ttnn.bfloat8_b, memory_config=_L1, program_config=gate_pc)
        up = ttnn.linear(x, self.tt_up, dtype=ttnn.bfloat8_b, memory_config=_L1, program_config=up_pc)
        prod = ttnn.multiply(gate, up, memory_config=_L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(prod, self.tt_down, dtype=ttnn.bfloat8_b, memory_config=_L1, program_config=down_pc)
        ttnn.deallocate(prod)
        return out


@trace_enabled
class TTNNPi05DenoiseExpertBlock(TTNNPi05AdaRMSGemmaBlock):
    @classmethod
    def from_torch(cls, block, config):
        new = super().from_torch(block, config)
        new.attention = TTNNPi05DenoiseExpertAttention.from_torch(block.attention, config)
        if _flag("PI0_DENOISE_MLP_WIDE"):
            new.mlp = TTNNPi05DenoiseExpertMLP.from_torch(block.mlp, config)
        return new

    def move_weights_to_device_impl(self):
        # L1-resident projection weights (each stage holds <=5 layers, fits in L1): removes the
        # per-matmul DRAM weight read. Block-level so it applies whichever MLP subclass is bound.
        super().move_weights_to_device_impl()
        a, m = self.attention, self.mlp
        a.tt_wqkv = ttnn.to_memory_config(a.tt_wqkv, _L1)
        a.tt_o = ttnn.to_memory_config(a.tt_o, _L1)
        m.tt_gate = ttnn.to_memory_config(m.tt_gate, _L1)
        m.tt_up = ttnn.to_memory_config(m.tt_up, _L1)
        m.tt_down = ttnn.to_memory_config(m.tt_down, _L1)

    @run_on_devices(DeviceArch.P150)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        adarms_cond: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
        precomputed_mod: Optional[Tuple[ttnn.Tensor, ...]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        owned = precomputed_mod is None
        if owned:
            sa1, sha, ga, sf1, shf, gf = self.precompute_mods(adarms_cond)
        else:
            sa1, sha, ga, sf1, shf, gf = precomputed_mod

        normed = self._apply_ada(hidden_states, sa1, sha, self._eps)
        attn_out, new_cache = self.attention(normed, cos, sin, attention_mask, past_key_value, use_cache)
        ttnn.deallocate(normed)
        gated_attn = ttnn.multiply(ga, attn_out, memory_config=_L1)
        ttnn.deallocate(attn_out)
        hidden_states = ttnn.add(hidden_states, gated_attn, memory_config=_L1)
        ttnn.deallocate(gated_attn)

        normed = self._apply_ada(hidden_states, sf1, shf, self._eps)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        gated_ffw = ttnn.multiply(gf, mlp_out, memory_config=_L1)
        ttnn.deallocate(mlp_out)
        hidden_states = ttnn.add(hidden_states, gated_ffw, memory_config=_L1)
        ttnn.deallocate(gated_ffw)

        if owned:
            for ten in (sa1, sha, ga, sf1, shf, gf):
                ttnn.deallocate(ten)
        return hidden_states, new_cache
