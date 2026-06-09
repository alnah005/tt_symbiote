# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN Gemma building blocks for the pi0.5 dual-expert backbone.

Ported into the :class:`~tt_symbiote.core.module.TTNNModule` lifecycle from the
tt-metal reference ``models/experimental/pi0_5/tt/ttnn_gemma.py`` and validated
against the PyTorch golden ``reference/torch_gemma.py`` (branch ``tt/pi0.5_bh``).

Classes
-------
* :class:`TTNNPi05GemmaMLP` -- GeGLU MLP (``down(gelu(gate(x)) * up(x))``).
* :class:`TTNNPi05GemmaAttention` -- MQA (8 Q heads / 1 KV head, head_dim 256)
  with fused QKV, RoPE (meta-format cos/sin), SDPA and optional KV-cache concat.
* :class:`TTNNPi05GemmaBlock` -- pre-norm decoder block (plain Gemma RMSNorm).
* :class:`TTNNPi05AdaRMSGemmaBlock` -- action-expert block with adaptive RMSNorm
  (scale/shift/gate modulation from ``adarms_cond``) and gated residuals.

Conventions
-----------
The golden ``reference/torch_*`` classes are *plain Python objects* (not
``nn.Module``) that hold raw weight tensors as attributes, so ``from_torch``
plucks tensors directly. Gemma RMSNorm uses the ``(weight + 1.0)`` offset; that
``+1.0`` is folded into the device weight during ``preprocess_weights_impl``.

``ttnn.linear`` computes ``x @ w`` (w is ``[in, out]``); torch weights are
``[out, in]``, so linear weights are transposed on host before upload.

VALIDATION STATUS: attention (RoPE + SDPA + KV-cache) and the adaRMS modulation
path require on-device PCC validation against the torch reference (Tier 1/2 of
the pcc-test-gen stage). The simple MLP / RMSNorm / projection paths mirror the
already-validated ``gemma4`` port and are low-risk.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import ttnn

from tt_symbiote.core.module import DeviceArch, StatefulTTNNModule, StatelessTTNNModule, run_on_devices
from tt_symbiote.core.run_config import trace_enabled

from .configuration_pi05 import GemmaConfig

# Runtime tt-metal commit the model executes against (installed/built ttnn, main).
# Reference patterns ported from branch tt/pi0.5_bh @ b0703a56465989da179480c93a8992ec519e1cde.
TT_METAL_COMMIT = "b2af0cd67b4e92dafeb2d0254e1c5b43c3ec5a25"

__all__ = [
    "TTNNPi05GemmaMLP",
    "TTNNPi05GemmaAttention",
    "TTNNPi05GemmaBlock",
    "TTNNPi05AdaRMSGemmaBlock",
]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG

# Block-sharded Tier-1 optimization layer (tracy-validated program configs; falls
# back to the core_grid/interleaved path when no clean grid divides the shape).
from tt_symbiote.models.pi05.modeling_pi05_bs import matmul_pcfg, sdpa_program_config, sharded_rms_norm  # noqa: E402


# ---------------------------------------------------------------------------
# Weight upload helpers (host-side; allowed to use torch).
# ---------------------------------------------------------------------------
def _linear_weight_to_tt(w: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat8_b) -> ttnn.Tensor:
    """Transpose a torch ``[out, in]`` linear weight to ttnn ``[in, out]`` host tensor."""
    return ttnn.from_torch(w.t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _norm_weight_to_tt(w: torch.Tensor) -> ttnn.Tensor:
    """Gemma RMSNorm weight with the ``+1.0`` offset folded in, shape ``[1, dim]``."""
    folded = (w + 1.0).reshape(1, w.shape[0]).contiguous()
    return ttnn.from_torch(folded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _rms_norm(x: ttnn.Tensor, weight: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Plain RMSNorm using a pre-offset weight (``weight`` already holds ``w+1``)."""
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=_L1)
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=_L1)


# ---------------------------------------------------------------------------
# GeGLU MLP
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05GemmaMLP(StatelessTTNNModule):
    """Gemma GeGLU MLP: ``down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))``.

    Reference: ``reference/torch_gemma.py::GemmaMLP.forward`` and
    ``tt/ttnn_gemma.py::GemmaMLPTTNN``.
    """

    @classmethod
    def from_torch(cls, mlp, config: Optional[GemmaConfig] = None, weight_dtype=ttnn.bfloat8_b) -> "TTNNPi05GemmaMLP":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = mlp
        new._config = config
        new._wdtype = weight_dtype
        # Plain-object reference holds raw weight tensors.
        new._gate_w = mlp.gate_proj
        new._up_w = mlp.up_proj
        new._down_w = mlp.down_proj
        return new

    def preprocess_weights_impl(self):
        self.tt_gate = _linear_weight_to_tt(self._gate_w, dtype=self._wdtype)
        self.tt_up = _linear_weight_to_tt(self._up_w, dtype=self._wdtype)
        self.tt_down = _linear_weight_to_tt(self._down_w, dtype=self._wdtype)

    def move_weights_to_device_impl(self):
        self.tt_gate = ttnn.to_device(self.tt_gate, self.device, memory_config=_DRAM)
        self.tt_up = ttnn.to_device(self.tt_up, self.device, memory_config=_DRAM)
        self.tt_down = ttnn.to_device(self.tt_down, self.device, memory_config=_DRAM)
        _g = self.device.compute_with_storage_grid_size()
        # down_proj grid is seq-conditional (op-sweep validated, PCC>=0.9998):
        #   small-M expert (64x4096x1024): 1-row width shard ~2.3x vs default.
        #   large-M VLM   (288x16384x2048): FULL 2D grid ~2.3x vs default (1-row is 3x WORSE).
        self._row_cg = ttnn.CoreGrid(y=1, x=_g.x)
        self._full_cg = ttnn.CoreGrid(y=_g.y, x=_g.x)

    @run_on_devices(DeviceArch.P150)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        seq = x.shape[-2]
        # deep-plan_4 widen rung 3 (env-gated, default OFF -> output dtype defaults to
        # input dtype, byte-identical to before). PI05_VLM_MLP_OUT_BF16=1 forces the
        # gate/up/down matmul OUTPUTS to bf16 + a HiFi4+fp32-dest matmul accumulation ck.
        # Weights tt_gate/up/down STAY bf8_b (preprocess_weights_impl, the invariant).
        _mlp_out_dtype = ttnn.bfloat16 if os.environ.get("PI05_VLM_MLP_OUT_BF16") == "1" else None
        _mlp_ck = None
        if _mlp_out_dtype is not None:
            _mlp_ck = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        # EXPERT path (small-M, seq<=96): tuned 2D block-shard program configs on
        # the 8x8 grid with FUSED gelu, mirroring upstream GemmaMLPTTNN. Replaces
        # our prior core_grid=None gate/up (+ separate ttnn.gelu) and 1-row/x-core
        # down -- on identical HW upstream's expert MLP is 0.498 vs our 0.781
        # ms/step on gate/up (same 64 cores; pure program-config + fused-gelu win)
        # and runs `down` on 64 cores vs our 11. Same default fidelity -> PCC-safe.
        # Falls back to the legacy path only when matmul_pcfg declines the shape
        # (reference builders absent / no clean grid). Main path (no env switch).
        if seq <= 96:
            mt = seq // 32
            k_gu, n_gu = self.tt_gate.shape[-2] // 32, self.tt_gate.shape[-1] // 32
            k_dn, n_dn = self.tt_down.shape[-2] // 32, self.tt_down.shape[-1] // 32
            gate_pc = matmul_pcfg(mt, k_gu, n_gu, 8, 8, activation=(ttnn.UnaryOpType.GELU, True))
            up_pc = matmul_pcfg(mt, k_gu, n_gu, 8, 8)
            # Down (64x4096->1024, K-heavy): 8x8 -> 32 cores is the Phase-B-swept
            # optimum (N=32 tiles caps the width-shard; wider/K-split grids regress:
            # (11,2)=27.9us@22c, (11,1)=26.0us@11c, (8,2)=33.4us@16c vs (8,8)=22.0us@32c).
            down_pc = matmul_pcfg(mt, k_dn, n_dn, 8, 8)
            if gate_pc is not None and up_pc is not None and down_pc is not None:
                gate = ttnn.linear(
                    x,
                    self.tt_gate,
                    dtype=_mlp_out_dtype,
                    memory_config=_L1,
                    program_config=gate_pc,
                    compute_kernel_config=_mlp_ck,
                )
                up = ttnn.linear(
                    x,
                    self.tt_up,
                    dtype=_mlp_out_dtype,
                    memory_config=_L1,
                    program_config=up_pc,
                    compute_kernel_config=_mlp_ck,
                )
                hidden = ttnn.multiply(gate, up, memory_config=_L1)
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
                out = ttnn.linear(
                    hidden,
                    self.tt_down,
                    dtype=_mlp_out_dtype,
                    memory_config=_L1,
                    program_config=down_pc,
                    compute_kernel_config=_mlp_ck,
                )
                ttnn.deallocate(hidden)
                return out
        # gate/up: large-M VLM (288x2048x16384) wins on the FULL 2D grid (tracy
        # device: 303us default -> 135us full); small-M expert keeps default.
        gu_cg = self._full_cg if seq > 96 else None
        gate = ttnn.linear(
            x, self.tt_gate, dtype=_mlp_out_dtype, memory_config=_L1, core_grid=gu_cg, compute_kernel_config=_mlp_ck
        )
        # fast_and_approximate_mode: matches the reference's tanh-approx gelu and is
        # ~2.9x faster on the 16384-wide VLM intermediate (96->33us; op-sweep, PCC>=0.999).
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=_L1)
        up = ttnn.linear(
            x, self.tt_up, dtype=_mlp_out_dtype, memory_config=_L1, core_grid=gu_cg, compute_kernel_config=_mlp_ck
        )
        hidden = ttnn.multiply(gate, up, memory_config=_L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # down: large-M VLM -> full grid; small-M expert -> 1-row width shard.
        # (Phase-B note: bf8_b down-proj activation was tested -> only 212->208us
        # (~2%): the VLM down is compute-bound at LoFi, NOT activation-bandwidth-
        # bound, so it does not justify the KV-propagation PCC risk. Not adopted.)
        down_cg = self._row_cg if seq <= 96 else self._full_cg
        out = ttnn.linear(
            hidden,
            self.tt_down,
            dtype=_mlp_out_dtype,
            memory_config=_L1,
            core_grid=down_cg,
            compute_kernel_config=_mlp_ck,
        )
        ttnn.deallocate(hidden)
        return out


# ---------------------------------------------------------------------------
# Multi-Query Attention (8 Q heads, 1 KV head, head_dim 256)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05GemmaAttention(StatefulTTNNModule):
    """Gemma MQA with fused QKV, meta-format RoPE, SDPA and KV cache.

    Reference: ``reference/torch_gemma.py::GemmaAttention.forward`` and
    ``tt/ttnn_gemma.py::GemmaAttentionTTNN.forward``.

    forward(hidden_states, cos, sin, attention_mask=None, past_key_value=None,
            use_cache=False) -> (output, new_cache)

    ``cos``/``sin`` are device tensors in meta format ``[1, 1, seq, head_dim]``
    (built by the backbone via ``precompute_freqs_cis_meta_format``).

    KV-cache modes:
    * STATIC (trace-safe cross-attention, default for the action expert): call
      ``init_static_kv(prefix_len, suffix_len)`` + ``fill_static_prefix(k, v)``
      once outside any trace region; ``forward`` then writes the new suffix K/V
      in-place into the pre-allocated buffer via ``ttnn.fill_cache`` (tile-aligned
      ``update_idx=prefix_len``) and runs SDPA over the full buffer. No ``concat``,
      so the captured per-step graph allocates nothing -- the original blocker
      (``ttnn.concat([past_k, k])`` allocating during TTNN trace capture) is gone.
    * CONCAT (fallback): when no static buffer is set and ``past_key_value`` is
      given, the prefix K/V are concatenated with the new K/V (eager use only).

    VALIDATE ON HW: the fused-QKV split (``nlp_create_qkv_heads``), the RoPE op
    (``ttnn.experimental.rotary_embedding``) and SDPA chunking must be PCC-checked
    against the torch reference. This is the functionality-first (interleaved,
    non-block-sharded) baseline; perf sharding is applied later by op-sweep /
    config-optimize.
    """

    @classmethod
    def from_torch(cls, attn, config: GemmaConfig) -> "TTNNPi05GemmaAttention":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = attn
        new._config = config
        new._q_w = attn.q_proj
        new._k_w = attn.k_proj
        new._v_w = attn.v_proj
        new._o_w = attn.o_proj
        new.num_heads = config.num_heads
        new.num_kv_heads = config.num_kv_heads
        new.head_dim = config.head_dim
        new.scale = 1.0 / math.sqrt(config.head_dim)
        new._eps = config.rms_norm_eps
        # Static KV buffer state (trace-safe cross-attention path). Populated by
        # init_static_kv() / fill_static_prefix() once per inference, outside any
        # trace region. When present, forward() writes the new suffix K/V in-place
        # into these buffers via ttnn.fill_cache (no concat -> no allocation in the
        # captured graph) and runs SDPA on the full buffer.
        new._static_k = None
        new._static_v = None
        new._static_prefix_len = 0
        # VLM-prefill prefix store (trace-safe self-attention path). Distinct from
        # the expert cross-attention buffer above: this store is WRITE-ONLY during
        # prefill -- forward() self-attends over the freshly computed prefix K/V and
        # mirrors them in-place into this pre-allocated buffer so the action expert
        # can cross-attend to them later. Populated by init_vlm_prefix_store()
        # outside any trace region.
        new._vlm_prefix_k = None
        new._vlm_prefix_v = None
        return new

    def preprocess_weights_impl(self):
        # Fuse Q/K/V on host along the output dim so a single matmul produces the
        # concatenated projection; split later via nlp_create_qkv_heads.
        wq = self._q_w.t().contiguous()
        wk = self._k_w.t().contiguous()
        wv = self._v_w.t().contiguous()
        wqkv = torch.cat([wq, wk, wv], dim=-1)  # [in, (Q+K+V)_out]
        self.tt_wqkv = ttnn.from_torch(wqkv, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.tt_o = _linear_weight_to_tt(self._o_w)

    def move_weights_to_device_impl(self):
        self.tt_wqkv = ttnn.to_device(self.tt_wqkv, self.device, memory_config=_DRAM)
        self.tt_o = ttnn.to_device(self.tt_o, self.device, memory_config=_DRAM)
        # 1D width-shard (one core row) for the small-M QKV/o projections
        # (~1.2-1.6x vs default 2D; validated PCC>=0.9998).
        self._row_cg = ttnn.CoreGrid(y=1, x=self.device.compute_with_storage_grid_size().x)

    # ------------------------------------------------------------------ static KV
    def init_static_kv(self, prefix_len: int, suffix_len: int) -> None:
        """Pre-allocate the static cross-attention K/V buffers (once per inference).

        Shape ``[1, num_kv_heads, prefix_len + suffix_len, head_dim]`` in DRAM,
        dtype bfloat8_b (matching the bf8_b QKV-linear K/V output). The buffer
        spans the full attended sequence (prefix + tile-aligned suffix). Called
        OUTSIDE any trace region so the allocation never happens during capture.

        ``prefix_len`` must be a multiple of TILE_HEIGHT (32) so the suffix can be
        written at ``update_idx=prefix_len`` via ``ttnn.fill_cache`` (tile-aligned
        offset requirement). For the canonical pi0.5 prefix (256 image + 32 lang =
        288 = 9*32) this holds.
        """
        import torch as _torch

        if prefix_len % 32 != 0:
            raise ValueError(f"static KV requires a tile-aligned prefix_len; got {prefix_len} (not a multiple of 32)")
        total = prefix_len + suffix_len
        # Reuse an existing same-shape buffer (persistent across inferences so a
        # captured trace keeps pointing at a valid buffer). Only (re)allocate when
        # absent or the attended length changed.
        if self._static_k is not None and self._static_k.shape[2] == total and self._static_prefix_len == prefix_len:
            return
        if self._static_k is not None:
            ttnn.deallocate(self._static_k)
            ttnn.deallocate(self._static_v)
        zeros = _torch.zeros(1, self.num_kv_heads, total, self.head_dim)
        # deep-plan_4 site (d): expert static cross-attn KV buffer. Flips to bf16 with
        # PI05_VLM_KV_BF16=1 (together with site c) so every fill_cache (prefix-copy
        # :311-312, expert suffix :418-419) sees matching bf16 dtypes. Weights stay bf8_b.
        _static_kv_dtype = ttnn.bfloat16 if os.environ.get("PI05_VLM_KV_BF16") == "1" else ttnn.bfloat8_b
        self._static_k = ttnn.from_torch(
            zeros, dtype=_static_kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM
        )
        self._static_v = ttnn.from_torch(
            zeros, dtype=_static_kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM
        )
        self._static_prefix_len = prefix_len

    def fill_static_prefix(self, past_k: ttnn.Tensor, past_v: ttnn.Tensor) -> None:
        """Copy the (already RoPE'd) prefix K/V into the static buffer prefix region.

        Done ONCE per inference, outside the trace region (the prefix is constant
        across all denoise steps). ``ttnn.fill_cache`` is an in-place device write
        at ``update_idx=0``; it allocates nothing.
        """
        ttnn.fill_cache(self._static_k, past_k, 0, update_idx=0)
        ttnn.fill_cache(self._static_v, past_v, 0, update_idx=0)

    def clear_static_kv(self) -> None:
        if self._static_k is not None:
            ttnn.deallocate(self._static_k)
            ttnn.deallocate(self._static_v)
        self._static_k = None
        self._static_v = None
        self._static_prefix_len = 0

    # ------------------------------------------------------------------ VLM prefix store
    def init_vlm_prefix_store(self, prefix_len: int) -> None:
        """Pre-allocate the VLM self-attention prefix K/V store (once per inference).

        Unlike the expert's cross-attention buffer (``_static_k``), this store is
        WRITE-ONLY during prefill: ``forward`` self-attends over the freshly
        computed prefix K/V and *also* mirrors them in-place into this
        pre-allocated buffer (``ttnn.fill_cache`` at ``update_idx=0``) so the
        action expert can later cross-attend to them. Because the buffer is
        allocated HERE -- outside any trace region -- and the per-call write is
        in-place, ``forward_vlm`` carries no allocation and becomes
        trace-capturable (the prior blocker was returning freshly-allocated
        persistent K/V per layer). Buffer is ``[1, num_kv_heads, prefix_len,
        head_dim]`` bfloat8_b (matches the bf8_b QKV-linear K/V output and the
        expert buffer dtype it is copied into). Persistent + same-shape reused
        across inferences so a captured trace keeps pointing at a valid buffer.
        """
        import torch as _torch

        if self._vlm_prefix_k is not None and self._vlm_prefix_k.shape[2] == prefix_len:
            return
        if self._vlm_prefix_k is not None:
            ttnn.deallocate(self._vlm_prefix_k)
            ttnn.deallocate(self._vlm_prefix_v)
        zeros = _torch.zeros(1, self.num_kv_heads, prefix_len, self.head_dim)
        # deep-plan_4 site (c): VLM-prefill self-attn prefix KV store. Flips to bf16 with
        # PI05_VLM_KV_BF16=1 (full change, together with site d) OR
        # PI05_VLM_QKV_OUT_BF16_PREFILLONLY=1 (the §4 cheap ladder-only probe -- _static_*
        # stays bf8_b there, the ladder never exercises the expert path). The VLM fill at
        # :432-433 then sees matching bf16 src (site a) -> dst dtypes. Weights stay bf8_b.
        _vlm_kv_dtype = ttnn.bfloat8_b
        if os.environ.get("PI05_VLM_KV_BF16") == "1" or os.environ.get("PI05_VLM_QKV_OUT_BF16_PREFILLONLY") == "1":
            _vlm_kv_dtype = ttnn.bfloat16
        self._vlm_prefix_k = ttnn.from_torch(
            zeros, dtype=_vlm_kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM
        )
        self._vlm_prefix_v = ttnn.from_torch(
            zeros, dtype=_vlm_kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM
        )

    def clear_vlm_prefix_store(self) -> None:
        if self._vlm_prefix_k is not None:
            ttnn.deallocate(self._vlm_prefix_k)
            ttnn.deallocate(self._vlm_prefix_v)
        self._vlm_prefix_k = None
        self._vlm_prefix_v = None

    def reset_trace_state(self) -> None:
        # STATEFUL: forward writes self._static_k/_v (expert) and self._vlm_prefix_k/_v (VLM
        # prefill) via ttnn.fill_cache. Both writes use a FIXED update_idx -- update_idx=0 for
        # the VLM store and update_idx=self._static_prefix_len for the expert suffix slot -- i.e.
        # an in-place OVERWRITE at a constant offset, NOT an advancing append (contrast
        # ttnn.update_cache, whose write position advances and so double-applies under the
        # double-run). Running forward twice during trace setup (warm-up + capture) overwrites the
        # SAME slot with the second run's value, leaving the buffer in exactly the state a single
        # capture run would -> the double-run is idempotent and there is NOTHING to revert. The
        # recorded fill_cache replays the current step's K/V into the same slot every replay.
        # Hence a justified no-op (NOT the bare TTNNModule sentinel -- this is a real, reasoned
        # reset). Were these ever changed to an advancing append, this MUST roll the write
        # position back to its pre-forward baseline instead.
        return None

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
        from tt_symbiote.models.pi05.modeling_pi05_common import get_sdpa_compute_kernel_config

        if len(hidden_states.shape) == 3:
            b, s, _ = hidden_states.shape
            hidden_states = ttnn.reshape(hidden_states, (b, 1, s, hidden_states.shape[-1]))
        else:
            b, _, s, _ = hidden_states.shape

        # QKV matmul: prefer the tracy-validated block-sharded program config
        # (build_matmul_pcfg: 1.24x VLM / 1.55x expert over the core_grid path);
        # fall back to the small-M 1-row width shard (seq<=96) / default 2D grid.
        qkv_cg = self._row_cg if s <= 96 else None
        _g = self.device.compute_with_storage_grid_size()
        # 2D-block matmul program config (build_matmul_pcfg: 1.24x VLM over the
        # core_grid path); fall back to the core_grid path when no clean grid divides.
        _qkv_pc = matmul_pcfg(
            s // 32, self.tt_wqkv.shape[-2] // 32, self.tt_wqkv.shape[-1] // 32, _g.x, _g.y, in0_block_w=8
        )
        # deep-plan_3 probe lever (env-gated, default OFF): pass an explicit HiFi4
        # compute_kernel_config to the VLM QKV matmul. The PRE/POST-SDPA-V split proved
        # the bf8_b QKV matmul output (V_pre, the SDPA INPUT) is the eroder, not the SDPA
        # accumulation; bumping the matmul accumulation fidelity targets that site directly.
        _qkv_ck = None
        if s > 96 and os.environ.get("PI05_VLM_QKV_HIFI4") == "1":
            _qkv_ck = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        # deep-plan_4 lever (env-gated, default OFF -> production/trace forward byte-identical):
        # de-quantize the QKV matmul OUTPUT to bf16 (weight self.tt_wqkv STAYS bf8_b). The
        # bf8_b matmul writeback is the per-layer V_pre eroder (the SDPA INPUT capped at ~0.92);
        # a bf16 output keeps V un-quantized FROM the matmul THROUGH the cache TO SDPA.
        #   PI05_VLM_QKV_OUT_BF16            -> bf16 on BOTH s>96 (VLM, site a) AND s<=96 (expert,
        #                                       site b). REQUIRED for the full E2E change: the expert
        #                                       suffix fill (:418-419) writes this output into the
        #                                       bf16 _static_* and fill_cache TT_FATALs on a dtype
        #                                       mismatch -> the s<=96 path MUST also emit bf16.
        #   PI05_VLM_QKV_OUT_BF16_PREFILLONLY-> bf16 on s>96 ONLY (the §4 cheap ladder probe; the
        #                                       expert path is never fed a bf16 buffer so no FATAL).
        # Optional matmul-path HiFi4+fp32-dest accumulation (SEPARATE from the SDPA fp32-dest ban,
        # allowed if bit-deterministic) gated by the same knob.
        _qkv_out_dtype = ttnn.bfloat8_b
        if os.environ.get("PI05_VLM_QKV_OUT_BF16") == "1":
            _qkv_out_dtype = ttnn.bfloat16
        elif s > 96 and os.environ.get("PI05_VLM_QKV_OUT_BF16_PREFILLONLY") == "1":
            _qkv_out_dtype = ttnn.bfloat16
        if _qkv_ck is None and _qkv_out_dtype == ttnn.bfloat16 and os.environ.get("PI05_VLM_QKV_OUT_HIFI4") == "1":
            _qkv_ck = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        if _qkv_pc is not None:
            qkv = ttnn.linear(
                hidden_states,
                self.tt_wqkv,
                dtype=_qkv_out_dtype,
                memory_config=_L1,
                program_config=_qkv_pc,
                compute_kernel_config=_qkv_ck,
            )
        else:
            qkv = ttnn.linear(
                hidden_states,
                self.tt_wqkv,
                dtype=_qkv_out_dtype,
                memory_config=_L1,
                core_grid=qkv_cg,
                compute_kernel_config=_qkv_ck,
            )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=_L1,
        )
        ttnn.deallocate(qkv)

        # RoPE (split-half rotation); cos/sin in meta format [1,1,seq,head_dim].
        # Pin output to L1 (else defaults to DRAM -> round-trip in the hot loop).
        q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=_L1)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=_L1)

        if self._static_k is not None:
            # Trace-safe cross-attention: write the new suffix K/V in-place into
            # the pre-allocated static buffer at the prefix offset (tile-aligned),
            # then SDPA over the full buffer. ttnn.fill_cache is an in-place device
            # write -- it allocates no new tensor, so the captured graph carries no
            # allocation (unlike ttnn.concat, which allocated a fresh KV tensor and
            # broke trace capture).
            ttnn.fill_cache(self._static_k, k, 0, update_idx=self._static_prefix_len)
            ttnn.fill_cache(self._static_v, v, 0, update_idx=self._static_prefix_len)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k = self._static_k
            v = self._static_v
            new_cache = None
        elif self._vlm_prefix_k is not None:
            # VLM prefill (trace-safe self-attention): mirror the freshly computed
            # prefix K/V into the pre-allocated store in-place (ttnn.fill_cache at
            # update_idx=0, allocates nothing -> capturable) for the action expert
            # to cross-attend to later, while self-attending over the same transient
            # K/V here. forward_vlm therefore returns no persistent cache and the
            # whole 18-layer prefill becomes trace-capturable.
            ttnn.fill_cache(self._vlm_prefix_k, k, 0, update_idx=0)
            ttnn.fill_cache(self._vlm_prefix_v, v, 0, update_idx=0)
            new_cache = None
        else:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = ttnn.concat([past_k, k], dim=2, memory_config=_L1)
                v = ttnn.concat([past_v, v], dim=2, memory_config=_L1)
            new_cache = (k, v) if use_cache else None

        # Expert SDPA (PI05_SDPA_CFG, default ON, seq<=96 only -- VLM is at parity
        # and large-seq L1 output risks OOM): pin the SDPA output to L1. The default
        # (no memory_config) lands it in DRAM, so the downstream nlp_concat_heads
        # reads from DRAM; L1 closes that (ConcatHeads 0.464->0.341/step = upstream
        # parity). The tuned SDPAProgramConfig is OPT-IN (PI05_SDPA_PCFG=1, default
        # OFF): on our 11x10 grid + expert shape (q=64,kv=576) it REGRESSED SDPA
        # 0.96->1.08/step (chunk bands tuned for upstream's grid; default SDPA grid
        # selection wins here).
        # Expert SDPA (main path, seq<=96): pin output to L1 (default DRAM would make
        # the downstream nlp_concat_heads read DRAM) + the swept SDPAProgramConfig on
        # grid (8,2). Grid (8,2) is the measured optimum for the small-q expert SDPA
        # (8-head/2-tile-q, kv~576): 0.687ms/step, BEATING upstream's 0.880 (the full
        # 110-core grid over-parallelizes -> regressed 1.078; sweep: (8,8)=0.809
        # (8,4)=0.699 (8,2)=0.687). VLM (s>96) keeps the default SDPA.
        _sdpa_kwargs = {}
        if s <= 96:
            # EXPERT (small-q): pin L1 output + the swept (8,2)-grid program config
            # (UNCHANGED -> in-trace expert path byte-identical -> captured==2 preserved).
            _sdpa_kwargs["memory_config"] = _L1  # L1 output -> concat_heads reads L1 not DRAM
            _spc = sdpa_program_config(q.shape[-2], k.shape[-2], min(_g.x, 8), min(_g.y, 2))
            if _spc is not None:
                _sdpa_kwargs["program_config"] = _spc
        elif os.environ.get("PI05_VLM_SDPA_PCFG", "0") == "1":
            # VLM PREFILL (s>96): OPT-IN restore of the reference long-prefix SDPA
            # program config (q_chunk=64, k_chunk=128 at 896 via sdpa_prefill_chunk_sizes;
            # exp_approx_mode=False) on the FULL device grid (reference ttnn_gemma.py:
            # 578-599 applies it UNCONDITIONALLY). deep-plan_3 MEASURED this (cfg CA):
            # the 18-layer PRE/POST-SDPA-V ladder proved the per-layer V eroder is the
            # bf8_b QKV matmul chain BEFORE attention (V_pre, the SDPA INPUT, is the low
            # quantity; the SDPA accumulation V_pre->V_post actually RAISES PCC), so the
            # program config moved V negligibly (meanVpre 0.9265->0.9268) and was
            # NET-NEGATIVE at E2E (own-KV E2E 0.6463 HiFi4-only -> 0.5833 HiFi4+pcfg).
            # => default OFF (kept config is HiFi4 SDPA fidelity only). PI05_VLM_SDPA_PCFG=1
            # re-enables it for A/B.
            _vlm_kc = os.environ.get("LADDER_VLM_KCHUNK")
            _vlm_qc = os.environ.get("LADDER_VLM_QCHUNK")
            _kw = {}
            if _vlm_kc is not None:
                _kw["k_chunk"] = int(_vlm_kc)
            if _vlm_qc is not None:
                _kw["q_chunk"] = int(_vlm_qc)
            _vlm_spc = sdpa_program_config(q.shape[-2], k.shape[-2], _g.x, _g.y, **_kw)
            if _vlm_spc is not None:
                _sdpa_kwargs["program_config"] = _vlm_spc
        # deep-plan_3 Phase-3 deterministic resort (env-gated, default OFF): cast the
        # SDPA V input to bf16 for the VLM prefill ONLY. The QKV matmul + cache stay
        # bf8_b (reference parity); only the SDPA online-softmax accumulation reads a
        # higher-mantissa V. Pure ttnn op; VLM-branch-only; pre-trace. PROBE lever.
        if s > 96 and os.environ.get("PI05_VLM_SDPA_V_BF16") == "1":
            v = ttnn.typecast(v, ttnn.bfloat16)
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
        # deep-plan_3 PRE/POST-SDPA-V split (env-gated, OFF by default -> production
        # forward byte-identical / trace-safe). When KV_LADDER_PRE_SDPA=1 the probe
        # reads this read-only snapshot of the post-online-softmax attention output
        # AFTER forward returns (never fed back into compute, never committed-active).
        if self._vlm_prefix_v is not None and os.environ.get("KV_LADDER_PRE_SDPA") == "1":
            self._ladder_post_sdpa_v = ttnn.to_torch(attn_out).float().clone()
        ttnn.deallocate(q)

        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=_L1)
        # o_proj matmul: 2D-block pcfg (same lever as qkv), else core_grid fallback.
        # Output is ALREADY bf16; weight tt_o STAYS bf8_b. deep-plan_4 widen rung 2
        # (env-gated, default OFF): PI05_VLM_OPROJ_HIFI4=1 bumps the o_proj matmul
        # ACCUMULATION fidelity (HiFi4 + fp32-dest -- MATMUL path, SEPARATE from the
        # SDPA fp32-dest ban, allowed if bit-deterministic). Both s>96 and s<=96.
        _o_ck = None
        if os.environ.get("PI05_VLM_OPROJ_HIFI4") == "1":
            _o_ck = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        _o_pc = matmul_pcfg(s // 32, self.tt_o.shape[-2] // 32, self.tt_o.shape[-1] // 32, _g.x, _g.y, in0_block_w=8)
        if _o_pc is not None:
            out = ttnn.linear(
                attn_out,
                self.tt_o,
                dtype=ttnn.bfloat16,
                memory_config=_L1,
                program_config=_o_pc,
                compute_kernel_config=_o_ck,
            )
        else:
            out = ttnn.linear(
                attn_out,
                self.tt_o,
                dtype=ttnn.bfloat16,
                memory_config=_L1,
                core_grid=qkv_cg,
                compute_kernel_config=_o_ck,
            )
        ttnn.deallocate(attn_out)
        if len(out.shape) == 4:
            out = ttnn.reshape(out, (b, s, out.shape[-1]))
        return out, new_cache


# ---------------------------------------------------------------------------
# Plain Gemma decoder block (VLM backbone)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05GemmaBlock(StatefulTTNNModule):
    """Pre-norm Gemma decoder block: norm -> attn -> +res -> norm -> mlp -> +res.

    Reference: ``reference/torch_gemma.py::GemmaBlock.forward``.

    STATEFUL because it contains the stateful ``TTNNPi05GemmaAttention`` (KV fill_cache) and is a
    trace unit during VLM prefill -- a Stateless trace unit may not own Stateful descendants. The
    block holds no own trace state; the framework resets the descendant attention via the trace
    tree-reset, so this is an own-state no-op.
    """

    @classmethod
    def from_torch(cls, block, config: GemmaConfig, mlp_weight_dtype=ttnn.bfloat8_b) -> "TTNNPi05GemmaBlock":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = block
        new._config = config
        new._eps = config.rms_norm_eps
        new._input_ln_w = block.input_layernorm_weight
        new._post_ln_w = block.post_attention_layernorm_weight
        new.attention = TTNNPi05GemmaAttention.from_torch(block.attention, config)
        new.mlp = TTNNPi05GemmaMLP.from_torch(block.mlp, config, weight_dtype=mlp_weight_dtype)
        return new

    def reset_trace_state(self) -> None:
        # No OWN trace state; the framework's trace tree-reset resets the stateful descendant
        # (self.attention). Stateful only because it owns that stateful descendant.
        return None

    def preprocess_weights_impl(self):
        self.tt_input_ln = _norm_weight_to_tt(self._input_ln_w)
        self.tt_post_ln = _norm_weight_to_tt(self._post_ln_w)
        self.attention.preprocess_weights()
        self.mlp.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.tt_input_ln = ttnn.to_device(self.tt_input_ln, self.device, memory_config=_DRAM)
        self.tt_post_ln = ttnn.to_device(self.tt_post_ln, self.device, memory_config=_DRAM)
        self.attention.move_weights_to_device()
        self.mlp.move_weights_to_device()

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
        _m, _h = hidden_states.shape[-2], hidden_states.shape[-1]
        normed = sharded_rms_norm(hidden_states, self.tt_input_ln, self._eps, _m, _h)
        attn_out, new_cache = self.attention(normed, cos, sin, attention_mask, past_key_value, use_cache)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=_L1)
        ttnn.deallocate(attn_out)

        normed = sharded_rms_norm(hidden_states, self.tt_post_ln, self._eps, _m, _h)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_out, memory_config=_L1)
        ttnn.deallocate(mlp_out)
        return hidden_states, new_cache


# ---------------------------------------------------------------------------
# AdaRMS Gemma block (action expert)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05AdaRMSGemmaBlock(StatefulTTNNModule):
    """Action-expert block with adaptive RMSNorm + gated residuals.

    adaRMS: ``normed = plain_rms(x); out = normed * (1 + scale) + shift`` where
    ``(scale, shift, gate) = chunk(linear(adarms_cond, mod_w, mod_b), 3)``.
    Residuals are gated: ``x = x + gate * sublayer(out)``.

    Reference: ``reference/torch_gemma.py::AdaRMSGemmaBlock.forward`` and
    ``tt/ttnn_gemma.py::AdaRMSGemmaBlockTTNN.forward``.

    VALIDATE ON HW: the chunked modulation + gated residual must be PCC-checked.
    """

    @classmethod
    def from_torch(cls, block, config: GemmaConfig) -> "TTNNPi05AdaRMSGemmaBlock":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = block
        new._config = config
        new._eps = config.rms_norm_eps
        new._width = config.width
        new._pre_attn_mod_w = block.pre_attn_mod_weight
        new._pre_attn_mod_b = getattr(block, "pre_attn_mod_bias", None)
        new._pre_ffw_mod_w = block.pre_ffw_mod_weight
        new._pre_ffw_mod_b = getattr(block, "pre_ffw_mod_bias", None)
        new.attention = TTNNPi05GemmaAttention.from_torch(block.attention, config)
        new.mlp = TTNNPi05GemmaMLP.from_torch(block.mlp, config)
        return new

    def reset_trace_state(self) -> None:
        # No OWN trace state; the framework's trace tree-reset resets the stateful descendant
        # (self.attention). Stateful only because it owns that stateful descendant.
        return None

    def preprocess_weights_impl(self):
        self.tt_pre_attn_mod_w = _linear_weight_to_tt(self._pre_attn_mod_w, dtype=ttnn.bfloat16)
        self.tt_pre_ffw_mod_w = _linear_weight_to_tt(self._pre_ffw_mod_w, dtype=ttnn.bfloat16)
        self.tt_pre_attn_mod_b = (
            ttnn.from_torch(
                self._pre_attn_mod_b.reshape(1, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            if self._pre_attn_mod_b is not None
            else None
        )
        self.tt_pre_ffw_mod_b = (
            ttnn.from_torch(
                self._pre_ffw_mod_b.reshape(1, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            if self._pre_ffw_mod_b is not None
            else None
        )
        self.attention.preprocess_weights()
        self.mlp.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.tt_pre_attn_mod_w = ttnn.to_device(self.tt_pre_attn_mod_w, self.device, memory_config=_DRAM)
        self.tt_pre_ffw_mod_w = ttnn.to_device(self.tt_pre_ffw_mod_w, self.device, memory_config=_DRAM)
        if self.tt_pre_attn_mod_b is not None:
            self.tt_pre_attn_mod_b = ttnn.to_device(self.tt_pre_attn_mod_b, self.device, memory_config=_DRAM)
        if self.tt_pre_ffw_mod_b is not None:
            self.tt_pre_ffw_mod_b = ttnn.to_device(self.tt_pre_ffw_mod_b, self.device, memory_config=_DRAM)
        self.attention.move_weights_to_device()
        self.mlp.move_weights_to_device()

    # Static-KV delegation to the attention sub-module (trace-safe cross-attn).
    def init_static_kv(self, prefix_len: int, suffix_len: int) -> None:
        self.attention.init_static_kv(prefix_len, suffix_len)

    def fill_static_prefix(self, past_k: ttnn.Tensor, past_v: ttnn.Tensor) -> None:
        self.attention.fill_static_prefix(past_k, past_v)

    def clear_static_kv(self) -> None:
        self.attention.clear_static_kv()

    def _compute_modulation(
        self, cond: ttnn.Tensor, mod_w: ttnn.Tensor, mod_b: Optional[ttnn.Tensor]
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """cond [B, width] -> (scale1, shift, gate) each [B, 1, width], scale1 = 1+scale.

        Reshape to [B,1,width] so they broadcast over the seq dim (torch reference
        does cond.unsqueeze(1)). Used both inline and for the precompute path.
        """
        mod = ttnn.linear(cond, mod_w, bias=mod_b, memory_config=_L1)
        w = self._width
        b = mod.shape[0]
        scale = ttnn.reshape(ttnn.slice(mod, [0, 0], [b, w]), (b, 1, w))
        shift = ttnn.reshape(ttnn.slice(mod, [0, w], [b, 2 * w]), (b, 1, w))
        gate = ttnn.reshape(ttnn.slice(mod, [0, 2 * w], [b, 3 * w]), (b, 1, w))
        ttnn.deallocate(mod)
        scale1 = ttnn.add(scale, 1.0, memory_config=_L1)
        ttnn.deallocate(scale)
        return scale1, shift, gate

    def precompute_mods(self, adarms_cond: ttnn.Tensor) -> Tuple[ttnn.Tensor, ...]:
        """Precompute the 6 modulation tensors for a fixed conditioning signal.

        Returns (scale1_a, shift_a, gate_a, scale1_f, shift_f, gate_f). Computed
        once per (step) outside the trace so the captured per-step graph drops
        the 2 mod-matmuls + slices/reshapes per layer (reference TIER A).
        """
        return (
            *self._compute_modulation(adarms_cond, self.tt_pre_attn_mod_w, self.tt_pre_attn_mod_b),
            *self._compute_modulation(adarms_cond, self.tt_pre_ffw_mod_w, self.tt_pre_ffw_mod_b),
        )

    def _apply_ada(self, x: ttnn.Tensor, scale1: ttnn.Tensor, shift: ttnn.Tensor, eps: float) -> ttnn.Tensor:
        # FUSED adaRMS (main path): fold the modulation into the norm kernel --
        # weight=(1+scale), bias=shift -> rms_norm(x)*scale1 + shift in ONE sharded op
        # (matches upstream _modulated_rms_norm). Drops the separate ttnn.multiply +
        # ttnn.add (2 BinaryNg per modulation x 2 modulations x 18 layers = 72 ops/
        # step). scale1/shift are [B,1,width] (precomputed, trace-safe). sharded_rms_norm
        # itself falls back to plain interleaved rms_norm if the reference builders
        # are absent, so this stays correct without the BS layer.
        return sharded_rms_norm(x, scale1, eps, x.shape[-2], x.shape[-1], bias=shift)

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
        # Modulation: precomputed (perf path) or computed inline from adarms_cond.
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
