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
from typing import Dict, Optional, Tuple

import torch
import ttnn

from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices
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


# ---------------------------------------------------------------------------
# Weight upload helpers (host-side; allowed to use torch).
# ---------------------------------------------------------------------------
def _linear_weight_to_tt(
    w: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat8_b
) -> ttnn.Tensor:
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
class TTNNPi05GemmaMLP(TTNNModule):
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
        # gate/up: large-M VLM (288x2048x16384) wins on the FULL 2D grid (tracy
        # device: 303us default -> 135us full); small-M expert keeps default.
        gu_cg = self._full_cg if seq > 96 else None
        gate = ttnn.linear(x, self.tt_gate, memory_config=_L1, core_grid=gu_cg)
        # fast_and_approximate_mode: matches the reference's tanh-approx gelu and is
        # ~2.9x faster on the 16384-wide VLM intermediate (96->33us; op-sweep, PCC>=0.999).
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=_L1)
        up = ttnn.linear(x, self.tt_up, memory_config=_L1, core_grid=gu_cg)
        hidden = ttnn.multiply(gate, up, memory_config=_L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # down: large-M VLM -> full grid; small-M expert -> 1-row width shard.
        down_cg = self._row_cg if seq <= 96 else self._full_cg
        out = ttnn.linear(hidden, self.tt_down, memory_config=_L1, core_grid=down_cg)
        ttnn.deallocate(hidden)
        return out


# ---------------------------------------------------------------------------
# Multi-Query Attention (8 Q heads, 1 KV head, head_dim 256)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05GemmaAttention(TTNNModule):
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
            raise ValueError(
                f"static KV requires a tile-aligned prefix_len; got {prefix_len} (not a multiple of 32)"
            )
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
        self._static_k = ttnn.from_torch(zeros, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM)
        self._static_v = ttnn.from_torch(zeros, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM)
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
        self._vlm_prefix_k = ttnn.from_torch(zeros, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM)
        self._vlm_prefix_v = ttnn.from_torch(zeros, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_DRAM)

    def clear_vlm_prefix_store(self) -> None:
        if self._vlm_prefix_k is not None:
            ttnn.deallocate(self._vlm_prefix_k)
            ttnn.deallocate(self._vlm_prefix_v)
        self._vlm_prefix_k = None
        self._vlm_prefix_v = None

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

        # QKV: small-M (action expert, seq<=96) wins on a 1-row width shard
        # (tracy device: expert qkv 22.9->20.9us); VLM keeps the default 2D grid.
        qkv_cg = self._row_cg if s <= 96 else None
        qkv = ttnn.linear(hidden_states, self.tt_wqkv, dtype=ttnn.bfloat8_b, memory_config=_L1, core_grid=qkv_cg)
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

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=get_sdpa_compute_kernel_config(),
        )
        ttnn.deallocate(q)

        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=_L1)
        # o_proj: small-M expert uses the 1-row shard. NOTE: a standalone device
        # bench said default beats row here (32.5 vs 58us), but the IN-MODEL traced
        # step disagrees (row 5.01ms vs default 5.24ms) -- the real concat_heads
        # input layout differs from the random-tensor bench, so we trust the
        # in-model traced measurement (the production metric).
        out = ttnn.linear(attn_out, self.tt_o, dtype=ttnn.bfloat16, memory_config=_L1, core_grid=qkv_cg)
        ttnn.deallocate(attn_out)
        if len(out.shape) == 4:
            out = ttnn.reshape(out, (b, s, out.shape[-1]))
        return out, new_cache


# ---------------------------------------------------------------------------
# Plain Gemma decoder block (VLM backbone)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05GemmaBlock(TTNNModule):
    """Pre-norm Gemma decoder block: norm -> attn -> +res -> norm -> mlp -> +res.

    Reference: ``reference/torch_gemma.py::GemmaBlock.forward``.
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
        normed = _rms_norm(hidden_states, self.tt_input_ln, self._eps)
        attn_out, new_cache = self.attention(
            normed, cos, sin, attention_mask, past_key_value, use_cache
        )
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=_L1)
        ttnn.deallocate(attn_out)

        normed = _rms_norm(hidden_states, self.tt_post_ln, self._eps)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_out, memory_config=_L1)
        ttnn.deallocate(mlp_out)
        return hidden_states, new_cache


# ---------------------------------------------------------------------------
# AdaRMS Gemma block (action expert)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05AdaRMSGemmaBlock(TTNNModule):
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
        # Explicit ones-weight for the adaRMS plain RMSNorm. Weightless
        # ``ttnn.rms_norm`` allocates+writes a default weight shard to device on
        # first call, which is an illegal host->device WRITE inside a TTNN trace
        # capture region. Pre-uploading a constant ones-weight makes the op
        # trace-safe (rms_norm with a ones gamma == weightless rms_norm).
        self.tt_ones = ttnn.from_torch(
            torch.ones(1, self._width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
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
        self.tt_ones = ttnn.to_device(self.tt_ones, self.device, memory_config=_DRAM)
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
        # Pass an explicit ones-weight (== weightless) so the op performs no
        # host->device write inside a TTNN trace capture region.
        normed = ttnn.rms_norm(x, weight=self.tt_ones, epsilon=eps, memory_config=_L1)  # plain
        out = ttnn.multiply(normed, scale1, memory_config=_L1)
        out = ttnn.add(out, shift, memory_config=_L1)
        ttnn.deallocate(normed)
        return out

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
        attn_out, new_cache = self.attention(
            normed, cos, sin, attention_mask, past_key_value, use_cache
        )
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
