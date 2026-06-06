# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN PaliGemma dual-expert backbone for pi0.5.

Combines the SigLIP vision tower, multimodal projector, Gemma-2B VLM stack
(18 plain Gemma blocks) and the Gemma-300M action expert (18 adaRMS blocks),
plus the VLM embedding table and final norms. K/V computed by the VLM during
prefix prefill are reused (cross-attention) by the expert during denoising.

Ported from the tt-metal reference ``tt/ttnn_paligemma.py`` /
``reference/torch_paligemma.py::Pi0_5PaliGemmaBackbone`` (branch ``tt/pi0.5_bh``).
The golden reference is a plain object whose attributes are:
``vlm_embed_tokens`` (tensor), ``vlm_norm`` (tensor), ``vlm_blocks`` (list of
``GemmaBlock``), ``expert_blocks`` (list of ``AdaRMSGemmaBlock``),
``expert_norm_mod_weight`` / ``expert_norm_mod_bias`` (final adaRMS dense),
``vision_tower`` (``SigLIPVisionTower``), ``mm_projector`` (``MultiModalProjector``).

VALIDATION STATUS: ``embed_image`` / ``embed_language_tokens`` wiring and the
``forward_vlm`` / ``forward_expert`` block loops are implemented (functionality
first; sequential-position RoPE baseline). The position-id-aware (cumsum) RoPE
override used by the openpi upstream path is a later refinement (see
``tt/ttnn_pi0_5_model.py::_build_upstream_attn_artifacts``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import ttnn

from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices

from .configuration_pi05 import PaliGemmaConfig
from .modeling_pi05_common import precompute_freqs_cis_meta
from .modeling_pi05_gemma import (
    TTNNPi05AdaRMSGemmaBlock,
    TTNNPi05GemmaBlock,
    _linear_weight_to_tt,
    _norm_weight_to_tt,
    _rms_norm,
)
from .modeling_pi05_siglip import TTNNPi05MultiModalProjector, TTNNPi05SigLIPVisionTower

TT_METAL_COMMIT = "b2af0cd67b4e92dafeb2d0254e1c5b43c3ec5a25"

__all__ = ["TTNNPi05PaliGemmaBackbone"]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


class TTNNPi05PaliGemmaBackbone(TTNNModule):
    """PaliGemma dual-expert backbone (VLM + adaRMS action expert)."""

    @classmethod
    def from_torch(cls, backbone, config: PaliGemmaConfig) -> "TTNNPi05PaliGemmaBackbone":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = backbone
        new._config = config
        new._vlm_config = config.vlm_config
        new._expert_config = config.expert_config

        new.vision_tower = TTNNPi05SigLIPVisionTower.from_torch(backbone.vision_tower, config.siglip_config)
        new.mm_projector = TTNNPi05MultiModalProjector.from_torch(backbone.mm_projector)
        # VLM MLP weights -> bfloat8_b (the current validated fast path). bfloat4_b
        # was op-sweep-only ~1ms/chunk faster on the 288x2048x16384 prefill MLP but
        # NOT E2E-PCC-validated (4-bit VLM weights propagate through the prefix KV
        # into all 10 denoise steps), so it is not part of the main path.
        _vlm_wd = ttnn.bfloat8_b
        new.vlm_blocks = [
            TTNNPi05GemmaBlock.from_torch(b, config.vlm_config, mlp_weight_dtype=_vlm_wd) for b in backbone.vlm_blocks
        ]
        new.expert_blocks = [
            TTNNPi05AdaRMSGemmaBlock.from_torch(b, config.expert_config) for b in backbone.expert_blocks
        ]

        # Raw tensors plucked from the reference (host); converted in preprocess.
        new._embed_tokens_w = backbone.vlm_embed_tokens
        new._vlm_norm_w = backbone.vlm_norm
        new._expert_norm_mod_w = backbone.expert_norm_mod_weight
        new._expert_norm_mod_b = getattr(backbone, "expert_norm_mod_bias", None)
        new._eps_vlm = config.vlm_config.rms_norm_eps
        new._eps_expert = config.expert_config.rms_norm_eps
        new._expert_width = config.expert_config.width
        return new

    # ------------------------------------------------------------------ weights
    def preprocess_weights_impl(self):
        # Embedding table stays ROW_MAJOR for ttnn.embedding lookup.
        self.tt_embed_tokens = ttnn.from_torch(
            self._embed_tokens_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.tt_vlm_norm = _norm_weight_to_tt(self._vlm_norm_w)
        # Final expert norm is adaRMS (no +1 fold; modulation handles it at runtime).
        self.tt_expert_norm_mod_w = _linear_weight_to_tt(self._expert_norm_mod_w, dtype=ttnn.bfloat16)
        self.tt_expert_norm_mod_b = (
            ttnn.from_torch(
                self._expert_norm_mod_b.reshape(1, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            if self._expert_norm_mod_b is not None
            else None
        )
        # Explicit ones-gamma for the final expert RMSNorm. That norm is weightless
        # (its gamma is folded into the adaRMS scale1 at runtime), but a weightless
        # ``ttnn.rms_norm`` writes a default-gamma shard to device on first call --
        # an illegal host->device WRITE inside a TTNN trace-capture region. Pre-uploading
        # a constant ones-gamma makes the op trace-safe (rms_norm with gamma=1 ==
        # weightless rms_norm, numerically identical).
        self.tt_expert_norm_ones = ttnn.from_torch(
            torch.ones(1, self._expert_width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.vision_tower.preprocess_weights()
        self.mm_projector.preprocess_weights()
        for b in self.vlm_blocks:
            b.preprocess_weights()
        for b in self.expert_blocks:
            b.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.tt_embed_tokens = ttnn.to_device(self.tt_embed_tokens, self.device, memory_config=_DRAM)
        self.tt_vlm_norm = ttnn.to_device(self.tt_vlm_norm, self.device, memory_config=_DRAM)
        self.tt_expert_norm_mod_w = ttnn.to_device(self.tt_expert_norm_mod_w, self.device, memory_config=_DRAM)
        if self.tt_expert_norm_mod_b is not None:
            self.tt_expert_norm_mod_b = ttnn.to_device(self.tt_expert_norm_mod_b, self.device, memory_config=_DRAM)
        self.tt_expert_norm_ones = ttnn.to_device(self.tt_expert_norm_ones, self.device, memory_config=_DRAM)
        self.vision_tower.move_weights_to_device()
        self.mm_projector.move_weights_to_device()
        for b in self.vlm_blocks:
            b.move_weights_to_device()
        for b in self.expert_blocks:
            b.move_weights_to_device()
        # RoPE meta tables (need the device). VLM and expert share head_dim 256.
        self.tt_cos_vlm, self.tt_sin_vlm = precompute_freqs_cis_meta(
            self._vlm_config.head_dim, self._config.max_seq_len, self.device, self._vlm_config.rope_base
        )
        self.tt_cos_expert, self.tt_sin_expert = precompute_freqs_cis_meta(
            self._expert_config.head_dim, self._config.max_seq_len, self.device, self._expert_config.rope_base
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _slice_rope(cos: ttnn.Tensor, sin: ttnn.Tensor, seq_len: int, offset: int = 0):
        """Slice meta cos/sin [1,1,max_seq,hd] to absolute positions [offset:offset+seq_len].

        ``offset`` places the suffix tokens at their absolute positions *after*
        the prefix in the shared attention (suffix RoPE positions are
        prefix_len + [0..seq_len-1]); offset=0 is the sequential/prefix path.
        """
        hd = cos.shape[-1]
        cos_s = ttnn.slice(cos, [0, 0, offset, 0], [1, 1, offset + seq_len, hd])
        sin_s = ttnn.slice(sin, [0, 0, offset, 0], [1, 1, offset + seq_len, hd])
        return cos_s, sin_s

    # ------------------------------------------------------------------ embed
    @run_on_devices(DeviceArch.P150)
    def embed_image(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """Images -> SigLIP -> projector -> (B, num_patches, vlm_width)."""
        vision_features = self.vision_tower(pixel_values)
        return self.mm_projector(vision_features)

    @run_on_devices(DeviceArch.P150)
    def embed_language_tokens(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Token ids -> Gemma embedding lookup -> (B, seq, vlm_width)."""
        emb = ttnn.embedding(token_ids, self.tt_embed_tokens, layout=ttnn.TILE_LAYOUT)
        return emb

    # ------------------------------------------------------------------ VLM
    @run_on_devices(DeviceArch.P150)
    def forward_vlm(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        seq_len = hidden_states.shape[-2] if len(hidden_states.shape) == 4 else hidden_states.shape[1]
        cos, sin = self._slice_rope(self.tt_cos_vlm, self.tt_sin_vlm, seq_len)
        # When the VLM prefix store is initialized (init_vlm_static_kv, outside any
        # trace region), each VLM attention mirrors its prefix K/V into the
        # pre-allocated store in-place -- so forward_vlm allocates no persistent
        # cache and the whole 18-layer prefill is trace-capturable. The K/V live in
        # the stores (read later via init_expert_static_kv_from_vlm), so we collect
        # no per-layer cache here. Otherwise fall back to returning the explicit
        # (k, v) list (eager-only path).
        store_active = bool(self.vlm_blocks) and self.vlm_blocks[0].attention._vlm_prefix_k is not None
        collect = use_cache and not store_active
        new_cache = [] if collect else None
        for block in self.vlm_blocks:
            hidden_states, new_kv = block(
                hidden_states, cos, sin, attention_mask, None, collect
            )
            if collect:
                new_cache.append(new_kv)
        hidden_states = _rms_norm(hidden_states, self.tt_vlm_norm, self._eps_vlm)
        return hidden_states, new_cache

    # ------------------------------------------------------------------ VLM static KV
    def init_vlm_static_kv(self, prefix_len: int) -> None:
        """Allocate each VLM layer's prefix K/V store (once, outside any trace).

        After this is set, ``forward_vlm`` mirrors each layer's prefix K/V into its
        store in-place and returns no persistent cache -> the prefill traces
        end-to-end. ``init_expert_static_kv_from_vlm`` then copies these stores into
        the expert cross-attention buffers for the denoise loop.
        """
        for blk in self.vlm_blocks:
            blk.attention.init_vlm_prefix_store(prefix_len)

    def clear_vlm_static_kv(self) -> None:
        for blk in self.vlm_blocks:
            blk.attention.clear_vlm_prefix_store()

    def init_expert_static_kv_from_vlm(self, prefix_len: int, suffix_len: int) -> None:
        """Allocate + prefill the expert cross-attention KV buffers from the VLM
        prefix stores (no intermediate persistent cache list).

        Equivalent to ``init_expert_static_kv`` but sources the prefix K/V from the
        VLM prefix stores (populated by a store-active ``forward_vlm``) instead of a
        returned cache list. Called ONCE per inference, between prefill and the
        denoise loop, outside any trace region.
        """
        for i, blk in enumerate(self.expert_blocks):
            vlm_attn = self.vlm_blocks[i].attention
            blk.init_static_kv(prefix_len, suffix_len)
            blk.fill_static_prefix(vlm_attn._vlm_prefix_k, vlm_attn._vlm_prefix_v)

    # ------------------------------------------------------------------ expert
    def init_expert_static_kv(
        self,
        prefix_kv_cache: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
        prefix_len: int,
        suffix_len: int,
    ) -> None:
        """Allocate + prefill each expert layer's static cross-attention KV buffer.

        Called ONCE per inference (after VLM prefill, before the denoise loop),
        OUTSIDE any trace region. Each expert attention gets a static
        ``[1,1,prefix_len+suffix_len,head_dim]`` buffer whose prefix region is
        filled with the (already-RoPE'd) VLM prefix K/V via ``ttnn.fill_cache``.
        Inside the denoise loop, each expert-attention forward writes only the new
        suffix K/V in-place (no concat -> trace-safe).
        """
        for i, blk in enumerate(self.expert_blocks):
            past_k, past_v = prefix_kv_cache[i]
            blk.init_static_kv(prefix_len, suffix_len)
            blk.fill_static_prefix(past_k, past_v)

    def clear_expert_static_kv(self) -> None:
        for blk in self.expert_blocks:
            blk.clear_static_kv()

    def precompute_final_mod(self, cond: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Precompute (scale1, shift) for the final expert adaRMS norm (no gate)."""
        mod = ttnn.linear(cond, self.tt_expert_norm_mod_w, bias=self.tt_expert_norm_mod_b, memory_config=_L1)
        w = self._expert_width
        b = mod.shape[0]
        scale = ttnn.reshape(ttnn.slice(mod, [0, 0], [b, w]), (b, 1, w))
        shift = ttnn.reshape(ttnn.slice(mod, [0, w], [b, 2 * w]), (b, 1, w))
        ttnn.deallocate(mod)
        scale1 = ttnn.add(scale, 1.0, memory_config=_L1)
        ttnn.deallocate(scale)
        return scale1, shift

    def _ada_rms_norm_no_gate(self, x: ttnn.Tensor, cond=None, precomputed=None) -> ttnn.Tensor:
        """Final expert adaRMS (discards gate): normed*(1+scale)+shift."""
        scale1, shift = precomputed if precomputed is not None else self.precompute_final_mod(cond)
        # Explicit ones-gamma (trace-safe; weightless rms_norm writes a default gamma on
        # first call -- illegal during trace capture). Numerically identical to weightless.
        normed = ttnn.rms_norm(x, weight=self.tt_expert_norm_ones, epsilon=self._eps_expert, memory_config=_L1)
        out = ttnn.multiply(normed, scale1, memory_config=_L1)
        out = ttnn.add(out, shift, memory_config=_L1)
        ttnn.deallocate(normed)
        if precomputed is None:
            ttnn.deallocate(scale1)
            ttnn.deallocate(shift)
        return out

    @staticmethod
    def _to_dram(tensors):
        """Move precomputed mod tensors L1 -> DRAM (they are load-once, read-per-step;
        keeping ~1080 of them in L1 clashes with matmul circular buffers)."""
        out = []
        for t in tensors:
            d = ttnn.to_memory_config(t, _DRAM)
            ttnn.deallocate(t)
            out.append(d)
        return tuple(out)

    def precompute_step_mods(self, adarms_cond: ttnn.Tensor):
        """Precompute all per-layer + final modulations for a fixed conditioning,
        parked in DRAM.

        Returns (block_mods, final_mod) where block_mods is a list of 18 six-tuples
        (one per expert layer) and final_mod is (scale1, shift). Called once per
        denoise step outside the trace so the captured graph carries no mod-matmuls.
        """
        block_mods = [self._to_dram(blk.precompute_mods(adarms_cond)) for blk in self.expert_blocks]
        final_mod = self._to_dram(self.precompute_final_mod(adarms_cond))
        return block_mods, final_mod

    @run_on_devices(DeviceArch.P150)
    def forward_expert(
        self,
        hidden_states: ttnn.Tensor,
        adarms_cond: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_offset: int = 0,
        precomputed_block_mods=None,
        precomputed_final_mod=None,
    ) -> ttnn.Tensor:
        """Action expert: 18 adaRMS blocks (cross-attending cached prefix KV) + final adaRMS norm.

        ``position_offset`` places the suffix tokens at their absolute RoPE
        positions after the prefix (= prefix_len) for the shared-attention KV
        concat. ``attention_mask`` (additive, on DRAM) masks phantom suffix
        positions when action_horizon is padded to a tile multiple.
        """
        seq_len = hidden_states.shape[-2] if len(hidden_states.shape) == 4 else hidden_states.shape[1]
        cos, sin = self._slice_rope(self.tt_cos_expert, self.tt_sin_expert, seq_len, offset=position_offset)
        # When static KV buffers are initialized (init_expert_static_kv), each
        # expert attention owns its prefix+suffix KV buffer and writes the suffix
        # in-place -- so we pass past_kv=None and let the static path run (no concat,
        # trace-safe). Otherwise fall back to the explicit past_key_values concat.
        static_kv = bool(self.expert_blocks) and self.expert_blocks[0].attention._static_k is not None
        for i, block in enumerate(self.expert_blocks):
            past_kv = None if static_kv else (past_key_values[i] if past_key_values else None)
            block_mod = precomputed_block_mods[i] if precomputed_block_mods is not None else None
            hidden_states, _ = block(
                hidden_states, cos, sin, adarms_cond, attention_mask, past_kv, False, precomputed_mod=block_mod
            )
        return self._ada_rms_norm_no_gate(
            hidden_states, cond=adarms_cond, precomputed=precomputed_final_mod
        )
