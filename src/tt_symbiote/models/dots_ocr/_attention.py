# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention mechanism implementations for TTNN."""

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
except ImportError:
    print("Could not import sdpa_attention_forward from transformers.integrations.sdpa_attention. ")

import ttnn

from tt_symbiote.core.module import SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS, StatelessTTNNModule, run_on_devices
from tt_symbiote.core.tensor import TorchTTNNTensor

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object

try:
    from transformers.cache_utils import CacheLayerMixin
except ImportError:
    CacheLayerMixin = None


def dp_batch_shard_tensor_mapper(device, batch_size: int):
    if not hasattr(device, "get_num_devices") or device.get_num_devices() <= 1:
        return None
    if batch_size != device.get_num_devices():
        return None
    shp = list(device.shape)
    if shp[0] == batch_size and shp[1] == 1:
        return ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(shp), dims=(0, None))
    if shp[1] == batch_size and shp[0] == 1:
        return ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(shp), dims=(None, 0))
    return None


@dataclass
class PagedAttentionConfig:
    block_size: int = 64
    max_num_blocks: int = 2048
    batch_size: int = 1

    @property
    def max_seq_length(self) -> int:
        return self.max_num_blocks * self.block_size

    @property
    def blocks_per_sequence(self) -> int:
        return self.max_num_blocks // self.batch_size


class TTNNPagedAttentionKVCache(Cache):
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        config: PagedAttentionConfig,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
        tt_cache_dtype=None,
    ):
        try:
            # HF's Cache class has a non-trivial __init__, so we need to call super().__init__() with the expected arguments
            super().__init__(layers=[])
        except:
            super().__init__()

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.config = config
        self.dtype = dtype
        # On-device cache dtype: BF16 by default for backward compat. Models
        # that can tolerate the BFP8 quantization noise on K/V cache (e.g.
        # dots.ocr in TP=2) override this to ``ttnn.bfloat8_b`` to halve the
        # K/V DRAM read bandwidth in ``paged_sdpa_decode``. The kernel still
        # accepts BF16 *input* tensors (paged_update_cache_device_operation
        # only allows BF16/FP32 input) and converts internally to the cache
        # dtype on write.
        self._tt_cache_dtype = tt_cache_dtype if tt_cache_dtype is not None else ttnn.bfloat16
        self._device = device
        self._seq_lengths: list[int] = [0] * num_layers
        self._seen_tokens = 0
        self._external_seq_tracking = False

        page_table = torch.arange(config.max_num_blocks, dtype=torch.int32)
        self.page_table = page_table.reshape(config.batch_size, config.blocks_per_sequence)

        self._tt_key_cache: list[Optional[ttnn.Tensor]] = [None] * num_layers
        self._tt_value_cache: list[Optional[ttnn.Tensor]] = [None] * num_layers
        self._tt_page_table: Optional[ttnn.Tensor] = None
        self._is_on_device = False

    def to_device(self, device) -> "TTNNPagedAttentionKVCache":
        if self._is_on_device and self._device == device:
            return self

        self._device = device
        bs = self.config.batch_size
        page_table_mapper = dp_batch_shard_tensor_mapper(device, bs)
        if page_table_mapper is None and device.get_num_devices() > 1:
            page_table_mapper = ttnn.ReplicateTensorToMesh(device)

        cache_shape = (
            self.config.max_num_blocks,
            self.num_kv_heads,
            self.config.block_size,
            self.head_dim,
        )

        for layer_idx in range(self.num_layers):
            self._tt_key_cache[layer_idx] = ttnn.zeros(
                cache_shape,
                dtype=self._tt_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._tt_value_cache[layer_idx] = ttnn.zeros(
                cache_shape,
                dtype=self._tt_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self._tt_page_table = ttnn.from_torch(
            self.page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=page_table_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._is_on_device = True
        return self

    def paged_fill_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        batch_idx: int = 0,
    ):
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")

        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        page_table = self._tt_page_table

        max_len = self.config.blocks_per_sequence * self.config.block_size
        seq_len = key_states.shape[2]
        if seq_len > max_len:
            key_states = key_states[:, :, :max_len, :]
            value_states = value_states[:, :, :max_len, :]
            seq_len = max_len

        ttnn.experimental.paged_fill_cache(k_cache, key_states, page_table, batch_idx=batch_idx)
        ttnn.experimental.paged_fill_cache(v_cache, value_states, page_table, batch_idx=batch_idx)

        # When external seq tracking is enabled (via update_seq_length()),
        # the caller is responsible for updating counters outside the trace
        # boundary. Otherwise, update counters here for backward compatibility
        # with callers that do not use update_seq_length().
        if not self._external_seq_tracking:
            self._seq_lengths[layer_idx] += seq_len
            if layer_idx == 0:
                self._seen_tokens += seq_len

    def paged_update_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
    ):
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")

        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        page_table = self._tt_page_table

        ttnn.experimental.paged_update_cache(
            k_cache,
            key_states,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            value_states,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )

        # When external seq tracking is enabled (via update_seq_length()),
        # the caller is responsible for updating counters outside the trace
        # boundary. Otherwise, update counters here for backward compatibility.
        if not self._external_seq_tracking:
            seq_len = key_states.shape[0]
            self._seq_lengths[layer_idx] += seq_len
            if layer_idx == 0:
                self._seen_tokens += seq_len

    def update_seq_length(self, layer_idx: int, seq_len: int = 1) -> None:
        """Increment Python-side sequence counters for a layer.

        This MUST be called outside the trace boundary (i.e. from the model's
        layer loop) so that the counters advance correctly during trace replay,
        warmup, and capture phases alike.

        Calling this method enables external sequence tracking, which disables
        the automatic counter increments inside paged_fill_on_device() and
        paged_update_on_device() to prevent double-counting.
        """
        if not self._external_seq_tracking:
            self._external_seq_tracking = True
        self._seq_lengths[layer_idx] += seq_len
        if layer_idx == 0:
            self._seen_tokens += seq_len

    def paged_sdpa_decode(
        self,
        query: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
        scale: float = 1.0,
        program_config=None,
        compute_kernel_config=None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """Paged SDPA decode with optional sliding window.

        Args:
            sliding_window: If set, restricts attention to the most recent
                sliding_window KV positions. Currently plumbed through but
                NOT enforced -- the paged_scaled_dot_product_attention_decode
                kernel in tt-metal 0.62.2 does not support sliding_window_size
                natively, and dynamic attn_mask construction is incompatible
                with trace capture. See Phase 2 plan for enforcement strategy.
        """
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")

        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        page_table = self._tt_page_table

        # TODO(Phase 2): Enforce sliding window via circular-buffer-as-pages
        # strategy. For now, sliding_window is accepted but not enforced.
        # This is correct for sequences shorter than the window size.

        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            k_cache,
            v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update CPU-side KV cache for PyTorch layers (matches DynamicCache.update contract)."""
        if isinstance(key_states, TorchTTNNTensor):
            key_states = key_states.to_torch
        if isinstance(value_states, TorchTTNNTensor):
            value_states = value_states.to_torch
        if isinstance(key_states, ttnn.Tensor):
            key_states = ttnn.to_torch(key_states)
        if isinstance(value_states, ttnn.Tensor):
            value_states = ttnn.to_torch(value_states)

        seq_len = key_states.shape[2]
        self._seq_lengths[layer_idx] = seq_len
        if layer_idx == 0:
            self._seen_tokens = seq_len

        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_lengths[layer_idx]

    def get_max_cache_shape(self) -> Optional[int]:
        return self.config.max_seq_length

    def reset(self) -> None:
        """Reset KV cache tracking for a new generation turn.

        Resets Python-side sequence tracking. Device buffer addresses are
        preserved so traces that reference them remain valid.  The stale
        values in the cache are harmless: prefill overwrites positions
        0..seq_len-1, and paged_sdpa_decode uses cur_pos_tensor to limit
        attention to valid positions.
        """
        self._seq_lengths = [0] * self.num_layers
        self._seen_tokens = 0


class TorchSDPAAttention(torch.nn.Module):
    def forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        transpose_output: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        attn_output = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )[0]
        if not transpose_output:  # revert the transpose in sdpa_attention_forward
            attn_output = attn_output.transpose(1, 2)
        return attn_output


class TTNNSDPAAttention(StatelessTTNNModule):
    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = TorchSDPAAttention()
        self.program_config = None
        self.compute_kernel_config = None
        self.memory_config = None
        self._sdpa_available = True

    def _matmul_attention(self, query, key, value, is_causal, scaling, attention_mask, transpose_output):
        import math

        scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
        key_t = ttnn.permute(key, (0, 1, 3, 2))
        scores = ttnn.matmul(query, key_t)
        scores = ttnn.multiply(scores, scale)

        if is_causal:
            seq_len = query.shape[2]
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(torch.bfloat16)
            causal_mask = ttnn.from_torch(
                causal_mask.unsqueeze(0).unsqueeze(0),
                layout=ttnn.TILE_LAYOUT,
                device=query.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            scores = ttnn.add(scores, causal_mask)
        elif attention_mask is not None:
            scores = ttnn.add(scores, attention_mask)

        scores = ttnn.softmax(scores, dim=-1)
        attn_output = ttnn.matmul(scores, value)

        if transpose_output:
            attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        return attn_output

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        transpose_output: bool = True,
        **kwargs,
    ) -> ttnn.Tensor:
        # if query.layout != ttnn.TILE_LAYOUT:
        #     query = ttnn.to_layout(query, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # if key.layout != ttnn.TILE_LAYOUT:
        #     key = ttnn.to_layout(key, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # if value.layout != ttnn.TILE_LAYOUT:
        #     value = ttnn.to_layout(value, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        # key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        # value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        assert len(query.shape) == 4, "Query tensor must be 4D"
        assert dropout == 0.0, "TTNNSDPAAttention does not support dropout"
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
        if attention_mask is not None:
            if attention_mask.layout != ttnn.TILE_LAYOUT:
                attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if attention_mask.dtype != query.dtype:
                attention_mask = ttnn.typecast(attention_mask, query.dtype)

        if self._sdpa_available:
            try:
                attn_output = ttnn.transformer.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=is_causal,
                    scale=scaling,
                    program_config=self.program_config,
                    attn_mask=attention_mask,
                    compute_kernel_config=self.compute_kernel_config,
                    memory_config=self.memory_config,
                )
                if transpose_output:
                    attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
                return attn_output
            except RuntimeError as e:
                print(
                    f"TTNNSDPAAttention: ttnn SDPA failed, falling back to matmul attention. "
                    f"Q={query.shape} K={key.shape} V={value.shape} is_causal={is_causal} "
                    f"Error: {e}"
                )
                self._sdpa_available = False

        return self._matmul_attention(query, key, value, is_causal, scaling, attention_mask, transpose_output)
