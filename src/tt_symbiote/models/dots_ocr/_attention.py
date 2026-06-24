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
    """dots.ocr's forked paged KV cache.

    TS-5 contract -- page-table ownership and the per-device write index:

      * **Default page table** is the contiguous identity ``arange`` (one
        sequence laid out in contiguous blocks). dots.ocr is DATA-PARALLEL: the
        table is ``[batch_size, blocks_per_sequence]`` and DP-sharded so device
        ``d`` holds row ``d`` (its own sequence). The identity default is correct
        and required for the standalone / HF ``generate()`` path, which does not
        page.
      * **Serving (vLLM Tier-S2)** overrides the identity table via
        :meth:`set_vllm_page_table`, which installs the block-manager-assigned
        physical block ids (one row per DP stream) IN PLACE (trace-stable).
      * **``batch_idx=0`` on ``paged_fill_on_device`` is intentional, not a TODO.**
        Because the page table is DP-sharded to ONE row per device, ``batch_idx=0``
        is the only valid (and correct) per-device index -- each chip fills its own
        single sequence. (A non-DP shared cache that packed multiple sequences into
        one device's batch dim would need a varying ``batch_idx``; that is the
        shared-cache design, not this DP fork.)

    So the ``arange`` default and ``batch_idx=0`` are deliberately kept: removing
    them would corrupt KV for the DP layout. The Tier-S2 deliverable is the
    ``set_vllm_page_table`` hook (installed) plus per-row decode positions.
    """

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

        # Default = contiguous identity (standalone/HF path). vLLM serving
        # overrides this via set_vllm_page_table (see class docstring, TS-5).
        page_table = torch.arange(config.max_num_blocks, dtype=torch.int32)
        self.page_table = page_table.reshape(config.batch_size, config.blocks_per_sequence)

        self._tt_key_cache: list[Optional[ttnn.Tensor]] = [None] * num_layers
        self._tt_value_cache: list[Optional[ttnn.Tensor]] = [None] * num_layers
        self._tt_page_table: Optional[ttnn.Tensor] = None
        self._is_on_device = False
        # TS-5 observability: True once a vLLM block-manager table is installed
        # (serving), False while the contiguous identity default is in use.
        self._vllm_page_table_installed = False

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

    def _page_table_mesh_mapper(self):
        """Mesh mapper for the page table, matching ``to_device``.

        Under DP (``batch_size == num_devices`` on an (N,1)/(1,N) mesh) the
        page table is sharded along the batch dim so device ``d`` holds the
        block ids for sequence ``d``. Otherwise it is replicated.
        """
        bs = self.config.batch_size
        mapper = dp_batch_shard_tensor_mapper(self._device, bs)
        if mapper is None and self._device.get_num_devices() > 1:
            mapper = ttnn.ReplicateTensorToMesh(self._device)
        return mapper

    def set_vllm_page_table(self, page_table: torch.Tensor) -> "TTNNPagedAttentionKVCache":
        """Install an externally-managed page table (e.g. vLLM block ids).

        By default the cache uses a contiguous identity mapping
        (``arange(max_num_blocks)``) built in ``to_device``. Under
        tt-inference-server's vLLM backend the block manager assigns physical
        block ids per request (one row per DP stream), so the serving adapter
        calls this to point the paged ops (``paged_fill_on_device`` /
        ``paged_update_on_device`` / ``paged_sdpa_decode``) at the blocks vLLM
        allocated.

        This is **additive and HF-preserving**: HF ``generate()`` never calls
        it, so the default contiguous mapping (and every existing numeric
        result) is unchanged. It is the dots.ocr Tier-S2 seam; see
        docs/development/tt_inference_server_integration.md §9.

        DP layout: ``page_table`` is ``[batch, blocks_per_sequence]`` with
        ``batch == config.batch_size`` (one row per mesh device). The row for
        device ``d`` is sharded onto device ``d`` via the same
        ``dp_batch_shard_tensor_mapper`` ``to_device`` uses.

        Trace stability: the device page-table tensor is updated **in place**
        via ``copy_host_to_device_tensor`` from a HOST-ONLY source tensor (no
        device buffer is allocated -- uniform host-upload fix), so its
        buffer identity is preserved AND no device allocation occurs while a
        trace is live. A captured decode trace references that buffer, so
        swapping block tables between requests does NOT require re-capturing the
        trace, and the install is allocation-safe even inside a serving step.

        Args:
            page_table: int32 tensor ``[batch, blocks_per_sequence]`` mapping
                logical block index -> physical block id.
        """
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")
        if page_table.dim() != 2:
            raise ValueError(
                f"page_table must be 2D [batch, blocks_per_sequence], got shape {tuple(page_table.shape)}"
            )
        bs = self.config.batch_size
        bps = self.config.blocks_per_sequence
        if int(page_table.shape[0]) != bs:
            raise ValueError(
                f"page_table batch dim {int(page_table.shape[0])} != cache batch_size {bs} "
                "(one row per DP stream is required)"
            )
        if int(page_table.shape[1]) > bps:
            raise ValueError(
                f"page_table has {int(page_table.shape[1])} blocks/seq > cache capacity {bps}"
            )

        # Build a full-width [bs, bps] host table so the in-place device copy is
        # shape-stable (the preallocated _tt_page_table is [bs, bps]). Keep the
        # existing mapping for any trailing columns vLLM did not provide; those
        # columns are never read (the kernels index logical block = pos //
        # block_size, bounded by the sequence length).
        full = self.page_table.clone().to(torch.int32)
        n_blocks = int(page_table.shape[1])
        full[:, :n_blocks] = page_table.to(torch.int32)
        self.page_table = full.contiguous()

        mapper = self._page_table_mesh_mapper()
        # UNIFORM FIX: HOST-only tensor (no device= -> no
        # device buffer allocated) built with the SAME mapper as the device buffer, then
        # written IN PLACE via copy_host_to_device_tensor into the pre-allocated
        # _tt_page_table. No per-request device allocation -> cannot corrupt a live trace.
        host_pt = ttnn.from_torch(
            self.page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pt, self._tt_page_table)
        self._vllm_page_table_installed = True
        return self

    def paged_fill_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        batch_idx: int = 0,
        page_table: Optional[ttnn.Tensor] = None,
    ):
        """Fill the paged KV cache with a (chunk of) prefill K/V.

        ``page_table`` overrides the installed full page table. TS-7 chunked
        prefill passes a *chunk* page table (the slice of physical blocks for the
        chunk, see :meth:`build_chunk_page_table`) so the ``paged_fill_cache``
        kernel -- which always writes from virtual position 0 -- lands the chunk's
        K/V at its true absolute blocks. The default (``None``) uses the full
        page table and writes from offset 0 (single-shot prefill, unchanged).
        """
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")

        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        if page_table is None:
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

    def build_chunk_page_table(self, chunk_start: int, chunk_end: int) -> ttnn.Tensor:
        """Upload the page-table slice covering absolute positions [chunk_start, chunk_end).

        Returns a fresh device tensor ``[batch, n_blocks_chunk]`` mapping the
        chunk's virtual block 0..k-1 onto the physical blocks that hold positions
        ``[chunk_start, chunk_end)``. ``paged_fill_cache`` writes from virtual
        position 0, so handing it this slice lands the chunk K/V at its true
        blocks (TS-7). The caller owns the returned tensor (deallocate it).

        Both ``chunk_start`` and ``chunk_end`` should be block-aligned except for
        the final chunk, whose ``chunk_end`` may be the (unaligned) sequence end;
        we round the end block up so the partial trailing block is included.
        """
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")
        block = self.config.block_size
        start_block = chunk_start // block
        end_block = (chunk_end + block - 1) // block
        chunk_pt = self.page_table[:, start_block:end_block].contiguous().to(torch.int32)
        return ttnn.from_torch(
            chunk_pt,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self._device,
            mesh_mapper=self._page_table_mesh_mapper(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _chunk_prefill_program_config(seq_len: int, chunk_start_idx: int):
        """Per-chunk SDPA program config (mirrors tt_transformers).

        ``q_chunk_size`` must divide ``chunk_start_idx`` (tt-metal constraint);
        ``(x & -x)`` is the largest power of two dividing x.
        """
        base = 256 if seq_len >= 2048 else 64
        if not chunk_start_idx:
            q_chunk = base
        else:
            q_chunk = min(base, chunk_start_idx & -chunk_start_idx)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=q_chunk,
        )

    def chunked_sdpa_prefill(
        self,
        query: ttnn.Tensor,
        layer_idx: int,
        chunk_start_idx: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """Chunked prefill SDPA: a Q chunk attends over the cached prefix.

        Reads K/V from the paged cache via the full page table and uses the host
        ``chunk_start_idx`` to apply the correct causal mask for the chunk's
        absolute position. Requires the chunk's K/V to have already been written
        (via :meth:`paged_fill_on_device` with the chunk page table). Mirrors
        tt_transformers' chunked-prefill attention (eager; the host int variant
        of ``chunk_start_idx``).

        ``scale`` is intentionally NOT passed: the installed kernel binding marks
        ``scale`` as ``.noconvert()`` and rejects a Python float for it on
        multi-device tensors. The kernel default is ``1/sqrt(head_dim)``, which
        equals dots.ocr's ``self.scaling`` (head_dim**-0.5), so the numerics are
        identical. (The legacy int ``chunk_start_idx`` overload also does not
        accept ``compute_kernel_config``.)
        """
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")
        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        program_config = self._chunk_prefill_program_config(seq_len, chunk_start_idx)
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=query,
            input_tensor_k=k_cache,
            input_tensor_v=v_cache,
            page_table_tensor=self._tt_page_table,
            chunk_start_idx=int(chunk_start_idx),
            program_config=program_config,
        )

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

    def reset(self, clear_seq: bool = True) -> None:
        """Reset KV cache tracking for a new generation turn.

        Resets Python-side sequence tracking. Device buffer addresses are
        preserved so traces that reference them remain valid.  The stale
        values in the cache are harmless: prefill overwrites positions
        0..seq_len-1, and paged_sdpa_decode uses cur_pos_tensor to limit
        attention to valid positions.

        ``clear_seq=False`` (TS-8 prefix caching) keeps the per-layer sequence
        counters so a follow-up prefill computes only the uncached suffix on top
        of an already-populated prefix; the caller is then responsible for the
        counters via :meth:`seed_seq_length` / :meth:`update_seq_length`.
        """
        if clear_seq:
            self._seq_lengths = [0] * self.num_layers
            self._seen_tokens = 0

    def seed_seq_length(self, prefix_len: int) -> None:
        """Seed per-layer counters to ``prefix_len`` (TS-8 prefix caching).

        Used when the first ``prefix_len`` tokens are already resident in the
        paged cache (a vLLM prefix-cache hit), so the next prefill chunk starts
        at absolute position ``prefix_len``. Enables external seq tracking.
        """
        self._external_seq_tracking = True
        self._seq_lengths = [int(prefix_len)] * self.num_layers
        self._seen_tokens = int(prefix_len)


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
