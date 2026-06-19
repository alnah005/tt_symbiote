# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Rotary Position Embedding (RoPE) implementations for TTNN."""

from typing import Any, Tuple, Union

import torch
import ttnn


def _get_rotation_transformation_mat(dhead: int) -> torch.Tensor:
    """Generate rotation transformation matrix for RoPE: [x1, x2] -> [-x2, x1]."""
    trans_mat = torch.zeros(1, 1, dhead, dhead)
    trans_mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    trans_mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return trans_mat


def _compute_cos_sin_cache(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
    use_head_dim_for_freq: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin cache in Meta interleaved format, identity-padded for partial rotary.

    Args:
        head_dim: Full head dimension.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency.
        partial_rotary_factor: Fraction of head_dim to apply rotary to.
        use_head_dim_for_freq: If True, use head_dim as the denominator in the
            inv_freq formula (matching HuggingFace Gemma3/4 convention where
            inv_freq is computed over the full head_dim and only a prefix is
            used). If False (default), use rotary_dim as the denominator
            (matching the standard RoPE / HuggingFace Phi convention where
            inv_freq is computed over only the rotary dimensions).
            When partial_rotary_factor == 1.0, both choices are equivalent.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    freq_dim = head_dim if use_head_dim_for_freq else rotary_dim
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / freq_dim))
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # Interleave pairs for Meta format
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Identity-pad for partial rotary (cos=1, sin=0 means pass-through)
    if partial_rotary_factor < 1.0:
        pad_width = head_dim - rotary_dim
        cos_pad = torch.ones(cos.shape[0], cos.shape[1], cos.shape[2], pad_width)
        sin_pad = torch.zeros(sin.shape[0], sin.shape[1], sin.shape[2], pad_width)
        cos = torch.cat([cos, cos_pad], dim=-1)
        sin = torch.cat([sin, sin_pad], dim=-1)

    return cos, sin


def _compute_cos_sin_cache_half_half(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
    use_head_dim_for_freq: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin cache in HuggingFace half-half format (no interleaving).

    Same frequency computation as _compute_cos_sin_cache but without the
    pair-interleave step.  Used with ttnn.experimental.rotary_embedding
    (non-llama) which natively implements the half-half convention.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    freq_dim = head_dim if use_head_dim_for_freq else rotary_dim
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / freq_dim))
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    if partial_rotary_factor < 1.0:
        pad_width = head_dim - rotary_dim
        cos_pad = torch.ones(cos.shape[0], cos.shape[1], cos.shape[2], pad_width)
        sin_pad = torch.zeros(sin.shape[0], sin.shape[1], sin.shape[2], pad_width)
        cos = torch.cat([cos, cos_pad], dim=-1)
        sin = torch.cat([sin, sin_pad], dim=-1)

    return cos, sin


class BailingRotarySetup:
    """Pre-computed RoPE cos/sin and transformation matrices with replicated topology."""

    def __init__(
        self,
        device: Any,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        partial_rotary_factor: float = 1.0,
        datatype: ttnn.DataType = ttnn.bfloat16,
        use_head_dim_for_freq: bool = False,
        rope_convention: str = "interleaved",
    ) -> None:
        """Initialize with pre-computed cos/sin cache and transformation matrices.

        Args:
            use_head_dim_for_freq: See _compute_cos_sin_cache docstring.
            rope_convention: "interleaved" (Meta/Llama) or "half_half" (HF/Qwen2).
        """
        self.device = device
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.max_seq_len = max_seq_len
        self.partial_rotary_factor = partial_rotary_factor
        self.datatype = datatype
        self.rope_convention = rope_convention

        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1

        cache_fn = _compute_cos_sin_cache_half_half if rope_convention == "half_half" else _compute_cos_sin_cache
        cos_cache_torch, sin_cache_torch = cache_fn(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            use_head_dim_for_freq=use_head_dim_for_freq,
        )

        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None

        # TILE_LAYOUT cos/sin for prefill
        self.cos_cache = ttnn.from_torch(
            cos_cache_torch.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self.sin_cache = ttnn.from_torch(
            sin_cache_torch.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # ROW_MAJOR cos/sin for decode embedding lookup
        self.cos_cache_row_major = ttnn.from_torch(
            cos_cache_torch.squeeze(0).squeeze(0).to(torch.bfloat16),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self.sin_cache_row_major = ttnn.from_torch(
            sin_cache_torch.squeeze(0).squeeze(0).to(torch.bfloat16),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Pre-allocate typecast padding buffer for trace-safe decode path.
        pad_amount = 32 - 1
        pad_torch = torch.zeros(pad_amount, dtype=torch.int32)
        self._typecast_pad_buffer = ttnn.from_torch(
            pad_torch,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Transformation matrices only needed for interleaved (llama) convention
        self._trans_mat_decode_sharded_cache = {}
        if rope_convention == "interleaved":
            trans_mat_decode_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
            self.trans_mat_decode = ttnn.from_torch(
                trans_mat_decode_torch.to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=datatype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            self._trans_mat_decode_torch = trans_mat_decode_torch.to(torch.bfloat16)
            self.get_trans_mat_decode_sharded(1)

            trans_mat_prefill_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
            self.trans_mat_prefill = ttnn.from_torch(
                trans_mat_prefill_torch.to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=datatype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        else:
            self.trans_mat_decode = None
            self.trans_mat_prefill = None
            self._trans_mat_decode_torch = None

    def get_cos_sin_for_prefill(
        self,
        seq_len: int,
        start: int = 0,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get cos/sin for prefill, optionally at an absolute offset.

        Returns ``[1, 1, seq_len, head_dim]`` covering absolute positions
        ``[start, start + seq_len)``. ``start > 0`` is used by TS-7 chunked
        prefill so each chunk's tokens get RoPE for their true absolute
        positions (not 0..seq_len-1). The default ``start=0`` reproduces the
        original single-shot slice exactly.
        """
        end = start + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"Requested prefill positions [{start}, {end}) exceed max_seq_len "
                f"{self.max_seq_len}. Reinitialize BailingRotarySetup with larger "
                f"max_seq_len."
            )

        cos = self.cos_cache[:, :, start:end, :]
        sin = self.sin_cache[:, :, start:end, :]

        return cos, sin

    def get_cos_sin_for_decode(
        self,
        position_ids: Union[torch.Tensor, ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get cos/sin for decode via embedding lookup. Returns [1, batch, 1, head_dim].

        Trace-compatible: if position_ids is already a device tensor (e.g. from
        TracedRun pre-allocated buffers), uses it directly via device ops to avoid
        host→device transfers that break trace capture.
        """
        embedding_slice_back_size = None
        # Check if position_ids is already on device (trace-compatible path)
        if (
            isinstance(position_ids, ttnn.Tensor)
            and hasattr(position_ids, "storage_type")
            and position_ids.storage_type() != ttnn.StorageType.HOST
        ):
            pos_ttnn = position_ids
            # Convert int32→uint32 for embedding lookup.
            # ttnn.typecast requires padded_shape[-1] % 32 == 0.
            # During trace capture, we can't use ttnn.pad (writes host fill value).
            # Instead, concat with a pre-allocated zeros buffer to reach size 32.
            if pos_ttnn.dtype != ttnn.uint32:
                orig_size = pos_ttnn.shape[-1] if len(pos_ttnn.shape) > 0 else 1
                if orig_size % 32 != 0:
                    pad_amount = 32 - (orig_size % 32)
                    # Lazily create/cache zeros buffer for padding
                    if not hasattr(self, "_typecast_pad_buffer") or self._typecast_pad_buffer is None:
                        import torch as _torch

                        pad_torch = _torch.zeros(pad_amount, dtype=_torch.int32)
                        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None
                        self._typecast_pad_buffer = ttnn.from_torch(
                            pad_torch,
                            device=self.device,
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=mesh_mapper,
                        )
                    # Concat to reach size 32 (pure device op, trace-safe)
                    pos_ttnn = ttnn.concat([pos_ttnn, self._typecast_pad_buffer], dim=-1)
                pos_ttnn = ttnn.typecast(pos_ttnn, ttnn.uint32)
                # Keep ``pos_ttnn`` tile-aligned in last dim so that
                # ``ttnn.embedding(..., layout=TILE_LAYOUT)`` takes the
                # ``EmbeddingsFusedProgramFactory`` path (fused_tilized=True in
                # embedding.cpp:47-58) and emits TILE output directly. Slicing
                # back to ``orig_size`` here would set padded_shape[-1]=1 and
                # force a separate single-core 12 μs ``TilizeWithValPadding`` on
                # every cos and sin embedding. Defer the trim to after the
                # embedding lookups, when cos/sin are still in TILE layout and
                # the slice stays tile-aware (no untilize).
                if orig_size % 32 != 0:
                    embedding_slice_back_size = orig_size
            # Reshape to [1, batch] if needed for embedding
            if len(pos_ttnn.shape) == 1:
                pos_ttnn = ttnn.reshape(pos_ttnn, (1, pos_ttnn.shape[0]))
        else:
            # Host-tensor path: convert to torch and send to device
            if isinstance(position_ids, ttnn.Tensor):
                if self.is_mesh_device:
                    pos_torch = ttnn.to_torch(
                        position_ids,
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                    pos_torch = pos_torch[: position_ids.shape[0]]
                else:
                    pos_torch = ttnn.to_torch(position_ids)
            else:
                pos_torch = position_ids

            if len(pos_torch.shape) == 2:
                pos_torch = pos_torch.squeeze(0)

            batch_size = pos_torch.shape[0]
            pos_indices = pos_torch.reshape(1, batch_size).to(torch.int32)
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None

            pos_ttnn = ttnn.from_torch(
                pos_indices,
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )

        cos = ttnn.embedding(
            pos_ttnn,
            self.cos_cache_row_major,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            pos_ttnn,
            self.sin_cache_row_major,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Trim batch dim back to ``orig_size`` if we kept ``pos_ttnn`` padded
        # above to engage the fused-tilize embedding path. Slicing on a
        # TILE+INTERLEAVED tensor with tile-aligned begins skips the
        # untilize/tilize round-trip (slice.cpp rm_only path is not taken).
        if embedding_slice_back_size is not None:
            rotary_dim = int(cos.shape[-1])
            cos = ttnn.slice(cos, [0, 0, 0], [1, embedding_slice_back_size, rotary_dim])
            sin = ttnn.slice(sin, [0, 0, 0], [1, embedding_slice_back_size, rotary_dim])

        # Reshape [1, batch, rotary_dim] -> [1, batch, 1, rotary_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        return cos, sin

    def get_cos_sin_for_decode_sharded(
        self,
        position_ids: Union[torch.Tensor, ttnn.Tensor],
        batch_size: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get HEIGHT_SHARDED cos/sin for the decode-idiomatic rotary_embedding_hf.

        Reuses the interleaved (L1, TILE) cos/sin from ``get_cos_sin_for_decode``
        (shape [1, batch, 1, head_dim]) then reshards to HEIGHT across the batch
        cores. Mirrors tt_transformers ``HfRotarySetup.get_rot_mats``: the shard
        core grid MUST be the same ``num_cores_to_corerangeset(min(batch, grid),
        grid)`` that ``nlp_create_qkv_heads_decode`` uses for Q/K, so the sharded
        ``rotary_embedding_hf`` kernel reads matching data on each core.
        """
        cos, sin = self.get_cos_sin_for_decode(position_ids)  # [1, batch, 1, head_dim] L1 TILE
        if cos.dtype != ttnn.bfloat16:
            cos = ttnn.typecast(cos, ttnn.bfloat16)
        if sin.dtype != ttnn.bfloat16:
            sin = ttnn.typecast(sin, ttnn.bfloat16)
        core_grid = self.device.compute_with_storage_grid_size()
        num_cores = min(batch_size, core_grid.x * core_grid.y)
        grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, True)
        mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.interleaved_to_sharded(cos, mem), ttnn.interleaved_to_sharded(sin, mem)

    def get_trans_mat(self, is_decode: bool = False) -> ttnn.Tensor:
        """Get the RoPE transformation matrix (decode or prefill)."""
        return self.trans_mat_decode if is_decode else self.trans_mat_prefill

    def get_trans_mat_decode_sharded(self, batch_size: int) -> ttnn.Tensor:
        """Get HEIGHT_SHARDED trans_mat for decode (lazily cached per batch_size)."""
        if batch_size not in self._trans_mat_decode_sharded_cache:
            trans_mat_torch = self._trans_mat_decode_torch.repeat(1, 1, batch_size, 1)
            batch_grid = ttnn.num_cores_to_corerangeset(batch_size, self.device.compute_with_storage_grid_size(), True)
            mem_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                core_grid=batch_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None
            self._trans_mat_decode_sharded_cache[batch_size] = ttnn.from_torch(
                trans_mat_torch,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.datatype,
                memory_config=mem_config,
                mesh_mapper=mesh_mapper,
            )
        return self._trans_mat_decode_sharded_cache[batch_size]
