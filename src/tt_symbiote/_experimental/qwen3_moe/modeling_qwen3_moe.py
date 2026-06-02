# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# This file was assembled during the Phase 2 mechanical migration
# (see scripts/merge_model_files.py). It concatenates the following
# original sources in order, deduplicating top-of-file imports:
#   - models/experimental/tt_symbiote/modules/qwen_attention.py
#   - models/experimental/tt_symbiote/modules/qwen_moe.py

import os
from typing import Optional

import torch
import torch.nn.functional as F
import ttnn

from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices, tree_map
from tt_symbiote.core.run_config import DistributedTensorConfig
from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.modules.ttnn_attention import PagedAttentionConfig, TTNNPagedAttentionKVCache, TTNNSDPAAttention
from tt_symbiote.modules.ttnn_linear import TTNNLinear, TTNNLinearIReplicatedWColSharded
from tt_symbiote.modules.ttnn_moe import (
    SPARSITY_BLOCK_SIZE,
    Glm4MoeRouteTokenToExperts,
    TTNNExperts,
    TTNNGlm4MoeMLP,
    TTNNGlm4MoeTopkRouter,
    TTNNMoE,
    TTNNMoERouterDecode,
    _make_sparse_matmul_program_config,
    even_int_div,
)
from tt_symbiote.modules.ttnn_rope import TTNNRotaryPositionEmbedding

# === content from models/experimental/tt_symbiote/modules/qwen_attention.py ===
"""Qwen3.5-35B-A3B Attention implementations for TTNN.

This module provides TTNN-accelerated attention mechanisms specific to Qwen3.5-35B-A3B:
- TTNNQwenPagedAttentionKVCache: Paged KV cache with layer_indices mapping for hybrid attention
- TTNNQwen3FullAttention: Full GQA attention with Q gating and Q/K normalization
- TTNNQwen3LinearAttention: Linear attention (DeltaNet) with TTNN-accelerated projections
"""


class CallableBool:
    """A bool-like object that is also callable, for backward compatibility.

    This enables has_previous_state to work as both a property and a method:
      - cache.has_previous_state        -> truthy/falsy (property access)
      - cache.has_previous_state()      -> bool (method call, no args)
      - cache.has_previous_state(idx)   -> bool (method call with layer_idx)
    """

    def __init__(self, value: bool, cache: "TTNNQwenPagedAttentionKVCache"):
        self._value = value
        self._cache = cache

    def __bool__(self) -> bool:
        return self._value

    def __call__(self, layer_idx: int | None = None) -> bool:
        if layer_idx is not None:
            return self._cache.conv_states.get(layer_idx) is not None
        return self._value

    def __repr__(self) -> str:
        return repr(self._value)

    def __eq__(self, other) -> bool:
        if isinstance(other, bool):
            return self._value == other
        return NotImplemented


class TTNNQwenPagedAttentionKVCache(TTNNPagedAttentionKVCache):
    """Paged attention KV cache with layer indices mapping for Qwen3.5 hybrid attention.

    Qwen3.5 uses hybrid attention with pattern [linear, linear, linear, full] x 10.
    Only full attention layers (10 total) use KV cache, so we need to map
    absolute layer_idx (3, 7, 11, ..., 39) to cache indices (0, 1, 2, ..., 9).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        config: PagedAttentionConfig,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
        layer_indices: Optional[list[int]] = None,
    ):
        """Initialize Qwen paged attention KV cache.

        Args:
            num_layers: Number of cache slots (10 for Qwen3.5 full attention layers)
            num_kv_heads: Number of KV heads (2 for Qwen3.5)
            head_dim: Head dimension (256 for Qwen3.5)
            config: Paged attention configuration
            device: TTNN device or mesh device
            dtype: Data type for cache tensors
            layer_indices: Optional list mapping cache indices to absolute layer indices.
                           If None, uses identity mapping.
                           For Qwen3.5: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        """
        super().__init__(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            config=config,
            device=device,
            dtype=dtype,
        )
        # Map absolute layer_idx to cache index
        # For Qwen3.5: layer_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        # This means layer_idx=3 maps to cache_idx=0, layer_idx=7 maps to cache_idx=1, etc.
        if layer_indices is not None:
            self._layer_to_cache_idx = {layer_idx: cache_idx for cache_idx, layer_idx in enumerate(layer_indices)}
        else:
            self._layer_to_cache_idx = None

        # Linear attention compatibility properties
        # Native Qwen linear attention layers (Qwen3_5MoeGatedDeltaNet) check these attributes
        # These are needed when TT_QWEN_CPU_LINEAR_ATTN=1 falls back to native PyTorch linear attention
        self._has_previous_state = False  # Track if cache has been used
        self.conv_states = {}  # DeltaNet conv states per layer
        self.recurrent_states = {}  # DeltaNet recurrent states per layer

    @property
    def has_previous_state(self) -> "CallableBool":
        """Check if cache has been used previously.

        Returns a CallableBool that supports both property access and method call:
          - cache.has_previous_state        -> truthy/falsy (existing tt-symbiote code)
          - cache.has_previous_state()      -> bool (HuggingFace model code)
          - cache.has_previous_state(idx)   -> bool (HuggingFace linear attention)
        """
        if self.conv_states:
            value = any(v is not None for v in self.conv_states.values())
        else:
            value = False
        return CallableBool(value, self)

    @has_previous_state.setter
    def has_previous_state(self, value: bool):
        """Set previous state flag."""
        self._has_previous_state = value

    def _get_cache_idx(self, layer_idx: int) -> int:
        """Convert absolute layer_idx to cache index.

        Args:
            layer_idx: Absolute layer index in the model (e.g., 3, 7, 11, ...)

        Returns:
            Cache index (0, 1, 2, ...)
        """
        if self._layer_to_cache_idx is not None:
            return self._layer_to_cache_idx.get(layer_idx, layer_idx)
        return layer_idx

    def paged_fill_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        batch_idx: int = 0,
    ):
        """Fill KV cache for a layer, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx)
        super().paged_fill_on_device(key_states, value_states, cache_idx, batch_idx)

    def paged_update_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
    ):
        """Update KV cache for a layer, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx)
        super().paged_update_on_device(key_states, value_states, cache_idx, current_pos)

    def update_seq_length(self, layer_idx: int, seq_len: int = 1) -> None:
        """Increment Python-side sequence counters, mapping layer_idx to cache_idx.

        Silently skips layers not in the cache mapping (e.g. linear attention
        layers in Qwen3.5 hybrid attention).
        """
        if self._layer_to_cache_idx is not None and layer_idx not in self._layer_to_cache_idx:
            return  # This layer doesn't use paged KV cache
        cache_idx = self._get_cache_idx(layer_idx)
        super().update_seq_length(cache_idx, seq_len)

    def paged_sdpa_decode(
        self,
        query: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
        scale: float = 1.0,
        program_config=None,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """Decode using paged KV cache, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx)
        return super().paged_sdpa_decode(query, cache_idx, current_pos, scale, program_config, compute_kernel_config)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length for a layer, mapping layer_idx to cache_idx."""
        cache_idx = self._get_cache_idx(layer_idx) if layer_idx != 0 else 0
        return super().get_seq_length(cache_idx)


class TTNNQwen3FullAttention(TTNNModule):
    """TTNN-accelerated Full Attention for Qwen3.5-35B-A3B.

    Implements Grouped Query Attention (GQA) with:
    - 16 attention heads
    - 2 KV heads (8:1 ratio, each KV head serves 8 Q heads)
    - head_dim = 256
    - RoPE position embeddings
    - Q gating: q_proj outputs 2x dimension, split into Q and gate
    - Q/K normalization: RMSNorm on head_dim

    Supports both standard DynamicCache and TTNNQwenPagedAttentionKVCache
    for paged attention with on-device KV storage.
    """

    def __init__(self):
        super().__init__()
        self.num_attention_heads = None
        self.num_key_value_heads = None
        self.num_key_value_groups = None
        self.head_dim = None
        self.hidden_size = None
        self.scaling = None
        self.is_causal = True
        self.layer_idx = None

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rope = None
        self.sdpa = None
        self.core_grid = None

        # Q gating support - q_proj outputs 2x dimension, split into Q and gate
        # gate is applied after attention: output *= sigmoid(gate)
        self.has_q_gate = False

        # Pre-allocated decode cur_pos buffer for trace safety (initialized in move_weights_to_device_impl)
        self._decode_cur_pos = None

        # Q/K normalization (RMSNorm on head_dim)
        self.q_norm_weight = None  # Host tensor
        self.k_norm_weight = None  # Host tensor
        self.tt_q_norm_weight = None  # Device tensor
        self.tt_k_norm_weight = None  # Device tensor
        self.rms_norm_eps = 1e-6

    @classmethod
    def from_torch(cls, torch_attn, distributed: bool = True):
        """Create TTNNQwen3FullAttention from PyTorch Qwen3_5MoeAttention.

        Args:
            torch_attn: PyTorch Qwen3_5MoeAttention layer
            distributed: Whether to use distributed linear layers (default True for T3K)

        Returns:
            TTNNQwen3FullAttention instance
        """
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        # Extract configuration from torch layer
        config = torch_attn.config
        new_attn.num_attention_heads = config.num_attention_heads  # 16
        new_attn.num_key_value_heads = config.num_key_value_heads  # 2
        new_attn.num_key_value_groups = new_attn.num_attention_heads // new_attn.num_key_value_heads  # 8
        new_attn.head_dim = config.head_dim  # 256
        new_attn.hidden_size = config.hidden_size  # 2048
        new_attn.scaling = new_attn.head_dim**-0.5
        new_attn.layer_idx = torch_attn.layer_idx

        # Check for Q gating: q_proj outputs 2x dimension for Q + gate
        # PyTorch: query_states, gate = torch.chunk(self.q_proj(hidden_states).view(..., head_dim * 2), 2, dim=-1)
        q_proj_out_features = torch_attn.q_proj.out_features
        expected_q_dim = new_attn.num_attention_heads * new_attn.head_dim
        new_attn.has_q_gate = q_proj_out_features == expected_q_dim * 2

        # Extract Q/K normalization weights if present
        # Qwen3 uses RMSNorm on head_dim with weight initialized to zeros
        # Forward: output = rms_norm(x) * (1.0 + weight)
        if hasattr(torch_attn, "q_norm") and torch_attn.q_norm is not None:
            new_attn.q_norm_weight = torch_attn.q_norm.weight.detach().clone()
            new_attn.rms_norm_eps = getattr(torch_attn.q_norm, "eps", 1e-6)
        if hasattr(torch_attn, "k_norm") and torch_attn.k_norm is not None:
            new_attn.k_norm_weight = torch_attn.k_norm.weight.detach().clone()

        # Choose linear layer classes based on distributed mode
        # Input projections take replicated input and produce col-sharded output (like linear attention)
        LinearClsIn = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        # Output projection also takes replicated input (after attention gather) and produces col-sharded output
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear

        # Create TTNN linear projections
        new_attn.q_proj = LinearClsIn.from_torch(torch_attn.q_proj)
        new_attn.k_proj = LinearClsIn.from_torch(torch_attn.k_proj)
        new_attn.v_proj = LinearClsIn.from_torch(torch_attn.v_proj)
        new_attn.o_proj = LinearClsOut.from_torch(torch_attn.o_proj)

        # RoPE for position embeddings
        # Qwen3.5 uses partial rotary (rotary_dim=64, head_dim=256, factor=0.25),
        # so we always use non-distributed RoPE which handles partial rotary correctly
        new_attn.rope = TTNNRotaryPositionEmbedding()

        # SDPA for attention computation
        new_attn.sdpa = TTNNSDPAAttention()

        # Default core grid (will be updated in move_weights_to_device_impl)
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def preprocess_weights_impl(self):
        """Preprocess Q/K normalization weights for TTNN."""
        super().preprocess_weights_impl()

        # Prepare Q/K norm weights for TTNN
        # Qwen3 RMSNorm applies: output = rms_norm(x) * (1.0 + weight)
        # TTNN rms_norm applies: output = rms_norm(x) * weight
        # So we need to add 1.0 to the weight before converting
        if self.q_norm_weight is not None:
            # (1.0 + weight) for Qwen3-style RMSNorm
            q_norm_adjusted = (1.0 + self.q_norm_weight.float()).to(self.q_norm_weight.dtype)
            # Expand to [1, head_dim] for broadcasting in rms_norm
            self.tt_q_norm_weight_host = ttnn.from_torch(
                q_norm_adjusted.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        if self.k_norm_weight is not None:
            k_norm_adjusted = (1.0 + self.k_norm_weight.float()).to(self.k_norm_weight.dtype)
            self.tt_k_norm_weight_host = ttnn.from_torch(
                k_norm_adjusted.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        """Initialize SDPA config and move norm weights to device."""
        super().move_weights_to_device_impl()

        # Query grid dynamically from device (REQUIRED per CLAUDE.md)
        grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        # Move Q/K norm weights to device with proper mesh replication
        # These weights must be replicated to ALL mesh devices, not just device 0
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if hasattr(self, "tt_q_norm_weight_host") and self.tt_q_norm_weight_host is not None:
            q_norm_torch = ttnn.to_torch(self.tt_q_norm_weight_host)
            self.tt_q_norm_weight = ttnn.from_torch(
                q_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if hasattr(self, "tt_k_norm_weight_host") and self.tt_k_norm_weight_host is not None:
            k_norm_torch = ttnn.to_torch(self.tt_k_norm_weight_host)
            self.tt_k_norm_weight = ttnn.from_torch(
                k_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=128,  # Reduced for head_dim=256 (matches DeepSeek V3)
                k_chunk_size=128,  # Reduced for head_dim=256 (matches DeepSeek V3)
                exp_approx_mode=False,
            )
            # Match DeepSeek V3 settings for head_dim=256 compatibility:
            # fp32_dest_acc_en=False increases dst_size from 4 to 8
            # packer_l1_acc=False reduces L1 pressure
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        # Pre-allocate decode cur_pos buffer for trace safety (matching Gemma4 pattern).
        # During trace replay, ttnn.from_torch() allocations get frozen, so we
        # pre-allocate once and use ttnn.copy() to update the value each decode step.
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        self._decode_cur_pos = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def set_output_tensors_config_impl(self, output_tensors):
        """Set output tensor config for col-sharded output.

        The o_proj output is col-sharded (each device has [batch, seq, hidden_size/8]).
        We need to use ConcatMeshToTensor on dim=-1 to concatenate the shards.
        """

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    # Use ConcatMeshToTensor on dim=-1 only (not batch dim)
                    # This concatenates the col-sharded output from all devices
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        """Compute logical shape by multiplying last dim by num_devices."""
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    config = DistributedTensorConfig(
                        mesh_mapper=mesh_mapper,
                        mesh_composer=mesh_composer,
                        logical_shape_fn=logical_shape_for_col_sharded,
                    )
                    e.set_distributed_tensor_config(config)
            return e

        # Use the default config from parent if not distributed
        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)

        return tree_map(set_col_sharded_config, output_tensors)

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor
        t = tensor
        gathered = ttnn.all_gather(
            t,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        # Synchronize to ensure all-gather completes before returning
        ttnn.synchronize_device(self.device)
        return gathered

    def _is_tensor_replicated(self, tensor) -> bool:
        """Check if tensor is replicated across devices (vs sharded).

        Returns True if tensor uses ReplicateTensorToMesh or has full hidden_size,
        False if tensor is sharded (each device has hidden_size/num_devices).
        """
        if tensor is None:
            return True

        # Check for distributed config first
        if hasattr(tensor, "ttnn_distributed_tensor_config"):
            config = tensor.ttnn_distributed_tensor_config
            if config is not None:
                mapper = config.mesh_mapper
                if mapper is not None:
                    mapper_type = type(mapper).__name__
                    if "Replicate" in mapper_type:
                        return True
                    if "Shard" in mapper_type:
                        return False
                return False

        # Check physical shape - sharded tensors have hidden_size/num_devices on last dim
        physical_shape = None
        if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
            physical_shape = tuple(int(i) for i in tensor.ttnn_tensor.shape)
        elif isinstance(tensor, ttnn.Tensor):
            physical_shape = tuple(int(i) for i in tensor.shape)
        elif hasattr(tensor, "shape") and tensor.shape is not None:
            physical_shape = tuple(tensor.shape)

        if physical_shape is not None and len(physical_shape) >= 1 and self.device is not None:
            num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
            if num_devices > 1:
                last_dim = physical_shape[-1]
                if last_dim == self.hidden_size:
                    return True  # Full hidden_size = replicated
                elif last_dim == self.hidden_size / num_devices:
                    return False  # Partial hidden_size = sharded

        return False  # Default to sharded (safer)

    def _repeat_kv(self, hidden_states: ttnn.Tensor, n_rep: int) -> ttnn.Tensor:
        """Repeat KV heads to match Q heads for GQA.

        For Qwen3.5: 2 KV heads -> 16 Q heads, so n_rep=8
        [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]

        Uses repeat_interleave for correct GQA head ordering:
        - Correct: [K0,K0,...,K0, K1,K1,...,K1] - Q heads 0-7 attend to K0, Q heads 8-15 attend to K1
        - Wrong (ttnn.repeat tiles): [K0,K1,K0,K1,...] - Q head 1 would wrongly attend to K1
        """
        if n_rep == 1:
            return hidden_states
        # Use repeat_interleave for correct GQA head ordering
        return ttnn.repeat_interleave(hidden_states, n_rep, dim=1)

    def _to_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Convert a multi-device tensor to an explicitly replicated tensor.

        After all-gather the data is identical on every device but the mesh
        topology metadata differs from ReplicateTensorToMesh. Paged-attention
        kernels require the replicated topology, so we round-trip through the
        host for decode tokens (tiny tensors, negligible overhead).
        """
        if self.device.get_num_devices() <= 1:
            return tensor
        t = tensor
        if isinstance(t, TorchTTNNTensor):
            t = t.to_ttnn if hasattr(t, "to_ttnn") else t
        orig_shape = list(t.shape)
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)
        t_torch = t_torch[: orig_shape[0]]
        return ttnn.from_torch(
            t_torch,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=t.dtype,
            layout=t.layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _project_qkv(self, hidden_states, batch_size, seq_length, position_embeddings):
        """Project hidden states to Q, K, V, apply Q/K norm, and apply RoPE.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            batch_size: Batch dimension
            seq_length: Sequence length
            position_embeddings: Tuple of (cos, sin) for RoPE

        Returns:
            Tuple of (query_states, key_states, value_states, gate, cos, sin)
            gate is None if has_q_gate is False
        """
        # ALL-GATHER INPUT IF SHARDED: TTNNLinearIReplicatedWColSharded expects
        # replicated input [batch, seq, hidden_size]. If input is col-sharded
        # (from previous MoE layer), all-gather it first.
        if self._is_distributed and not self._is_tensor_replicated(hidden_states):
            hidden_states = self._maybe_all_gather(hidden_states)

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Project to Q, K, V
        # For Qwen3: q_proj outputs [batch, seq, num_heads * head_dim * 2] when has_q_gate=True
        q_proj_output = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]

        # All-gather for distributed mode
        q_proj_output = self._maybe_all_gather(q_proj_output)
        key_states = self._maybe_all_gather(key_states)
        value_states = self._maybe_all_gather(value_states)

        # Handle Q gating: split q_proj output into query_states and gate
        # PyTorch: query_states, gate = torch.chunk(q_proj(x).view(..., head_dim * 2), 2, dim=-1)
        gate = None
        if self.has_q_gate:
            # q_proj_output: [batch, seq, num_heads * head_dim * 2]
            # Reshape to [batch, seq, num_heads, head_dim * 2] then split
            q_proj_output = ttnn.reshape(
                q_proj_output, (batch_size, seq_length, self.num_attention_heads, self.head_dim * 2)
            )
            # Split along last dim: first half is Q, second half is gate
            # TTNN doesn't have a direct chunk, use slicing
            query_states = q_proj_output[:, :, :, : self.head_dim]  # [batch, seq, num_heads, head_dim]
            gate = q_proj_output[:, :, :, self.head_dim :]  # [batch, seq, num_heads, head_dim]
            # Reshape gate to [batch, seq, num_heads * head_dim] for later sigmoid multiply
            gate = ttnn.reshape(gate, (batch_size, seq_length, self.num_attention_heads * self.head_dim))
        else:
            # No gating - standard Q projection
            query_states = ttnn.reshape(
                q_proj_output, (batch_size, seq_length, self.num_attention_heads, self.head_dim)
            )

        # Apply Q/K normalization (RMSNorm on head_dim) before RoPE
        # PyTorch: query_states = self.q_norm(query_states.view(hidden_shape))
        # query_states shape here: [batch, seq, num_heads, head_dim]
        if self.tt_q_norm_weight is not None:
            # Flatten for rms_norm: [batch * seq * num_heads, head_dim]
            orig_q_shape = query_states.shape
            query_states = ttnn.reshape(
                query_states, (batch_size * seq_length * self.num_attention_heads, self.head_dim)
            )
            query_states = ttnn.rms_norm(query_states, weight=self.tt_q_norm_weight, epsilon=self.rms_norm_eps)
            query_states = ttnn.reshape(query_states, orig_q_shape)

        # Reshape K to [batch, seq, num_kv_heads, head_dim] for normalization
        key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))

        if self.tt_k_norm_weight is not None:
            orig_k_shape = key_states.shape
            key_states = ttnn.reshape(key_states, (batch_size * seq_length * self.num_key_value_heads, self.head_dim))
            key_states = ttnn.rms_norm(key_states, weight=self.tt_k_norm_weight, epsilon=self.rms_norm_eps)
            key_states = ttnn.reshape(key_states, orig_k_shape)

        # Permute to [batch, num_heads, seq_len, head_dim] for attention
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))

        value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_key_value_heads, self.head_dim))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        # Apply RoPE to Q and K
        cos, sin = position_embeddings
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, 1)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, 1)
        if self._is_distributed:
            cos = self._maybe_all_gather(cos)
            sin = self._maybe_all_gather(sin)

        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        # Expand KV to match Q heads (GQA)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states, gate, cos, sin

    def _forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Prefill path for attention computation."""
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        query_states, key_states, value_states, gate, cos, sin = self._project_qkv(
            hidden_states, batch_size, seq_length, position_embeddings
        )

        use_paged = isinstance(past_key_values, TTNNQwenPagedAttentionKVCache) or isinstance(
            past_key_values, TTNNPagedAttentionKVCache
        )

        if past_key_values is not None:
            # For paged cache with layer_indices mapping, pass the actual layer_idx.
            # The KV cache will internally map it to the correct cache slot.
            # For non-paged cache, also use the actual layer_idx.
            cache_layer_idx = self.layer_idx

            if use_paged:
                # For paged cache, key_states and value_states need to be in
                # [batch, num_kv_heads, seq, head_dim] format before fill
                # But they're already expanded for GQA, so we need the original
                # Recompute unexpanded KV for cache storage
                kv_key = key_states[:, :: self.num_key_value_groups, :, :]  # Sample every n-th head
                kv_value = value_states[:, :: self.num_key_value_groups, :, :]

                past_key_values.paged_fill_on_device(
                    kv_key,
                    kv_value,
                    layer_idx=cache_layer_idx,
                    batch_idx=0,
                )
            else:
                # Standard cache path (non-paged) - uses absolute layer_idx
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                # Get unexpanded KV for cache
                kv_key = key_states[:, :: self.num_key_value_groups, :, :]
                kv_value = value_states[:, :: self.num_key_value_groups, :, :]

                torch_tensors = [TorchTTNNTensor(kv_key), TorchTTNNTensor(kv_value)]
                orig_shapes = [kv_key.shape, kv_value.shape]

                torch_tensors = [
                    torch_tensor.to_torch[: orig_shape[0], : orig_shape[1], : orig_shape[2], : orig_shape[3]]
                    for orig_shape, torch_tensor in zip(orig_shapes, torch_tensors)
                ]

                cached_key, cached_value = past_key_values.update(
                    *torch_tensors,
                    self.layer_idx,  # Standard cache uses absolute layer_idx
                    cache_kwargs,
                )
                cached_key, cached_value = [TorchTTNNTensor(cached_key), TorchTTNNTensor(cached_value)]
                cached_key = ttnn.to_device(cached_key.to_ttnn, self.device)
                cached_value = ttnn.to_device(cached_value.to_ttnn, self.device)
                cached_key = self._maybe_all_gather(cached_key)
                cached_value = self._maybe_all_gather(cached_value)

                # Expand cached KV for attention
                key_states = self._repeat_kv(cached_key, self.num_key_value_groups)
                value_states = self._repeat_kv(cached_value, self.num_key_value_groups)

        # Compute attention
        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=True,
        )

        # Reshape output: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
        # Use actual tensor shape after SDPA
        attn_shape = list(attn_output.shape)
        attn_batch = attn_shape[0]
        attn_seq = attn_shape[1]
        attn_output = ttnn.reshape(attn_output, (attn_batch, attn_seq, self.num_attention_heads * self.head_dim))

        # Apply Q gating: attn_output = attn_output * sigmoid(gate)
        # PyTorch: attn_output = attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        if gate is not None:
            gate_sigmoid = ttnn.sigmoid(gate)
            attn_output = ttnn.mul(attn_output, gate_sigmoid)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _forward_decode_paged(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Decode path using paged attention with on-device KV cache.

        TTNN paged kernels require tensors in [1, batch, heads, head_dim]
        layout (S B H D) whereas _project_qkv returns the standard
        [batch, heads, seq, head_dim] (B H S D). This method handles
        the permute, L1 sharding required by paged_update_cache, and
        the GQA-aware SDPA decode call.
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        query_states, key_states, value_states, gate, cos, sin = self._project_qkv(
            hidden_states, batch_size, seq_length, position_embeddings
        )
        # _project_qkv returns [B, H, S, D]:
        #   Q : [B, num_attention_heads, 1, head_dim]
        #   K : [B, num_attention_heads, 1, head_dim] (already expanded)
        #   V : [B, num_attention_heads, 1, head_dim] (already expanded)
        # gate: [B, S, num_heads * head_dim] or None

        layer_idx = self.layer_idx

        # Get unexpanded KV for cache update
        kv_key = key_states[:, :: self.num_key_value_groups, :, :]
        kv_value = value_states[:, :: self.num_key_value_groups, :, :]

        # --- resolve cache position to a 1-D torch int32 tensor [batch] ---
        if cache_position is None:
            cur_pos = past_key_values.get_seq_length(layer_idx)
            cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
        else:
            cp = cache_position
            if isinstance(cp, TorchTTNNTensor):
                cp = cp.to_torch
            if isinstance(cp, ttnn.Tensor):
                mesh_composer = None
                if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
                    mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
                cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)
            cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        # Trace-safe cur_pos: copy into pre-allocated buffer instead of
        # allocating a new device tensor each step (matching Gemma4 pattern).
        # During trace replay, ttnn.from_torch() allocations are frozen;
        # ttnn.copy() into a stable address keeps the trace valid.
        cur_pos_host = ttnn.from_torch(
            cache_position_tensor,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        if self._decode_cur_pos is not None:
            cur_pos_device = ttnn.to_device(cur_pos_host, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(cur_pos_device, self._decode_cur_pos)
            cur_pos_tt = self._decode_cur_pos
        else:
            # Fallback: allocate on device (non-trace path or before move_weights_to_device)
            cur_pos_tt = ttnn.to_device(cur_pos_host, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # --- permute B H S D -> S B H D (the layout paged kernels expect) ---
        query_states_paged = ttnn.permute(query_states, (2, 0, 1, 3))
        kv_key = ttnn.permute(kv_key, (2, 0, 1, 3))
        kv_value = ttnn.permute(kv_value, (2, 0, 1, 3))

        # --- multi-device: convert all-gathered topology -> replicated ---
        if self.device.get_num_devices() > 1:
            query_states_paged = self._to_replicated(query_states_paged)
            kv_key = self._to_replicated(kv_key)
            kv_value = self._to_replicated(kv_value)

        tile_size = 32
        shard_h = ((self.num_key_value_heads + tile_size - 1) // tile_size) * tile_size

        core_grid = ttnn.CoreGrid(y=1, x=batch_size)
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(shard_h, self.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        kv_key = ttnn.to_memory_config(kv_key, shard_cfg)
        kv_value = ttnn.to_memory_config(kv_value, shard_cfg)

        # --- update the on-device paged KV cache ---
        # NOTE: _seq_lengths / _seen_tokens are updated by the caller via
        # update_seq_length() outside the trace boundary.
        past_key_values.paged_update_on_device(
            kv_key,
            kv_value,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(kv_key)
        ttnn.deallocate(kv_value)

        # --- paged SDPA decode (Q stays in DRAM) ---
        # Note: For GQA, the paged attention handles the KV head expansion internally
        attn_output = past_key_values.paged_sdpa_decode(
            query_states_paged,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.program_config,
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )
        # attn_output: [1, B, H, head_dim]

        # --- convert back to [B, S, H*D] for the output projection ---
        attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, head_dim]
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_attention_heads * self.head_dim))

        # Apply Q gating: attn_output = attn_output * sigmoid(gate)
        if gate is not None:
            gate_sigmoid = ttnn.sigmoid(gate)
            attn_output = ttnn.mul(attn_output, gate_sigmoid)

        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Qwen3 full attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask
            past_key_values: Optional KV cache (TTNNQwenPagedAttentionKVCache or DynamicCache)
            cache_position: Optional cache position tensor
            position_ids: Optional position IDs (unused, for API compatibility)
            **kwargs: Additional arguments for forward compatibility

        Returns:
            Tuple of (attention_output, None)
        """
        # CPU fallback for debugging
        use_cpu_fallback = os.environ.get("TT_QWEN_CPU_FULL_ATTN", "0").lower() in ("1", "true", "yes")
        if use_cpu_fallback:
            # Log only once per layer instance
            if not getattr(self, "_cpu_fallback_logged", False):
                print(f"[DEBUG] TT_QWEN_CPU_FULL_ATTN=1: Using PyTorch for full attention (layer {self.layer_idx})")
                self._cpu_fallback_logged = True

            # Helper to convert any tensor type to PyTorch, handling multi-device tensors
            def to_pytorch(t):
                if t is None:
                    return None

                # Get the underlying TTNN tensor and distributed config if it's a TorchTTNNTensor wrapper
                ttnn_t = None
                dist_config = None
                if hasattr(t, "ttnn_tensor") and t.ttnn_tensor is not None:
                    ttnn_t = t.ttnn_tensor
                    if hasattr(t, "ttnn_distributed_tensor_config"):
                        dist_config = t.ttnn_distributed_tensor_config
                elif isinstance(t, ttnn.Tensor):
                    ttnn_t = t

                # If we have a TTNN tensor, convert it with mesh composer
                if ttnn_t is not None:
                    mesh_composer = None
                    is_replicated = False
                    try:
                        dev = ttnn_t.device()
                        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                            # Use the tensor's configured mesh_composer if available
                            if dist_config is not None and hasattr(dist_config, "mesh_composer"):
                                mesh_composer = dist_config.mesh_composer
                                # Check if it's replicated (post_process_fn is set to slice first element)
                                if hasattr(dist_config, "post_process_fn") and dist_config.post_process_fn is not None:
                                    is_replicated = True
                            else:
                                # Default to concat on dim=-1 (col-sharded is most common for hidden_states)
                                mesh_composer = ttnn.ConcatMeshToTensor(dev, dim=-1)
                    except Exception:
                        pass
                    result = ttnn.to_torch(ttnn_t, mesh_composer=mesh_composer)
                    # Apply post-processing for replicated tensors
                    if (
                        dist_config is not None
                        and hasattr(dist_config, "post_process")
                        and callable(dist_config.post_process)
                    ):
                        result = dist_config.post_process(result)
                    return result

                # If the TorchTTNNTensor has a PyTorch elem, try to use it
                if hasattr(t, "elem") and t.elem is not None:
                    return t.elem

                # Fallback - just return as-is (already PyTorch tensor)
                return t

            # Convert hidden_states to PyTorch
            hs_pt = to_pytorch(hidden_states)

            # Convert position_embeddings tuple (cos, sin) to PyTorch
            if isinstance(position_embeddings, (tuple, list)):
                pos_emb_pt = tuple(to_pytorch(p) for p in position_embeddings)
            else:
                pos_emb_pt = to_pytorch(position_embeddings)

            # Convert cache_position to PyTorch
            cache_pos_pt = to_pytorch(cache_position)

            # Convert attention_mask to PyTorch
            attn_mask_pt = to_pytorch(attention_mask)

            return self._fallback_torch_layer(
                hs_pt,
                position_embeddings=pos_emb_pt,
                attention_mask=attn_mask_pt,
                past_key_values=None,  # Can't use TTNN paged cache with PyTorch
                cache_position=cache_pos_pt,
                **kwargs,
            )

        seq_length = hidden_states.shape[1]

        use_paged = isinstance(past_key_values, TTNNQwenPagedAttentionKVCache) or isinstance(
            past_key_values, TTNNPagedAttentionKVCache
        )

        if use_paged and seq_length == 1:
            ttnn_output, _ = self._forward_decode_paged(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
            )
        else:
            ttnn_output, _ = self._forward_prefill(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
            )

        return ttnn_output, None


class TTNNQwen3LinearAttention(TTNNModule):
    """TTNN-accelerated Linear Attention (DeltaNet/Mamba-style) for Qwen3.5-35B-A3B.

    This module handles 30/40 layers in Qwen3.5 that use linear attention instead
    of full attention. Linear attention uses state-space model computation:
    - NO KV cache (state is computed directly each step)
    - O(n) complexity vs O(n^2) for full attention
    - Uses gating mechanisms (beta, g) and 1D convolutions

    Key components from Qwen3_5MoeGatedDeltaNet:
    - in_proj_qkv: Projects hidden_states to Q, K, V
    - in_proj_z: Projects for gating
    - in_proj_a, in_proj_b: Projects for alpha/beta gates
    - conv1d: Causal convolution for sequential processing
    - norm: RMS norm with gating
    - out_proj: Output projection

    Implementation Strategy (Hybrid):
    - TTNN acceleration for linear projections (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj)
    - PyTorch fallback for DeltaNet kernel (conv1d, gating, chunk_gated_delta_rule)
    - Feature-flagged via TTNN_LINEAR_ATTN_PROJECTIONS env var (default: enabled)
    """

    def __init__(self, config=None, distributed: bool = True):
        super().__init__()
        self.config = config
        self.distributed = distributed

        # Layer dimensions (will be set from torch layer)
        self.hidden_size = None
        self.num_v_heads = None
        self.num_k_heads = None
        self.head_k_dim = None
        self.head_v_dim = None
        self.key_dim = None
        self.value_dim = None
        self.conv_kernel_size = None
        self.layer_idx = None

        # TTNN linear projections
        self.in_proj_qkv = None  # TTNNLinear: hidden_size -> key_dim * 2 + value_dim
        self.in_proj_z = None  # TTNNLinear: hidden_size -> value_dim
        self.in_proj_a = None  # TTNNLinear: hidden_size -> num_v_heads
        self.in_proj_b = None  # TTNNLinear: hidden_size -> num_v_heads
        self.out_proj = None  # TTNNLinear: value_dim -> hidden_size

        # DeltaNet kernel parameters (kept on PyTorch)
        self.conv1d = None  # PyTorch Conv1d for causal convolution
        self.dt_bias = None  # Time step bias
        self.A_log = None  # A parameter (log space)
        self.norm = None  # RMSNorm with gating

        # Feature flag for TTNN projections (default: enabled)
        self.use_ttnn_projections = os.environ.get("TTNN_LINEAR_ATTN_PROJECTIONS", "1") == "1"

    @classmethod
    def from_torch(cls, torch_layer, distributed: bool = True):
        """Create TTNNQwen3LinearAttention from PyTorch Qwen3_5MoeGatedDeltaNet.

        Args:
            torch_layer: PyTorch Qwen3_5MoeGatedDeltaNet layer
            distributed: Whether to use distributed linear layers (default True for T3K)

        Returns:
            TTNNQwen3LinearAttention instance
        """
        config = torch_layer.config if hasattr(torch_layer, "config") else None
        new_layer = cls(config, distributed=distributed)
        new_layer._fallback_torch_layer = torch_layer

        # Extract layer dimensions
        new_layer.hidden_size = torch_layer.hidden_size
        new_layer.num_v_heads = torch_layer.num_v_heads
        new_layer.num_k_heads = torch_layer.num_k_heads
        new_layer.head_k_dim = torch_layer.head_k_dim
        new_layer.head_v_dim = torch_layer.head_v_dim
        new_layer.key_dim = torch_layer.key_dim
        new_layer.value_dim = torch_layer.value_dim
        new_layer.conv_kernel_size = torch_layer.conv_kernel_size
        new_layer.layer_idx = torch_layer.layer_idx

        # Choose linear layer classes based on distributed mode
        # Input projections: replicated input -> col-sharded weights -> sharded output
        # Output projection: sharded input -> row-sharded weights -> replicated output (with all-reduce)
        # Note: in_proj_a and in_proj_b have small output dims (num_v_heads=4) that can't be sharded,
        # so they use non-sharded linear layers
        LinearClsIn = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        LinearClsSmall = TTNNLinear  # Always non-sharded for small projections

        # Create TTNN linear projections
        # These take replicated input (full hidden_states) and produce sharded output
        new_layer.in_proj_qkv = LinearClsIn.from_torch(torch_layer.in_proj_qkv)
        new_layer.in_proj_z = LinearClsIn.from_torch(torch_layer.in_proj_z)
        # in_proj_a and in_proj_b have tiny output dims (num_v_heads=4) that can't be col-sharded
        # Keep as PyTorch layers to avoid distributed weight replication issues
        # (TTNNLinear doesn't replicate weights across mesh devices, causing garbage on non-device-0)
        new_layer.in_proj_a = torch_layer.in_proj_a  # PyTorch nn.Linear
        new_layer.in_proj_b = torch_layer.in_proj_b  # PyTorch nn.Linear
        new_layer.out_proj = LinearClsOut.from_torch(torch_layer.out_proj)

        # Keep DeltaNet kernel components as references (not TTNN)
        new_layer.conv1d = torch_layer.conv1d
        new_layer.dt_bias = torch_layer.dt_bias
        new_layer.A_log = torch_layer.A_log
        new_layer.norm = torch_layer.norm

        # Store kernel functions
        new_layer.causal_conv1d_fn = torch_layer.causal_conv1d_fn
        new_layer.causal_conv1d_update = torch_layer.causal_conv1d_update
        new_layer.chunk_gated_delta_rule = torch_layer.chunk_gated_delta_rule
        new_layer.recurrent_gated_delta_rule = torch_layer.recurrent_gated_delta_rule
        new_layer.activation = torch_layer.activation

        return new_layer

    def deallocate_weights_impl(self):
        """Deallocate TTNN weights from device.

        Note: in_proj_a and in_proj_b are PyTorch layers, not TTNN, so we skip them.
        """
        if self.in_proj_qkv is not None:
            self.in_proj_qkv.deallocate_weights()
        if self.in_proj_z is not None:
            self.in_proj_z.deallocate_weights()
        # in_proj_a and in_proj_b are PyTorch nn.Linear layers, not TTNN modules
        # They don't have deallocate_weights() method
        if self.out_proj is not None:
            self.out_proj.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        """Set output tensor config for col-sharded output.

        The out_proj output is col-sharded (each device has [batch, seq, hidden_size/8]).
        We need to use ConcatMeshToTensor on dim=-1 to concatenate the shards.
        """

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    # Use ConcatMeshToTensor on dim=-1 only (not batch dim)
                    # This concatenates the col-sharded output from all devices
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        """Compute logical shape by multiplying last dim by num_devices."""
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    config = DistributedTensorConfig(
                        mesh_mapper=mesh_mapper,
                        mesh_composer=mesh_composer,
                        logical_shape_fn=logical_shape_for_col_sharded,
                    )
                    e.set_distributed_tensor_config(config)
            return e

        # Use the default config from parent if not distributed
        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)

        return tree_map(set_col_sharded_config, output_tensors)

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager.

        Returns True only if:
        1. Layer was created with distributed=True (from_torch)
        2. Device state has a CCL manager for all-gather operations
        """
        return (
            getattr(self, "distributed", True)  # Check distributed flag from from_torch
            and self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor
        t = tensor
        gathered = ttnn.all_gather(
            t,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        # Synchronize to ensure all-gather completes before returning
        ttnn.synchronize_device(self.device)
        return gathered

    def _is_tensor_replicated(self, tensor) -> bool:
        """Check if tensor is replicated across devices (vs sharded).

        Returns True if tensor uses ReplicateTensorToMesh, False otherwise.
        This allows auto-detection of the correct conversion mode.
        """
        if tensor is None:
            return False

        # Check if tensor has distributed config indicating replication
        if hasattr(tensor, "ttnn_distributed_tensor_config"):
            config = tensor.ttnn_distributed_tensor_config
            if config is not None:
                # KEY FIX: If tensor has a logical_shape_fn, physical shape differs from logical
                # This means it's sharded (each device has a portion of the data)
                # ShardTensorToMesh returns CppTensorToMesh whose name doesn't contain "Shard",
                # so we must check logical_shape_fn instead of relying on mapper type name
                if config.logical_shape_fn is not None:
                    return False  # Sharded, not replicated

                if hasattr(config, "mesh_mapper"):
                    mapper = config.mesh_mapper
                    # Check if it's a ReplicateTensorToMesh mapper
                    mapper_type = type(mapper).__name__
                    if "Replicate" in mapper_type:
                        return True
                    # If it's Shard, return False
                    if "Shard" in mapper_type:
                        return False
                # Config exists but doesn't indicate replication
                return False

        # NO CONFIG CASE: Need to check physical shape directly
        # For TorchTTNNTensor, we must access the underlying ttnn_tensor shape
        # (NOT the .shape property which returns logical shape)

        physical_shape = None

        if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
            # TorchTTNNTensor with underlying ttnn tensor - get PHYSICAL shape
            physical_shape = tuple(int(i) for i in tensor.ttnn_tensor.shape)
        elif isinstance(tensor, ttnn.Tensor):
            # Raw ttnn.Tensor - shape is already physical
            physical_shape = tuple(int(i) for i in tensor.shape)
        else:
            # Regular torch tensor or no way to get physical shape
            if hasattr(tensor, "shape") and tensor.shape is not None:
                physical_shape = tuple(tensor.shape)

        if physical_shape is not None and len(physical_shape) >= 1 and self.device is not None:
            num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
            if num_devices > 1:
                last_dim = physical_shape[-1]
                # If last dim equals hidden_size, tensor is replicated (full data per device)
                if last_dim == self.hidden_size:
                    return True
                # If last dim equals hidden_size/num_devices, tensor is sharded
                elif last_dim == self.hidden_size // num_devices:
                    return False

        # Unable to determine, default to sharded (safer for distributed ops)
        return False

    def _to_pytorch(self, tensor, replicated=None):
        """Convert TTNN tensor to PyTorch, handling multi-device meshes.

        Args:
            tensor: TTNN or TorchTTNNTensor to convert
            replicated: If True, tensor is replicated (same data on all devices),
                        so we take from first device instead of concatenating.
                        If None, auto-detect based on tensor config.
                        Use True for tensors after all_gather operations.
        """
        # Auto-detect if not specified
        if replicated is None:
            replicated = self._is_tensor_replicated(tensor)
        if tensor is None:
            return None

        # Get original batch size BEFORE conversion to handle multi-device slicing correctly
        original_batch_size = 1
        if hasattr(tensor, "shape") and tensor.shape is not None:
            shape = tensor.shape
            if len(shape) > 0:
                original_batch_size = shape[0]

        # Handle raw ttnn.Tensor objects directly (e.g., from TTNN projections)
        if isinstance(tensor, ttnn.Tensor):
            try:
                device = tensor.device()
                if device is not None and hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                    num_devices = device.get_num_devices()
                    if replicated:
                        # For replicated tensors, concat on dim=0 then take first batch
                        pt_tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                        batch_per_device = pt_tensor.shape[0] // num_devices
                        return pt_tensor[:batch_per_device]
                    else:
                        # For sharded tensors, concat on last dim to get full tensor
                        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
                else:
                    return ttnn.to_torch(tensor)
            except Exception as e:
                import logging

                logging.warning(f"TTNNQwen3LinearAttention._to_pytorch: Error converting raw ttnn.Tensor: {e}")
                return ttnn.to_torch(tensor)

        if hasattr(tensor, "to_torch") and callable(tensor.to_torch):
            # Check if tensor is on multi-device mesh
            try:
                device = getattr(tensor, "device", None)
                if device is not None:
                    if callable(device):
                        device = device()
                    if device is not None:
                        num_devices = 1
                        mesh_shape = getattr(device, "shape", None)
                        if mesh_shape is not None and hasattr(mesh_shape, "num_devices"):
                            num_devices = mesh_shape.num_devices
                        elif hasattr(device, "get_num_devices"):
                            num_devices = device.get_num_devices()

                        if num_devices > 1:
                            if replicated:
                                # For replicated tensors (after all-gather), take from first device
                                # Using ConcatMeshToTensor on dim=0 and then slicing gives us one copy
                                pt_tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                                return pt_tensor[0:original_batch_size]  # Use original batch size for slicing
                            else:
                                # For sharded tensors, concatenate along last dim
                                return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
            except Exception as e:
                import logging

                logging.warning(f"TTNNQwen3LinearAttention._to_pytorch: Error during mesh handling: {e}")
            return tensor.to_torch()
        elif isinstance(tensor, TorchTTNNTensor):
            # For TorchTTNNTensor, use the underlying ttnn_tensor if available
            if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
                # Get batch size from TorchTTNNTensor shape
                if hasattr(tensor, "shape") and tensor.shape is not None and len(tensor.shape) > 0:
                    original_batch_size = tensor.shape[0]
                device = tensor.ttnn_tensor.device() if hasattr(tensor.ttnn_tensor, "device") else None
                if device is not None and hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                    if replicated:
                        pt_tensor = ttnn.to_torch(
                            tensor.ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
                        )
                        return pt_tensor[0:original_batch_size]  # Use original batch size for slicing
                    else:
                        return ttnn.to_torch(tensor.ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
            return tensor.torch_tensor
        elif hasattr(tensor, "ttnn_tensor"):
            # Get batch size from wrapper object
            if hasattr(tensor, "shape") and tensor.shape is not None and len(tensor.shape) > 0:
                original_batch_size = tensor.shape[0]
            mesh_composer = None
            if hasattr(tensor.ttnn_tensor, "device") and tensor.ttnn_tensor.device() is not None:
                device = tensor.ttnn_tensor.device()
                if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                    if replicated:
                        pt_tensor = ttnn.to_torch(
                            tensor.ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
                        )
                        return pt_tensor[0:original_batch_size]  # Use original batch size for slicing
                    else:
                        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=-1)
            return ttnn.to_torch(tensor.ttnn_tensor, mesh_composer=mesh_composer)
        return tensor

    def _to_ttnn(self, tensor):
        """Convert PyTorch tensor to TTNN tensor on device."""
        if tensor is None:
            return None
        if self.device is None:
            return tensor
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        return ttnn.from_torch(
            tensor,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_pytorch_replicated(self, tensor):
        """Convert a tensor that's known to be replicated (identical on all devices).

        After all_gather, all devices have identical data. This method properly
        handles the extraction without relying on auto-detection.
        """
        if tensor is None:
            return None
        if isinstance(tensor, ttnn.Tensor):
            device = tensor.device()
            if device is not None and device.get_num_devices() > 1:
                # After all_gather, all devices have identical data.
                # Use ConcatMeshToTensor with dim=0 and take first element (all devices have identical data)
                mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
                pt_tensor = ttnn.to_torch(tensor, mesh_composer=mesh_composer)
                return pt_tensor[0].unsqueeze(0).contiguous()  # [0] removes batch, unsqueeze adds it back
            return ttnn.to_torch(tensor).contiguous()
        elif isinstance(tensor, TorchTTNNTensor):
            return self._to_pytorch_replicated(tensor.ttnn_tensor)
        elif hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
            return self._to_pytorch_replicated(tensor.ttnn_tensor)
        # Already PyTorch tensor
        if hasattr(tensor, "contiguous"):
            return tensor.contiguous()
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through Qwen3 linear attention with TTNN-accelerated projections.

        Uses TTNN for linear projections and PyTorch for DeltaNet kernel.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            cache_params: Optional Qwen3_5MoeDynamicCache for recurrent state
            cache_position: Optional position tensor for decode
            attention_mask: Optional attention mask
            **kwargs: Additional arguments for forward compatibility

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # DEBUG: Compare INPUT with CPU reference at the START of forward
        import os

        if os.environ.get("DEBUG_LINEAR_ATTN_INPUT", "0") == "1":
            # Get input as PyTorch
            if hasattr(hidden_states, "elem") and hidden_states.elem is not None:
                hs_input = hidden_states.elem
                input_source = "elem"
            elif hasattr(hidden_states, "ttnn_tensor") and hidden_states.ttnn_tensor is not None:
                hs_input = self._to_pytorch(hidden_states)
                input_source = "ttnn_tensor"
            else:
                hs_input = hidden_states
                input_source = "raw"

            # Check config
            config_info = "None"
            if hasattr(hidden_states, "ttnn_distributed_tensor_config"):
                cfg = hidden_states.ttnn_distributed_tensor_config
                if cfg is not None:
                    config_info = f"logical_shape_fn={cfg.logical_shape_fn is not None}, mapper={type(cfg.mesh_mapper).__name__ if cfg.mesh_mapper else 'None'}"

            print(f"[INPUT DEBUG] Layer {self.layer_idx}:")
            print(f"  source={input_source}, shape={hs_input.shape}, config={config_info}")
            print(f"  type={type(hs_input).__name__}")

        # If TTNN projections disabled or no device, use pure PyTorch fallback
        if not self.use_ttnn_projections or self.device is None:
            # Layer 0: input from embedding is REPLICATED (full hidden_size on each device)
            # Layers 1-39: input from previous MoE is COL-SHARDED (hidden_size/num_devices)
            # Auto-detect via _is_tensor_replicated() to handle both cases correctly
            hidden_states_pt = self._to_pytorch(hidden_states)  # Auto-detect via _is_tensor_replicated
            cache_position_pt = self._to_pytorch(cache_position)  # Auto-detect
            attention_mask_pt = self._to_pytorch(attention_mask)  # Auto-detect

            output = self._fallback_torch_layer(
                hidden_states_pt,
                cache_params=cache_params,
                cache_position=cache_position_pt,
                attention_mask=attention_mask_pt,
            )

            # When using pure PyTorch fallback, return raw PyTorch tensor
            # Don't wrap in TorchTTNNTensor to avoid distributed config issues
            # The output is a regular CPU tensor that will flow through subsequent layers
            return output

        # === HYBRID FORWARD: TTNN projections + PyTorch DeltaNet kernel ===

        # ALL-GATHER INPUT IF SHARDED: Ensure we have full hidden_size before projections
        # The projections expect replicated input [batch, seq, hidden_size]
        # If input is col-sharded (from previous MoE), all-gather it first
        if self._is_distributed and not self._is_tensor_replicated(hidden_states):
            hidden_states = self._maybe_all_gather(hidden_states)

        # Apply mask to hidden states (from PyTorch implementation)
        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_mask_to_padding_states

            # After all-gather, input is REPLICATED (full hidden_size on each device)
            # Use explicit replicated=True since we just all-gathered
            hidden_states_pt = self._to_pytorch(hidden_states, replicated=True)
            attention_mask_pt = self._to_pytorch(attention_mask)  # Auto-detect
            hidden_states_masked = apply_mask_to_padding_states(hidden_states_pt, attention_mask_pt)
            # Convert back to TTNN for projections
            hidden_states_ttnn = self._to_ttnn(hidden_states_masked)
        except ImportError:
            hidden_states_ttnn = hidden_states

        batch_size, seq_len, _ = hidden_states_ttnn.shape

        # Linear attention uses a different cache format than paged attention
        # Check if cache_params is compatible with linear attention (has_previous_state, conv_states, recurrent_states)
        is_linear_attn_cache = (
            cache_params is not None
            and hasattr(cache_params, "has_previous_state")
            and hasattr(cache_params, "conv_states")
            and hasattr(cache_params, "recurrent_states")
        )

        use_precomputed_states = is_linear_attn_cache and cache_params.has_previous_state and seq_len == 1

        # Get cache states if available (only for linear attention-compatible caches)
        # Use .get() to safely access - during first forward pass, these may not exist yet
        conv_state = None
        recurrent_state = None
        if is_linear_attn_cache:
            conv_state = cache_params.conv_states.get(self.layer_idx)
            recurrent_state = cache_params.recurrent_states.get(self.layer_idx)

        # === TTNN Linear Projections ===
        # Ensure tile layout for TTNN operations
        if hidden_states_ttnn.layout != ttnn.TILE_LAYOUT:
            hidden_states_ttnn = ttnn.to_layout(
                hidden_states_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # Project: in_proj_qkv -> [batch, seq, key_dim * 2 + value_dim]
        mixed_qkv_ttnn = self.in_proj_qkv(hidden_states_ttnn)

        # Project: in_proj_z -> [batch, seq, value_dim]
        z_ttnn = self.in_proj_z(hidden_states_ttnn)

        # All-gather for distributed mode (results are replicated - same data on all devices)
        mixed_qkv_ttnn = self._maybe_all_gather(mixed_qkv_ttnn)
        z_ttnn = self._maybe_all_gather(z_ttnn)

        # === Convert to PyTorch for DeltaNet kernel ===
        # Use explicit replicated conversion since all_gather makes them replicated
        # The _to_pytorch_replicated method properly handles multi-device extraction
        mixed_qkv = self._to_pytorch_replicated(mixed_qkv_ttnn)
        z = self._to_pytorch_replicated(z_ttnn)

        # DEBUG: Compare with PyTorch reference
        import os

        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            # Get hidden_states as PyTorch - auto-detect replicated vs sharded
            if (
                isinstance(hidden_states, ttnn.Tensor)
                or hasattr(hidden_states, "ttnn_tensor")
                or hasattr(hidden_states, "to_ttnn")
            ):
                hs_pt = self._to_pytorch(hidden_states)  # Auto-detect via _is_tensor_replicated
            else:
                hs_pt = hidden_states

            # Run PyTorch reference
            mixed_qkv_ref = self._fallback_torch_layer.in_proj_qkv(hs_pt)
            z_ref = self._fallback_torch_layer.in_proj_z(hs_pt)

            print(f"[DEBUG] hidden_states shape for reference: {hs_pt.shape}")
            print(f"[DEBUG] mixed_qkv shape: TTNN={mixed_qkv.shape}, PyTorch={mixed_qkv_ref.shape}")
            print(f"[DEBUG] z shape: TTNN={z.shape}, PyTorch={z_ref.shape}")
            print(f"[DEBUG] mixed_qkv TTNN sample: {mixed_qkv[0,0,:5].tolist()}")
            print(f"[DEBUG] mixed_qkv PyTorch sample: {mixed_qkv_ref[0,0,:5].tolist()}")

            # Max absolute difference
            diff = (mixed_qkv.float() - mixed_qkv_ref.float()).abs()
            print(f"[DEBUG] mixed_qkv max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")

            z_diff = (z.float() - z_ref.float()).abs()
            print(f"[DEBUG] z max diff: {z_diff.max().item():.6f}, mean: {z_diff.mean().item():.6f}")

        # Get hidden_states as PyTorch tensor for small projections
        # in_proj_a and in_proj_b are PyTorch nn.Linear layers (not TTNN) to avoid
        # distributed weight replication issues - they have tiny output dims (num_v_heads=4)
        # hidden_states_ttnn was converted via _to_ttnn which replicates, so use replicated conversion
        hidden_states_pt = self._to_pytorch_replicated(hidden_states_ttnn)
        b = self.in_proj_b(hidden_states_pt).contiguous()  # PyTorch nn.Linear -> [batch, seq, num_v_heads]
        a = self.in_proj_a(hidden_states_pt).contiguous()  # PyTorch nn.Linear -> [batch, seq, num_v_heads]

        # DEBUG: Compare a and b projections with PyTorch reference
        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            # Get hs_pt for reference if not already available
            if "hs_pt" not in dir():
                if (
                    isinstance(hidden_states, ttnn.Tensor)
                    or hasattr(hidden_states, "ttnn_tensor")
                    or hasattr(hidden_states, "to_ttnn")
                ):
                    hs_pt = self._to_pytorch(hidden_states)  # Auto-detect via _is_tensor_replicated
                else:
                    hs_pt = hidden_states
            a_ref = self._fallback_torch_layer.in_proj_a(hs_pt)
            b_ref = self._fallback_torch_layer.in_proj_b(hs_pt)
            print(f"[DEBUG] a max diff: {(a.float() - a_ref.float()).abs().max().item():.6f}")
            print(f"[DEBUG] b max diff: {(b.float() - b_ref.float()).abs().max().item():.6f}")

        # Correct batch_size based on actual converted tensor shapes (not input shape which may be inflated)
        # The _to_pytorch with replicated=True extracts data for a single batch from all devices
        actual_batch_size = mixed_qkv.shape[0]
        if actual_batch_size != batch_size:
            batch_size = actual_batch_size

        # If tensors have 8x batch from mesh replication, take first slice
        if mixed_qkv.shape[0] > batch_size:
            mixed_qkv = mixed_qkv[:batch_size]
            z = z[:batch_size]
            b = b[:batch_size]
            a = a[:batch_size]

        # Reshape z for gated norm: [batch, seq, num_v_heads, head_v_dim]
        # Call contiguous() to ensure memory layout is correct for DeltaNet kernel
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim).contiguous()

        # Transpose mixed_qkv for conv1d: [batch, key_dim * 2 + value_dim, seq]
        # Call contiguous() after transpose to ensure memory layout is correct
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()

        # === PyTorch DeltaNet Kernel ===
        if use_precomputed_states:
            # Decode path: use causal_conv1d_update
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # Prefill path: use causal_conv1d_fn or fallback
            if is_linear_attn_cache:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state

            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # Transpose back: [batch, seq, key_dim * 2 + value_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()

        # Split into Q, K, V
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        # Reshape for multi-head attention - ensure contiguous memory for DeltaNet kernel
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim).contiguous()

        # Compute gating parameters
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Repeat Q, K for value heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # Run delta rule kernel
        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=is_linear_attn_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=is_linear_attn_cache,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache (only for linear attention-compatible caches)
        if is_linear_attn_cache:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # DEBUG: Print core_attn_out info after DeltaNet kernel
        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            print(f"[DEBUG] core_attn_out shape: {core_attn_out.shape}, sample: {core_attn_out[0,0,:3].tolist()}")

        # Apply gated RMS norm
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # === TTNN Output Projection ===
        # Convert back to TTNN for output projection
        core_attn_out_ttnn = self._to_ttnn(core_attn_out)
        if core_attn_out_ttnn.layout != ttnn.TILE_LAYOUT:
            core_attn_out_ttnn = ttnn.to_layout(
                core_attn_out_ttnn, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        output_ttnn = self.out_proj(core_attn_out_ttnn)

        # DEBUG: Compare out_proj with PyTorch reference
        if os.environ.get("DEBUG_LINEAR_ATTN", "0") == "1":
            out_ref = self._fallback_torch_layer.out_proj(core_attn_out)
            out_ttnn_pt = self._to_pytorch(output_ttnn, replicated=False)  # col-sharded output
            print(f"[DEBUG] out_proj max diff: {(out_ttnn_pt.float() - out_ref.float()).abs().max().item():.6f}")

        # NOTE: Do NOT all_gather here. Output should stay col-sharded to match:
        # 1. Full attention output (also col-sharded)
        # 2. MoE input expectation (does all_gather internally)

        # DEBUG: Compare FULL LAYER OUTPUT with CPU fallback
        if os.environ.get("DEBUG_LINEAR_ATTN_OUTPUT", "0") == "1":
            # Get the input as PyTorch tensor
            hs_input = self._to_pytorch(hidden_states)

            # Run CPU fallback for comparison - may return 1 or 2 values
            cpu_result = self._fallback_torch_layer(
                hs_input,
                cache_params=None,
                cache_position=cache_position.cpu() if cache_position is not None else None,
                attention_mask=self._to_pytorch(attention_mask) if attention_mask is not None else None,
            )
            cpu_output = cpu_result[0] if isinstance(cpu_result, tuple) else cpu_result

            # Get TTNN output as PyTorch
            ttnn_output = self._to_pytorch(output_ttnn, replicated=False)  # col-sharded

            # Compare
            diff = (ttnn_output.float() - cpu_output.float()).abs()
            print(f"[LAYER OUTPUT DEBUG] Layer {self.layer_idx}:")
            print(f"  TTNN shape: {ttnn_output.shape}, CPU shape: {cpu_output.shape}")
            print(f"  Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
            if diff.max().item() > 1.0:
                print(f"  TTNN sample: {ttnn_output[0,0,:5].tolist()}")
                print(f"  CPU sample: {cpu_output[0,0,:5].tolist()}")

        # out_proj returns TorchTTNNTensor already, return it directly
        return output_ttnn


# === content from models/experimental/tt_symbiote/modules/qwen_moe.py ===
"""Qwen3.5-35B-A3B specific MoE implementations for TTNN.

This module contains Qwen-specific subclasses that inherit from the GLM base classes
in moe.py. Key differences:
- TTNNQwenMoERouterDecode: Uses softmax activation instead of sigmoid
- TTNNQwenExperts: Uses sparse_matmul with fused w1/w3 (gate/up) projections
- TTNNQwen3MoE: Handles Qwen's shared_expert (singular) and optional shared_expert_gate

Environment Variables:
- TT_QWEN_CPU_EXPERTS: Set to "1" to use CPU fallback for experts (for debugging).
  When enabled, TTNNQwenExperts is NOT created and the PyTorch experts are used instead.
"""


class TTNNQwenMoERouterDecode(TTNNMoERouterDecode):
    """Qwen-specific router using simple softmax -> topk -> normalize.

    Qwen3.5 architecture uses a straightforward routing algorithm:
        scores = softmax(logits)
        top_values, top_indices = topk(scores, k)
        weights = top_values / sum(top_values)
        weights *= routed_scaling_factor

    No bias, no groups, no group masks -- just simple topk on softmax scores.

    Inheritance:
        - from_torch(): Inherited (unchanged)
        - preprocess_weights_impl(): OVERRIDDEN - only creates scale tensor (no bias/scatter)
        - move_weights_to_device_impl(): OVERRIDDEN - only moves scale tensor
        - forward(): OVERRIDDEN - simple softmax -> topk -> normalize -> scale
    """

    def preprocess_weights_impl(self):
        """Preprocess weights: only the routing scale tensor is needed.

        Qwen3.5 routing has no bias and no group-based selection, so we skip
        creating the bias, scatter_input, and scatter_src tensors that the
        parent class creates for GLM4.
        """
        r = self._fallback_torch_layer
        self._scale_torch = torch.full((1, 1, 1, r.top_k), r.routed_scaling_factor, dtype=torch.bfloat16)

    def move_weights_to_device_impl(self):
        """Move only the scale tensor to device."""
        self._scale_dev = ttnn.to_device(
            ttnn.from_torch(self._scale_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.device,
        )

    def forward(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass: softmax -> 3-pass centering topk -> gather -> normalize -> scale.

        Implements the actual Qwen3.5 routing algorithm:
            scores = softmax(logits, dim=-1)
            top_values, top_indices = topk(scores, k)
            weights = top_values / sum(top_values)
            weights *= routed_scaling_factor

        Uses the 3-pass centering technique from the parent class to work around
        BF16 precision limits in ttnn.topk over 256 experts. The centering shifts
        scores so the decision boundary sits near zero where BF16 has highest
        precision, enabling accurate top-k selection.
        """
        r = self._fallback_torch_layer

        # --- Prepare logits as float32 4D tensor ---
        if logits.layout != ttnn.TILE_LAYOUT:
            logits = ttnn.to_layout(logits, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.reshape(logits, ttnn.Shape((1, 1, logits.shape[0], logits.shape[1])))
        if logits.dtype != ttnn.float32:
            logits_f32 = ttnn.typecast(logits, ttnn.float32)
            ttnn.deallocate(logits)
        else:
            logits_f32 = logits

        # --- Softmax activation (Qwen uses softmax, NOT sigmoid) ---
        scores_f32 = ttnn.softmax(logits_f32, dim=-1)

        T = scores_f32.shape[2]
        top_k = r.top_k

        # --- 3-pass centering topk ---
        # With 256 experts, softmax scores are clustered near ~1/256 = 0.0039.
        # BF16 has only ~3 decimal digits of precision, so a naive BF16 topk
        # cannot reliably distinguish the top-8 from the rest. The centering
        # technique subtracts a coarse threshold each pass, moving the decision
        # boundary toward zero where BF16 resolution is highest.

        # Pass 1: rough BF16 topk(k+1) to find coarse threshold
        scores_bf16_p1 = ttnn.typecast(scores_f32, ttnn.bfloat16)
        rough_vals, _ = ttnn.topk(scores_bf16_p1, k=top_k + 1, dim=3, largest=True, sorted=True)
        ttnn.deallocate(scores_bf16_p1)
        # (k+1)-th value gives coarse threshold
        rough_thr_bf16 = ttnn.slice(rough_vals, [0, 0, 0, top_k], [1, 1, T, top_k + 1])
        ttnn.deallocate(rough_vals)
        rough_thr_f32 = ttnn.typecast(rough_thr_bf16, ttnn.float32)
        ttnn.deallocate(rough_thr_bf16)
        # Center scores around the decision boundary (float32 precision preserved)
        scores_c1 = ttnn.sub(scores_f32, rough_thr_f32)
        ttnn.deallocate(rough_thr_f32)

        # Pass 2: refined BF16 topk(k+1) on centered scores
        scores_bf16_p2 = ttnn.typecast(scores_c1, ttnn.bfloat16)
        refined_vals, _ = ttnn.topk(scores_bf16_p2, k=top_k + 1, dim=3, largest=True, sorted=True)
        ttnn.deallocate(scores_bf16_p2)
        # Second threshold is now near 0 -> BF16 step ~ 0.0001 (very precise)
        refined_thr_bf16 = ttnn.slice(refined_vals, [0, 0, 0, top_k], [1, 1, T, top_k + 1])
        ttnn.deallocate(refined_vals)
        refined_thr_f32 = ttnn.typecast(refined_thr_bf16, ttnn.float32)
        ttnn.deallocate(refined_thr_bf16)
        scores_c2 = ttnn.sub(scores_c1, refined_thr_f32)
        ttnn.deallocate(scores_c1)
        ttnn.deallocate(refined_thr_f32)

        # Final pass: exact topk(k) on doubly-centered scores
        scores_bf16_final = ttnn.typecast(scores_c2, ttnn.bfloat16)
        ttnn.deallocate(scores_c2)
        _, topk_expert_idx = ttnn.topk(scores_bf16_final, k=top_k, dim=3, largest=True, sorted=True)
        ttnn.deallocate(scores_bf16_final)

        # --- Gather raw softmax scores for selected experts ---
        topk_weights = ttnn.gather(scores_f32, dim=3, index=topk_expert_idx)
        ttnn.deallocate(scores_f32)

        # --- Normalize weights by their sum ---
        denom = ttnn.sum(topk_weights, dim=3, keepdim=True)
        # Add epsilon to match PyTorch reference and prevent division by zero
        denom = ttnn.add(denom, 1e-20)
        topk_weights = ttnn.div(topk_weights, denom)
        ttnn.deallocate(denom)

        # --- Apply routing scale ---
        scale_rep_rm = ttnn.repeat(self._scale_dev, ttnn.Shape((1, 1, T, 1)))
        scale_bf16 = ttnn.to_layout(scale_rep_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if scale_bf16.dtype != ttnn.float32:
            scale_f32 = ttnn.typecast(scale_bf16, ttnn.float32)
            ttnn.deallocate(scale_bf16)
        else:
            scale_f32 = scale_bf16
        topk_weights = ttnn.mul(topk_weights, scale_f32)
        ttnn.deallocate(scale_f32)

        # --- Reshape outputs to (T, top_k) ---
        topk_expert_idx = ttnn.reshape(topk_expert_idx, ttnn.Shape((T, r.top_k)))
        topk_weights = ttnn.reshape(topk_weights, ttnn.Shape((T, r.top_k)))
        return topk_expert_idx, topk_weights


class TTNNQwenExperts(TTNNExperts):
    """Qwen-specific experts using sparse_matmul with fused w1/w3 projections.

    This subclass overrides preprocess_weights_impl() to pre-reshape weights to 4D
    and forward() to use sparse_matmul with fused gate/up projections. This approach
    eliminates duplicate memory bandwidth by reading the input tensor once instead
    of twice.

    Inheritance:
        - __init__(): Inherited (unchanged)
        - _get_num_experts_per_device(): Inherited (unchanged)
        - from_torch(): Inherited (unchanged)
        - preprocess_weights_impl(): OVERRIDDEN - creates fused w1_w3 weights, shards on dim=1
        - move_weights_to_device_impl(): OVERRIDDEN - simplified (no reshape needed)
        - forward(): OVERRIDDEN - uses fused sparse_matmul
    """

    def preprocess_weights_impl(self):
        """Preprocess expert weights: reshape to 4D on host, convert to bfloat16, TILE_LAYOUT.

        Creates fused w1_w3 weights for single sparse_matmul, eliminating duplicate memory
        bandwidth by reading the input tensor once instead of twice.
        Shape: (num_experts, H, I) -> (1, num_experts, H, 2*I) for fused w1_w3
        """
        # Reshape to 4D on host (torch) before converting to ttnn
        torch_w1_4d = self.torch_w1_proj.unsqueeze(0).to(torch.bfloat16)
        torch_w3_4d = self.torch_w3_proj.unsqueeze(0).to(torch.bfloat16)
        torch_w2_4d = self.torch_w2_proj.unsqueeze(0).to(torch.bfloat16)

        # Create fused w1_w3 weights for single sparse_matmul
        torch_w1_w3_fused = torch.cat([torch_w1_4d, torch_w3_4d], dim=-1)
        self.tt_w1_w3_proj = ttnn.from_torch(
            torch_w1_w3_fused,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1),
        )
        del torch_w1_w3_fused

        # w2 for down projection
        self.tt_w2_proj = ttnn.from_torch(
            torch_w2_4d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1),
        )

        del self.torch_w1_proj
        del self.torch_w3_proj
        del self.torch_w2_proj

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device - simplified since weights are already 4D.

        Moves fused w1_w3 weights and w2 weights to device, and creates program configs
        for sparse_matmul operations.
        """
        self.num_experts_per_device = self._get_num_experts_per_device(self.config, self.device)
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.get_num_devices()

        # Move fused w1_w3 weights to device
        self.tt_w1_w3_proj = ttnn.to_device(self.tt_w1_w3_proj, self.device)

        # Move w2 weights to device
        self.tt_w2_proj = ttnn.to_device(self.tt_w2_proj, self.device)

        # Create expert mapping tensors for all-to-all ops
        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(self.num_devices, dtype=torch.int32)
            .repeat_interleave(self.num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Create remap topk mask for expert token remap
        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, self.num_dispatch_devices, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Program configs for sparse_matmul operations
        hidden_tiles = self.hidden_size // ttnn.TILE_SIZE
        intermediate_tiles = self.intermediate_size // ttnn.TILE_SIZE

        # Fused gate/up program config (output is 2*intermediate_size)
        self._fused_gate_up_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.intermediate_size * 2),  # 2*I for fused output
            in0_block_w=min(4, hidden_tiles),
            per_core_M=1,
        )

        # Down projection program config
        self._down_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.hidden_size),
            in0_block_w=min(4, intermediate_tiles),
            per_core_M=1,
        )
        self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(
        self, x: ttnn.Tensor, topk_experts_indices: ttnn.Tensor, topk_experts_weights: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Execute expert pipeline using fused sparse_matmul.

        Uses sparse_matmul with fused W1/W3 weights to compute only activated experts.
        This eliminates duplicate memory bandwidth by reading the input tensor once.

        Args:
            x: Input tensor of shape (batch_size_per_device, 1, seq_len, hidden_size)
            topk_experts_indices: Expert indices of shape (batch_size_per_device*seq_len, num_experts_per_tok)
            topk_experts_weights: Expert weights of shape (batch_size_per_device*seq_len, num_experts_per_tok)

        Returns:
            Output tensor of shape (1, 1, batch_size_per_device*seq_len, hidden_size)
        """
        # Extract dimensions
        batch_size_per_device = x.shape[0]
        seq_len = x.shape[2]
        batch_size = batch_size_per_device * self.num_dispatch_devices

        # Decode mode detection for L1 memory optimization
        is_decode_mode = seq_len == 1
        decode_memory_config = ttnn.L1_MEMORY_CONFIG if is_decode_mode else ttnn.DRAM_MEMORY_CONFIG
        tokens_per_device = batch_size_per_device * seq_len

        # Store original num_tokens for unpadding later
        original_num_tokens = tokens_per_device

        if topk_experts_indices.dtype != ttnn.uint16:
            if topk_experts_indices.layout != ttnn.TILE_LAYOUT:
                topk_experts_indices = ttnn.to_layout(
                    topk_experts_indices,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            topk_experts_indices = ttnn.typecast(topk_experts_indices, ttnn.uint16)

        # Padding: Use SPARSITY_BLOCK_SIZE for sparse matmul
        pad_block_size = SPARSITY_BLOCK_SIZE

        total_tokens = tokens_per_device * self.num_dispatch_devices
        pad_amount = 0

        if total_tokens % pad_block_size != 0:
            padded_tokens_per_device = (
                ((total_tokens + pad_block_size - 1) // pad_block_size) * pad_block_size // self.num_dispatch_devices
            )
            if padded_tokens_per_device * self.num_dispatch_devices < total_tokens:
                padded_tokens_per_device += 1
            pad_amount = padded_tokens_per_device - tokens_per_device

            if pad_amount > 0:
                # Pad x: (batch, 1, seq_len, hidden) -> (batch, 1, padded_seq_len, hidden)
                x = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)

                # Pad indices: (tokens, k) -> (padded_tokens, k) with value 0 (route to expert 0)
                topk_experts_indices = ttnn.pad(topk_experts_indices, padding=((0, pad_amount), (0, 0)), value=0)

                # Pad weights: (tokens, k) -> (padded_tokens, k) with value 0.0 (zero weight = no contribution)
                topk_experts_weights = ttnn.pad(topk_experts_weights, padding=((0, pad_amount), (0, 0)), value=0.0)

                # Update tokens_per_device and seq_len for padded tensors
                tokens_per_device = padded_tokens_per_device
                seq_len = tokens_per_device // batch_size_per_device
                total_tokens = tokens_per_device * self.num_dispatch_devices

        x = ttnn.typecast(x, ttnn.bfloat16)

        # STEP 1: PREPARE INPUTS FOR ALL_TO_ALL_DISPATCH
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, self.hidden_size))

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, self.num_experts_per_tok)
        )

        # STEP 2: ALL_TO_ALL_DISPATCH - Route tokens to expert devices
        all_to_all_dispatch_output, all_to_all_dispatch_metadata = ttnn.all_to_all_dispatch(
            x_rm,
            topk_experts_indices_rm,
            self.expert_mapping_tensors,
            cluster_axis=1,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(x_rm)
        ttnn.deallocate(topk_experts_indices_rm)

        # STEP 3: PREPARE DISPATCH OUTPUT FOR EXPERT COMPUTATION
        post_dispatch = ttnn.reshape(all_to_all_dispatch_output, shape=(1, 1, total_tokens, self.hidden_size))
        post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)

        # STEP 4: EXPERT COMPUTATION using fused sparse_matmul
        # NO ttnn.repeat - this is the key optimization!
        # Generate sparsity tensor to only compute activated experts

        num_tokens = total_tokens

        # Generate sparsity tensor
        remap_topk_mask_expanded = ttnn.repeat(self.remap_topk_mask, ttnn.Shape((1, batch_size_per_device, 1, 1)))
        _, sparsity_t = ttnn.moe_expert_token_remap(
            remap_topk_mask_expanded,
            self.expert_mapping_tensors,
            all_to_all_dispatch_metadata,
            reduction_size=SPARSITY_BLOCK_SIZE,
        )

        num_sparse_blocks = num_tokens // SPARSITY_BLOCK_SIZE
        x_sparse = ttnn.reshape(post_dispatch, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, self.hidden_size))

        # Fused gate/up projection - single sparse_matmul for both w1 (gate) and w3 (up)
        # Output shape: [1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, 2*intermediate_size]
        w1_w3_out = ttnn.sparse_matmul(
            x_sparse,
            self.tt_w1_w3_proj,  # Fused weights: [1, E, H, 2*I]
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=self._fused_gate_up_program_config,
            compute_kernel_config=self._expert_compute_cfg,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(x_sparse)

        # Split fused output into w1 (gate) and w3 (up) components
        # Shape: [1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, 2*I] -> two [1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, I]
        intermediate_size = self.intermediate_size

        # Get actual tensor shape to build correct slice indices
        actual_shape = list(w1_w3_out.shape)
        rank = len(actual_shape)

        # Build slice indices dynamically - only slice the last dimension
        # First half: w1 (gate projection)
        slice_start_w1 = [0] * rank
        slice_end_w1 = list(actual_shape)
        slice_end_w1[-1] = intermediate_size

        w1_out = ttnn.slice(
            w1_w3_out,
            slice_start=slice_start_w1,
            slice_end=slice_end_w1,
        )

        # Second half: w3 (up projection)
        slice_start_w3 = [0] * rank
        slice_start_w3[-1] = intermediate_size
        slice_end_w3 = list(actual_shape)

        w3_out = ttnn.slice(
            w1_w3_out,
            slice_start=slice_start_w3,
            slice_end=slice_end_w3,
        )
        ttnn.deallocate(w1_w3_out)

        # SwiGLU activation: silu(gate) * up
        w1_activated = ttnn.silu(w1_out, memory_config=decode_memory_config)
        ttnn.deallocate(w1_out)
        intermediate = ttnn.mul(w1_activated, w3_out, memory_config=decode_memory_config)
        ttnn.deallocate(w1_activated)
        ttnn.deallocate(w3_out)

        intermediate = ttnn.squeeze(intermediate, 0)
        intermediate = ttnn.squeeze(intermediate, 1)

        # Down projection (w2) with sparse_matmul
        expert_output = ttnn.sparse_matmul(
            intermediate,
            self.tt_w2_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=self._down_program_config,
            compute_kernel_config=self._expert_compute_cfg,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(intermediate)

        # Reshape to expected format
        expert_output = ttnn.permute(expert_output, (1, 0, 2, 3))
        expert_output = ttnn.reshape(
            expert_output, shape=(1, self.num_experts_per_device, num_tokens, self.hidden_size)
        )

        ttnn.deallocate(post_dispatch)

        # STEP 5: PREPARE EXPERT OUTPUT FOR ALL_TO_ALL_COMBINE
        expert_output = ttnn.reshape(
            expert_output,
            shape=(self.num_experts_per_device, 1, total_tokens, self.hidden_size),
        )
        expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape to match combine expected format
        expert_output = ttnn.reshape(
            expert_output, shape=(self.num_experts_per_device, batch_size, seq_len, self.hidden_size)
        )

        # STEP 6: ALL_TO_ALL_COMBINE - Route expert outputs back to token positions
        combined_output = ttnn.all_to_all_combine(
            expert_output,
            all_to_all_dispatch_metadata,
            self.expert_mapping_tensors,
            cluster_axis=1,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(expert_output)
        ttnn.deallocate(all_to_all_dispatch_metadata)

        # STEP 7: APPLY ROUTING WEIGHTS AND REDUCE ACROSS EXPERTS
        actual_shape = list(combined_output.shape)
        if len(actual_shape) == 5:
            combined_output = ttnn.reshape(combined_output, shape=(self.num_experts_per_tok, 1, -1, self.hidden_size))
        else:
            combined_output = ttnn.reshape(
                combined_output, shape=(self.num_experts_per_tok, 1, tokens_per_device, self.hidden_size)
            )
        combined_output = ttnn.to_layout(combined_output, ttnn.TILE_LAYOUT)

        # Prepare routing weights for broadcasting
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        # topk_experts_weights shape: [tokens, K] -> transpose to [K, tokens]
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (1, 0))
        # Now [K, tokens] -> [K, 1, tokens, 1] for broadcasting
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 1)
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 3)
        topk_experts_weights_tile = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)

        # Broadcast multiply: [K, 1, tokens, 1] * [K, 1, tokens, hidden] -> [K, 1, tokens, hidden]
        weighted_output = ttnn.mul(
            combined_output,
            topk_experts_weights_tile,
        )
        ttnn.deallocate(combined_output)
        ttnn.deallocate(topk_experts_weights_tile)

        # Sum over experts dimension
        final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        ttnn.deallocate(weighted_output)

        # UNPAD: Remove padding added at the start to restore original token count
        if original_num_tokens != tokens_per_device:
            final_output = ttnn.slice(
                final_output,
                slice_start=[0, 0, 0, 0],
                slice_end=[1, 1, original_num_tokens, self.hidden_size],
                slice_step=[1, 1, 1, 1],
            )

        return final_output


class TTNNQwen3MoE(TTNNMoE):
    """TTNN MoE for Qwen3.5-35B-A3B architecture with 256 experts, top-8 routing.

    Handles the Qwen3_5MoeSparseMoeBlock structure:
    - gate: Qwen3_5MoeTopKRouter
    - experts: Qwen3_5MoeExperts (with gate_up_proj and down_proj)
    - shared_expert: Qwen3_5MoeMLP (singular, not plural like GLM)
    - shared_expert_gate: Optional gating for shared expert output

    Inheritance:
        - __init__(): Inherited (unchanged)
        - from_torch(): OVERRIDDEN - handles Qwen-specific structure
        - preprocess_weights_impl(): OVERRIDDEN - adds shared_expert_gate
        - move_weights_to_device_impl(): OVERRIDDEN - adds shared_expert_gate
        - forward(): OVERRIDDEN - adds shared_expert_gate support
        - _adapt_config(): NEW static method
        - _consolidate_experts(): NEW static method
        - _adapt_gate(): NEW static method
    """

    @classmethod
    def from_torch(cls, torch_moe):
        """Create TTNNQwen3MoE from PyTorch Qwen3_5MoeSparseMoeBlock module.

        KEY DIFFERENCES from parent:
        1. Gets config from torch_moe.experts.config (not torch_moe.config)
        2. Uses TTNNQwenMoERouterDecode instead of TTNNMoERouterDecode
        3. Uses TTNNQwenExperts instead of TTNNExperts
        4. Accesses shared_expert (singular) instead of shared_experts (plural)
        5. Handles optional shared_expert_gate
        """
        # 1. Adapt config to match expected structure
        adapted_config = cls._adapt_config(torch_moe)

        # 2. Consolidate experts from Qwen3 format
        consolidated_experts = cls._consolidate_experts(torch_moe.experts, adapted_config)

        # 3. Adapt gate to match expected structure
        adapted_gate = cls._adapt_gate(torch_moe.gate)

        # 4. Create module instance
        module = cls(adapted_config)
        module._fallback_torch_layer = torch_moe

        # 5. Initialize submodules using parent's pattern
        module.gate = TTNNGlm4MoeTopkRouter.from_parameters(adapted_gate.weight, adapted_gate.e_score_correction_bias)

        # KEY DIFFERENCE: Use Qwen-specific router with softmax activation
        module.route_tokens_to_experts = TTNNQwenMoERouterDecode.from_torch(
            Glm4MoeRouteTokenToExperts(
                adapted_gate.e_score_correction_bias,
                adapted_config.n_routed_experts,
                adapted_config.n_group,
                adapted_config.topk_group,
                adapted_config.num_experts_per_tok,
                True,  # norm_topk_prob (Qwen3 uses normalized probabilities)
                adapted_config.routed_scaling_factor,
            )
        )

        # KEY DIFFERENCE: Use Qwen-specific experts with batched matmul
        # Check if CPU experts fallback is enabled for debugging
        use_cpu_experts = os.environ.get("TT_QWEN_CPU_EXPERTS", "0").lower() in ("1", "true", "yes")
        if use_cpu_experts:
            # Keep original PyTorch experts for CPU execution (for debugging accuracy issues)
            module.experts = torch_moe.experts
            module._use_cpu_experts = True
            print("[DEBUG] TT_QWEN_CPU_EXPERTS=1: Using CPU fallback for experts")
        else:
            module.experts = TTNNQwenExperts.from_torch(consolidated_experts)
            module._use_cpu_experts = False

        # KEY DIFFERENCE: Qwen3 uses singular "shared_expert" not "shared_experts"
        module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_moe.shared_expert)

        # Store replicated gate weight for preprocessing
        module._gate_weight_torch = adapted_gate.weight.to(torch.bfloat16)

        # KEY DIFFERENCE: Store shared_expert_gate weight for gating the shared expert output
        if hasattr(torch_moe, "shared_expert_gate"):
            module._shared_expert_gate_weight_torch = torch_moe.shared_expert_gate.weight.to(torch.bfloat16)
        else:
            module._shared_expert_gate_weight_torch = None

        return module

    @staticmethod
    def _adapt_config(torch_moe):
        """Adapt Qwen3 MoE config to match Glm4MoeConfig structure.

        KEY DIFFERENCES:
        1. Gets config from torch_moe.experts.config (not torch_moe.config)
        2. num_experts -> n_routed_experts
        3. Provides default n_group=4, topk_group=2 (Qwen3 doesn't have these)
        """
        original_config = torch_moe.experts.config

        class AdaptedConfig:
            pass

        config = AdaptedConfig()

        # Map Qwen3 attributes to Glm4MoeConfig naming
        config.hidden_size = original_config.hidden_size
        config.moe_intermediate_size = original_config.moe_intermediate_size
        config.num_experts_per_tok = original_config.num_experts_per_tok

        # Key difference: num_experts -> n_routed_experts
        config.n_routed_experts = original_config.num_experts

        # Qwen3 doesn't use group-based routing at all - simple softmax -> topk.
        # Set n_group=1, topk_group=1 so that group logic is effectively a no-op
        # (the single group always contains all experts).
        config.n_group = 1
        config.topk_group = 1

        # Scaling factor - use 1.0 if not specified
        config.routed_scaling_factor = getattr(original_config, "routed_scaling_factor", 1.0)

        # Additional attributes needed by TTNNExperts
        config.hidden_act = getattr(original_config, "hidden_act", "silu")

        return config

    @staticmethod
    def _consolidate_experts(qwen_experts, config):
        """Adapt Qwen3_5MoeExperts to the structure expected by TTNNExperts.from_torch().

        Qwen3 experts already have the right shape:
        - gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
        - down_proj: [num_experts, hidden_size, intermediate_size]

        This method wraps them in an object with the expected attributes.
        """

        class ConsolidatedExperts:
            pass

        consolidated = ConsolidatedExperts()

        # Qwen3 gate_up_proj already has the right shape
        consolidated.gate_up_proj = qwen_experts.gate_up_proj
        consolidated.down_proj = qwen_experts.down_proj
        consolidated.config = config

        return consolidated

    @staticmethod
    def _adapt_gate(qwen_gate):
        """Adapt Qwen3_5MoeTopKRouter to match expected structure with e_score_correction_bias.

        KEY DIFFERENCE: Qwen3 router may not have e_score_correction_bias, so we create zeros.
        """

        class AdaptedGate:
            pass

        adapted = AdaptedGate()
        adapted.weight = qwen_gate.weight

        # Qwen3 router doesn't have e_score_correction_bias - create zeros
        if hasattr(qwen_gate, "e_score_correction_bias"):
            adapted.e_score_correction_bias = qwen_gate.e_score_correction_bias
        else:
            # Create zeros tensor with shape [num_experts]
            adapted.e_score_correction_bias = torch.zeros(qwen_gate.weight.shape[0])

        return adapted

    def preprocess_weights_impl(self):
        """Preprocess weights including shared_expert_gate.

        Extends parent to also preprocess shared_expert_gate weight if present.
        """
        # Call parent preprocess for gate weight and submodules
        super().preprocess_weights_impl()

        # KEY DIFFERENCE: Preprocess shared_expert_gate weight if present
        if self._shared_expert_gate_weight_torch is not None:
            # Shape: [1, hidden_size] -> transpose to [hidden_size, 1] for linear
            self._shared_expert_gate_tt_host = ttnn.from_torch(
                self._shared_expert_gate_weight_torch.T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        """Move weights to device including shared_expert_gate.

        Extends parent to also move shared_expert_gate weight to device.
        """
        # Call parent move_weights_to_device
        super().move_weights_to_device_impl()

        # KEY DIFFERENCE: Move shared_expert_gate weight to device with replication
        if self._shared_expert_gate_weight_torch is not None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            gate_torch = ttnn.to_torch(self._shared_expert_gate_tt_host)
            self._shared_expert_gate_tt = ttnn.from_torch(
                gate_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def set_output_tensors_config_impl(self, output_tensors):
        """Set output tensor config for col-sharded output.

        The reduce_scatter output is col-sharded (each device has [batch, seq, hidden_size/8]).
        We need to use ConcatMeshToTensor on dim=-1 to concatenate the shards.
        """

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self.device is not None and self.device.get_num_devices() > 1:
                    # Use ConcatMeshToTensor on dim=-1 only (not batch dim)
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        """Compute logical shape by multiplying last dim by num_devices."""
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    config = DistributedTensorConfig(
                        mesh_mapper=mesh_mapper,
                        mesh_composer=mesh_composer,
                        logical_shape_fn=logical_shape_for_col_sharded,
                    )
                    e.set_distributed_tensor_config(config)
            return e

        # Only set col-sharded config if in distributed mode
        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        return tree_map(set_col_sharded_config, output_tensors)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with shared_expert_gate support.

        KEY DIFFERENCE: Applies sigmoid gating to shared expert output:
            shared_output = sigmoid(linear(x, gate_weight)) * shared_expert(x)
        """
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.get_num_devices()
        self.num_experts_per_device = even_int_div(self.config.n_routed_experts, self.num_devices)
        # Store original input for shared experts
        residual = x

        # 1. All-gather to revert tensor parallelism
        x = ttnn.all_gather(
            x,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # 2. MoE gate routing
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x.dtype != ttnn.float32:
            x_f32 = ttnn.typecast(x, ttnn.float32)
        else:
            x_f32 = x
        router_logits_f32 = ttnn.linear(
            x_f32,
            self._gate_weight_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        if x_f32 is not x:
            ttnn.deallocate(x_f32)
        # Convert back to bfloat16 for router (ttnn.softmax / ttnn.topk require bf16)
        router_logits = ttnn.typecast(router_logits_f32, ttnn.bfloat16)
        ttnn.deallocate(router_logits_f32)

        T = router_logits.shape[-2]
        router_logits = ttnn.reshape(router_logits, ttnn.Shape((T, self.n_routed_experts)))

        topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)

        x = ttnn.unsqueeze(x, 1)  # Add experts dimension for compatibility with experts module

        # 3. Experts handle dispatch -> compute -> combine -> weight
        if getattr(self, "_use_cpu_experts", False):
            # CPU EXPERTS FALLBACK: Convert to PyTorch, run experts, convert back
            # This is for debugging - bypasses TTNN expert computation
            x_cpu = ttnn.squeeze(x, 1)  # Remove experts dimension for CPU experts
            x_cpu = ttnn.to_layout(x_cpu, ttnn.ROW_MAJOR_LAYOUT)

            # IMPORTANT: After all_gather on hidden dim, all devices have IDENTICAL data.
            # ConcatMeshToTensor concatenates all device copies, creating num_devices x copies.
            # We only need ONE copy, so slice to take only the first device's portion.
            num_devices = self.device.get_num_devices()

            x_torch_full = ttnn.to_torch(x_cpu, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            x_batch_per_device = x_torch_full.shape[0] // num_devices
            x_torch = x_torch_full[:x_batch_per_device]  # Take only first device's data
            x_torch = x_torch.view(-1, x_torch.shape[-1])  # Flatten to (tokens, hidden)

            # Convert indices and weights to PyTorch
            # Extract underlying TTNN tensor from TorchTTNNTensor wrapper
            idx_ttnn = topk_experts_indices
            idx_rm = ttnn.to_layout(idx_ttnn, ttnn.ROW_MAJOR_LAYOUT)
            idx_torch_full = ttnn.to_torch(idx_rm, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            idx_batch_per_device = idx_torch_full.shape[0] // num_devices
            idx_torch = idx_torch_full[:idx_batch_per_device]  # Take only first device's data
            idx_torch = idx_torch.to(torch.int64)

            wgt_ttnn = topk_experts_weights
            wgt_rm = ttnn.to_layout(wgt_ttnn, ttnn.ROW_MAJOR_LAYOUT)
            wgt_torch_full = ttnn.to_torch(wgt_rm, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            wgt_batch_per_device = wgt_torch_full.shape[0] // num_devices
            wgt_torch = wgt_torch_full[:wgt_batch_per_device]  # Take only first device's data

            # Call PyTorch experts
            routed_torch = self.experts(x_torch, idx_torch, wgt_torch)
            routed_torch = routed_torch.view(1, 1, -1, routed_torch.shape[-1])

            # Convert back to TTNN (replicate across devices since we'll reduce-scatter)
            routed_output = ttnn.from_torch(
                routed_torch.to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)

        # 4. Reduce-scatter final output.
        n_rs = self.device.get_num_devices()  # devices along cluster_axis=1
        # Extract underlying TTNN tensor - handle both wrapped and raw tensors
        routed_out = routed_output
        if n_rs > 1:
            routed_out = ttnn.mul(routed_out, 1.0 / float(n_rs))
        routed_output = ttnn.reduce_scatter(
            routed_out,
            dim=3,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 5. Add shared experts output with gating
        shared_output = self.shared_experts(residual)

        # KEY DIFFERENCE: Apply shared expert gate if present
        if self._shared_expert_gate_weight_torch is not None:
            # Compute gate values: sigmoid(linear(x_gathered, gate_weight))
            x_for_gate = ttnn.squeeze(x, 1)  # Remove experts dimension added earlier
            if x_for_gate.layout != ttnn.TILE_LAYOUT:
                x_for_gate = ttnn.to_layout(x_for_gate, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate_logits = ttnn.linear(
                x_for_gate,
                self._shared_expert_gate_tt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate_values = ttnn.sigmoid(gate_logits)
            ttnn.deallocate(gate_logits)
            # Gate the shared expert output - broadcast gate_values (shape [..., 1]) to shared_output shape
            shared_output_gated = ttnn.mul(shared_output, gate_values)
            ttnn.deallocate(gate_values)
            output = ttnn.add(routed_output, shared_output_gated)
            ttnn.deallocate(shared_output_gated)
        else:
            output = ttnn.add(routed_output, shared_output)

        output = ttnn.squeeze(output, 1)  # Remove experts dimension

        return output
