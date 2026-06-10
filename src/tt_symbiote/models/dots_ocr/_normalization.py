# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Normalization layer implementations for TTNN."""

import torch
import ttnn

from tt_symbiote.core.module import SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS, StatelessTTNNModule, run_on_devices
from tt_symbiote.core.run_config import trace_enabled


def _mesh_num_devices(device) -> int:
    if device is None or not hasattr(device, "get_num_devices"):
        return 1
    return int(device.get_num_devices())


@trace_enabled
class TTNNDistributedRMSNorm(StatelessTTNNModule):
    """
    Distributed RMSNorm implementation that performs the reduction across devices in the forward pass.

    """

    @classmethod
    def from_torch(cls, rms_norm: "RMSNorm"):
        """Create from PyTorch RMSNorm."""
        if not hasattr(rms_norm, "weight") or rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        dim = self.torch_layer.weight.shape[0]
        # Pad to nearest multiple of 32 for TILE compatibility
        padded_dim = ((dim + 31) // 32) * 32
        weight = self.torch_layer.weight
        if padded_dim != dim:
            weight = torch.nn.functional.pad(weight, (0, padded_dim - dim), value=1.0)
        self.weight_distributed = ttnn.as_tensor(
            weight.unsqueeze(0).view(1, 1, padded_dim).reshape([1, 1, padded_dim // 32, 32]).to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=(ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=list(self.device.shape))),
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
        # Compute kernel matches the proven tt_transformers/multimodal/llama_layernorm.py
        # pattern: HiFi4 (RMSNorm is sensitive — keep) but fp32_dest_acc_en=False to
        # double the dst register from 4 -> 8 tiles (~halves the kernel passes), and
        # packer_l1_acc=False to drop the L1 accumulator buffer. The variance reduction
        # itself runs in the kernel's internal FP32 path, so output accuracy is preserved.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # Single-device meshes cannot use fabric-backed all_gather in the distributed path.
        self.tt_weight_local = None
        if _mesh_num_devices(self.device) <= 1:
            self.tt_weight_local = ttnn.from_torch(
                weight.unsqueeze(0).expand(32, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self.tt_weight_local = ttnn.to_device(self.tt_weight_local, self.device)

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, inp):
        original_shape = inp.shape
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))

        if _mesh_num_devices(self.device) <= 1:
            if len(original_shape) == 3:
                inp = ttnn.unsqueeze(inp, 1)
            if inp.layout != ttnn.TILE_LAYOUT:
                inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tt_out = ttnn.rms_norm(
                inp,
                weight=self.tt_weight_local,
                epsilon=eps,
                compute_kernel_config=self.compute_kernel_config,
            )
            if len(original_shape) == 3 and len(tt_out.shape) == 4:
                tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
            return tt_out

        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if getattr(self.device, "shape", [1, 1])[-1] == 1:
            tt_out = ttnn.rms_norm(
                inp,
                weight=self.weight_distributed,
                epsilon=eps,
                compute_kernel_config=self.compute_kernel_config,
            )
            if len(original_shape) == 3 and len(tt_out.shape) == 4:
                tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
            return tt_out
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
        )
        # AllGather stats — use Ring topology for trace compatibility.
        # Linear topology may allocate dynamic intermediates not pinned by trace.
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Ring,
        )
        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=eps,
            weight=self.weight_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_stats.deallocate(True)

        # Squeeze back to original shape if we added a batch dimension
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])

        return tt_out
