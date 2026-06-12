# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Embedding layer implementations for TTNN."""

import ttnn
from torch import nn

from tt_symbiote.core.module import DeviceArch, StatelessTTNNModule, run_on_devices
from tt_symbiote.core.run_config import DistributedTensorConfig, trace_enabled


@trace_enabled
class TTNNEmbedding(StatelessTTNNModule):
    """TTNN-accelerated embedding lookup for Ling/Bailing models.

    Replaces nn.Embedding (word_embeddings). Weight is replicated across all
    devices on a mesh — no CCL needed since downstream column-parallel linears
    expect replicated input.
    """

    @classmethod
    def from_torch(cls, embedding: nn.Embedding, scale_factor=None):
        new_layer = cls()
        new_layer._fallback_torch_layer = embedding
        new_layer._scale_factor = scale_factor
        return new_layer

    def call(self, *args, **kwargs):
        import torch

        from tt_symbiote.core.tensor import TorchTTNNTensor

        if (
            args
            and isinstance(args[0], (torch.Tensor, TorchTTNNTensor))
            and self.device is not None
            and hasattr(self.device, "get_num_devices")
            and self.device.get_num_devices() > 1
        ):
            input_ids = args[0]
            if not isinstance(input_ids, TorchTTNNTensor):
                input_ids = TorchTTNNTensor(input_ids)
            replicated_config = DistributedTensorConfig(
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
            )
            input_ids.set_distributed_tensor_config(replicated_config)
            args = (input_ids,) + args[1:]
        return super().call(*args, **kwargs)

    def preprocess_weights_impl(self):
        self.tt_weight_host = ttnn.from_torch(
            self.torch_layer.weight.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 1), mesh_shape=list(self.device.shape))
        self.tt_weight = ttnn.to_device(
            ttnn.from_torch(
                self.torch_layer.weight.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            ),
            self.device,
        )

    def deallocate_weights_impl(self):
        ttnn.deallocate(self.tt_weight)
        super().deallocate_weights_impl()

    @run_on_devices(DeviceArch.N300, DeviceArch.T3K, DeviceArch.P150x4)
    def forward(self, tt_indices):
        # Embedding op requires UINT32 input; tokenizer ids arrive as INT32.
        # Typecast requires last dim to be a multiple of 32 for row-major, so
        # pad -> typecast -> slice back if needed.
        if tt_indices.dtype != ttnn.uint32:
            seq_len = int(tt_indices.shape[-1])
            if seq_len % 32 == 0:
                tt_indices = ttnn.typecast(tt_indices, ttnn.uint32)
            else:
                pad_to = ((seq_len + 31) // 32) * 32
                tt_indices = ttnn.pad(
                    tt_indices,
                    padding=tuple(
                        (0, pad_to - seq_len if i == len(tt_indices.shape) - 1 else 0)
                        for i in range(len(tt_indices.shape))
                    ),
                    value=0,
                )
                tt_indices = ttnn.typecast(tt_indices, ttnn.uint32)
                starts = [0] * len(tt_indices.shape)
                ends = list(tt_indices.shape)
                ends[-1] = seq_len
                tt_indices = ttnn.slice(tt_indices, starts, ends)
        out = ttnn.embedding(
            tt_indices,
            self.tt_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self._scale_factor is not None:
            out = ttnn.multiply(out, self._scale_factor)
        return out
