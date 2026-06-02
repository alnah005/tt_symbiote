# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN wrappers for the Qwen3-VL vision tower (Phase 8 Wave B).

Vision-side companion to :mod:`.modeling_qwen3_vl_text`. The complex
parts of the Qwen3-VL vision tower (Conv3d patch embed, M-RoPE 3-axis
precompute, varlen-packed SDPA) stay declared ``cpu_fallback``; what
this module does ship are the two pure feedforward / linear-only
blocks:

* :class:`TTNNQwen3VLVisionMLP` ← ``Qwen3VLVisionMLP``. A two-layer
  ``linear_fc1 -> act_fn -> linear_fc2`` block with biases. Activation
  is ``gelu_pytorch_tanh`` by default (Qwen3-VL vision config); the
  closest TTNN match is ``ttnn.gelu``.

* :class:`TTNNQwen3VLVisionPatchMerger` ←
  ``Qwen3VLVisionPatchMerger``. Composite ``LayerNorm -> reshape ->
  Linear -> GELU -> Linear`` block. The reshape is a host-side
  ``view`` so the wrapper keeps it on host; everything else moves
  on-device.
"""

from __future__ import annotations

import ttnn

from tt_symbiote.core.module import TTNNModule
from tt_symbiote.integrations.ttnn_linear import TTNNLinear

__all__ = [
    "TTNNQwen3VLVisionMLP",
    "TTNNQwen3VLVisionPatchMerger",
]


# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------
#
# HF ``Qwen3VLVisionMLP``:
#
#     hidden -> linear_fc1 (with bias) -> act_fn -> linear_fc2 (with bias)
#
# The activation defaults to ``gelu_pytorch_tanh`` which TTNN's
# ``ttnn.gelu`` matches (tanh approximation). The two linears each
# carry a bias term.


class TTNNQwen3VLVisionMLP(TTNNModule):
    """``Qwen3VLVisionMLP`` -> on-device two-layer FFN with GELU.

    Layout::

        x -> linear_fc1(bias=True) -> gelu -> linear_fc2(bias=True) -> y

    Both linears become :class:`TTNNLinear`; the activation is
    ``ttnn.gelu``.
    """

    @classmethod
    def from_torch(cls, mlp):
        new = cls()
        new._fallback_torch_layer = mlp
        new.linear_fc1 = TTNNLinear.from_torch(mlp.linear_fc1)
        new.linear_fc2 = TTNNLinear.from_torch(mlp.linear_fc2)
        return new

    def preprocess_weights_impl(self):
        self.linear_fc1.preprocess_weights()
        self.linear_fc2.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.linear_fc1.move_weights_to_device()
        self.linear_fc2.move_weights_to_device()
        super().move_weights_to_device_impl()

    def forward(self, hidden_state: ttnn.Tensor) -> ttnn.Tensor:
        hidden_state = self.linear_fc1(hidden_state)
        hidden_state = ttnn.gelu(hidden_state)
        return self.linear_fc2(hidden_state)


# ---------------------------------------------------------------------------
# Vision PatchMerger
# ---------------------------------------------------------------------------
#
# HF ``Qwen3VLVisionPatchMerger``:
#
#     norm: nn.LayerNorm(merged_or_hidden, eps=1e-6)
#     linear_fc1: nn.Linear(hidden * spatial_merge_size^2, hidden * spatial_merge_size^2)
#     linear_fc2: nn.Linear(hidden * spatial_merge_size^2, out_hidden_size)
#
# Forward:
#
#     x = norm(reshape(x, [-1, merged_dim])).view(-1, merged_dim)
#     x = linear_fc2(gelu(linear_fc1(x)))
#
# The reshape (``.view(-1, self.hidden_size)``) is a torch-only op and
# we keep it on host. Everything else runs on device.


class TTNNQwen3VLVisionPatchMerger(TTNNModule):
    """``Qwen3VLVisionPatchMerger`` -> 2-layer GELU block on device, LN on host.

    The LN at the front of the merger is left on host inside the
    preserved ``_fallback_torch_layer`` (same approach as Gemma-4's
    :class:`TTNNGemma4MultimodalEmbedder`): the existing single-device
    ``TTNNLayerNorm`` integration trips on the Qwen3-VL merged-dim
    shape and falls back at runtime, which is more friction than
    keeping the LN host-resident. The two FC projections (and the
    GELU between them) -- which are the actual compute -- run on
    device via :class:`TTNNLinear` + ``ttnn.gelu``.
    """

    @classmethod
    def from_torch(cls, merger):
        new = cls()
        new._fallback_torch_layer = merger
        new._use_postshuffle_norm = bool(getattr(merger, "use_postshuffle_norm", False))
        new._hidden_size = int(merger.hidden_size)
        new.linear_fc1 = TTNNLinear.from_torch(merger.linear_fc1)
        new.linear_fc2 = TTNNLinear.from_torch(merger.linear_fc2)
        return new

    def preprocess_weights_impl(self):
        self.linear_fc1.preprocess_weights()
        self.linear_fc2.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.linear_fc1.move_weights_to_device()
        self.linear_fc2.move_weights_to_device()
        super().move_weights_to_device_impl()

    def forward(self, x):
        # LN + the two view() reshapes stay in host torch -- HF's
        # forward is a one-liner once you've picked the pre/post-shuffle
        # branch, and the source layer already implements it.
        merger = self.torch_layer
        if not merger.use_postshuffle_norm:
            x = merger.norm(x).view(-1, self._hidden_size)
        else:
            x = merger.norm(x.view(-1, self._hidden_size)).view(-1, self._hidden_size)
        # Hand off to device for the actual compute.
        x = self.linear_fc1(x)
        x = ttnn.gelu(x)
        return self.linear_fc2(x)
