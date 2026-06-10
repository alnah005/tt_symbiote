# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN wrappers for the Qwen3-VL text decoder (Phase 8 Wave B).

Same high-value-first scope as the Gemma-4 wrappers under
:mod:`tt_symbiote.models.gemma4.modeling_gemma4_text`: wire the
structurally simple compute (RMSNorm, MLP) to existing TTNN
integrations and leave the bespoke parts (text attention with Q/K
head-norms + M-RoPE, DeepStack visual injection in
``Qwen3VLTextModel``) declared ``cpu_fallback``.

Module map
----------

* :class:`TTNNQwen3VLTextRMSNorm` ← ``Qwen3VLTextRMSNorm``. Standard
  RMSNorm with ``weight + variance_epsilon`` — direct ``ttnn.rms_norm``.

* :class:`TTNNQwen3VLTextMLP` ← ``Qwen3VLTextMLP``. SwiGLU:
  ``down_proj(silu(gate_proj(x)) * up_proj(x))``.
"""

from __future__ import annotations

import ttnn

from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices
from tt_symbiote.modules.ttnn_linear import TTNNLinear

__all__ = [
    "TTNNQwen3VLTextMLP",
    "TTNNQwen3VLTextRMSNorm",
]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TTNNQwen3VLTextRMSNorm(TTNNModule):
    """``Qwen3VLTextRMSNorm`` -> ``ttnn.rms_norm``.

    The HF layer is the standard
    ``weight * rsqrt(mean(x^2) + variance_epsilon)`` form, with a learnable
    scale of shape ``[hidden_size]``. The wrapper preprocesses the weight
    onto the device as ``[1, hidden_size]`` so ``ttnn.rms_norm`` broadcasts
    cleanly across whatever rank the input arrives at.
    """

    @classmethod
    def from_torch(cls, rms_norm):
        if not hasattr(rms_norm, "weight") or rms_norm.weight is None:
            return rms_norm
        new = cls()
        new._fallback_torch_layer = rms_norm
        new._eps = float(getattr(rms_norm, "variance_epsilon", getattr(rms_norm, "eps", 1e-6)))
        return new

    def preprocess_weights_impl(self):
        self.tt_weight = ttnn.from_torch(
            self.torch_layer.weight.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(
            self.tt_weight,
            self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self._eps)


# ---------------------------------------------------------------------------
# Text SwiGLU MLP
# ---------------------------------------------------------------------------
#
# HF ``Qwen3VLTextMLP``:
#
#     down_proj(act_fn(gate_proj(x)) * up_proj(x))
#
# where ``act_fn = ACT2FN[config.hidden_act]``. Qwen3-VL text config
# defaults to ``"silu"``. The three projections have no bias.


class TTNNQwen3VLTextMLP(TTNNModule):
    """``Qwen3VLTextMLP`` -> on-device SwiGLU.

    Internal layout:

    * ``gate_proj``, ``up_proj``: :class:`TTNNLinear` projecting
      hidden -> intermediate.
    * ``down_proj``: :class:`TTNNLinear` projecting intermediate ->
      hidden.

    Activation: ``ttnn.silu`` applied to ``gate_proj(x)`` before the
    elementwise multiply with ``up_proj(x)``.
    """

    @classmethod
    def from_torch(cls, mlp):
        new = cls()
        new._fallback_torch_layer = mlp
        # Child TTNN modules must be public-attribute named so the
        # set_device walker descends into them. Same convention as
        # Gemma-4 / ResNet wrappers.
        new.gate_proj = TTNNLinear.from_torch(mlp.gate_proj)
        new.up_proj = TTNNLinear.from_torch(mlp.up_proj)
        new.down_proj = TTNNLinear.from_torch(mlp.down_proj)
        return new

    def preprocess_weights_impl(self):
        self.gate_proj.preprocess_weights()
        self.up_proj.preprocess_weights()
        self.down_proj.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.gate_proj.move_weights_to_device()
        self.up_proj.move_weights_to_device()
        self.down_proj.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = self.gate_proj(x)
        gate = ttnn.silu(gate)
        up = self.up_proj(x)
        intermediate = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return self.down_proj(intermediate)
