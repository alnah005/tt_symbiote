# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN suffix embedding for the pi0.5 action expert.

Ported into the :class:`~tt_symbiote.core.module.TTNNModule` lifecycle from the
tt-metal reference ``models/experimental/pi0_5/tt/ttnn_suffix.py``
(``Pi0_5SuffixEmbeddingTTNN``) and validated against the PyTorch golden
``reference/torch_suffix.py`` (``Pi0_5SuffixEmbedding``), branch ``tt/pi0.5_bh``.

The suffix embeds the noisy action chunk and the flow-matching timestep for the
pi0.5 action expert. Unlike pi0, pi0.5 has **no state token** and uses adaRMS
time conditioning (no fused action-time MLP):

* ``embed_actions``: ``action_in_proj`` projects ``(B, action_horizon, action_dim)``
  -> ``(B, action_horizon, expert_width)``.
* ``embed_adarms_cond``: ``sincos(timestep) -> time_mlp_in -> silu -> time_mlp_out
  -> silu`` producing ``(B, expert_width)``. The trailing SiLU is critical -- it
  matches the openpi/lerobot pi05 reference; without it the scale/shift/gate
  modulations downstream flip sign of model outputs.
* ``project_output``: ``action_out_proj`` projects expert output back to
  ``(B, action_horizon, action_dim)``.

The reference ``Pi0_5SuffixEmbedding`` is a *plain Python object* holding raw
weight tensors (``action_in_proj.{weight,bias}``, ``action_out_proj.{weight,bias}``,
``time_mlp_in.{weight,bias}``, ``time_mlp_out.{weight,bias}``); ``from_torch``
plucks them directly. ``ttnn.linear`` computes ``x @ w`` (w is ``[in, out]``); the
torch weights are ``[out, in]`` so they are transposed on host before upload.

VALIDATION STATUS: the action projections mirror the already-validated gemma4
linear pattern (low-risk). The sincos timestep embedding + double-SiLU adaRMS
conditioning must be PCC-checked against the torch reference (Tier 1/2 of the
pcc-test-gen stage).
"""

from __future__ import annotations

from typing import Optional, Tuple

import ttnn

from tt_symbiote.core.module import DeviceArch, StatelessTTNNModule, run_on_devices
from tt_symbiote.models.pi05.modeling_pi05_common import create_sinusoidal_pos_embedding
from tt_symbiote.models.pi05.modeling_pi05_gemma import _linear_weight_to_tt

from .configuration_pi05 import SuffixConfig

# Runtime tt-metal commit the model executes against (installed/built ttnn, main).
# Reference patterns ported from branch tt/pi0.5_bh @ b0703a56465989da179480c93a8992ec519e1cde.
TT_METAL_COMMIT = "b2af0cd67b4e92dafeb2d0254e1c5b43c3ec5a25"

__all__ = ["TTNNPi05SuffixEmbedding"]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _bias_to_tt(b: Optional["ttnn.Tensor"]) -> Optional[ttnn.Tensor]:
    """Reshape a torch ``[out]`` bias to ttnn ``[1, out]`` bf16 TILE host tensor."""
    if b is None:
        return None
    return ttnn.from_torch(b.reshape(1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


class TTNNPi05SuffixEmbedding(StatelessTTNNModule):
    """pi0.5 suffix embedding (action + timestep) for the action expert.

    Reference: ``reference/torch_suffix.py::Pi0_5SuffixEmbedding`` and
    ``tt/ttnn_suffix.py::Pi0_5SuffixEmbeddingTTNN``.

    Multiple entry points (this module is not a single-forward leaf):
      * ``embed_actions(noisy_actions)`` -> ``(B, action_horizon, expert_width)``
      * ``embed_adarms_cond(timestep)`` -> ``(B, expert_width)``
      * ``project_output(expert_output)`` -> ``(B, action_horizon, action_dim)``
      * ``embed_suffix(noisy_actions, timestep)`` -> ``(suffix_embs, adarms_cond)``
        convenience combining the first two. ``forward`` delegates to it.
    """

    @classmethod
    def from_torch(cls, suffix, config: SuffixConfig) -> "TTNNPi05SuffixEmbedding":
        assert config.pi05, "TTNNPi05SuffixEmbedding requires config.pi05=True"
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = suffix
        new._config = config
        # Plain-object reference holds raw weight tensors as attributes.
        new._action_in_w = suffix.action_in_weight
        new._action_in_b = suffix.action_in_bias
        new._action_out_w = suffix.action_out_weight
        new._action_out_b = suffix.action_out_bias
        new._time_mlp_in_w = suffix.time_mlp_in_weight
        new._time_mlp_in_b = suffix.time_mlp_in_bias
        new._time_mlp_out_w = suffix.time_mlp_out_weight
        new._time_mlp_out_b = suffix.time_mlp_out_bias
        new._expert_width = config.expert_width
        return new

    def preprocess_weights_impl(self):
        self.tt_action_in_w = _linear_weight_to_tt(self._action_in_w)
        self.tt_action_in_b = _bias_to_tt(self._action_in_b)
        self.tt_action_out_w = _linear_weight_to_tt(self._action_out_w)
        self.tt_action_out_b = _bias_to_tt(self._action_out_b)
        self.tt_time_mlp_in_w = _linear_weight_to_tt(self._time_mlp_in_w)
        self.tt_time_mlp_in_b = _bias_to_tt(self._time_mlp_in_b)
        self.tt_time_mlp_out_w = _linear_weight_to_tt(self._time_mlp_out_w)
        self.tt_time_mlp_out_b = _bias_to_tt(self._time_mlp_out_b)

    def move_weights_to_device_impl(self):
        self.tt_action_in_w = ttnn.to_device(self.tt_action_in_w, self.device, memory_config=_DRAM)
        self.tt_action_out_w = ttnn.to_device(self.tt_action_out_w, self.device, memory_config=_DRAM)
        self.tt_time_mlp_in_w = ttnn.to_device(self.tt_time_mlp_in_w, self.device, memory_config=_DRAM)
        self.tt_time_mlp_out_w = ttnn.to_device(self.tt_time_mlp_out_w, self.device, memory_config=_DRAM)
        if self.tt_action_in_b is not None:
            self.tt_action_in_b = ttnn.to_device(self.tt_action_in_b, self.device, memory_config=_DRAM)
        if self.tt_action_out_b is not None:
            self.tt_action_out_b = ttnn.to_device(self.tt_action_out_b, self.device, memory_config=_DRAM)
        if self.tt_time_mlp_in_b is not None:
            self.tt_time_mlp_in_b = ttnn.to_device(self.tt_time_mlp_in_b, self.device, memory_config=_DRAM)
        if self.tt_time_mlp_out_b is not None:
            self.tt_time_mlp_out_b = ttnn.to_device(self.tt_time_mlp_out_b, self.device, memory_config=_DRAM)

    @run_on_devices(DeviceArch.P150)
    def embed_actions(self, noisy_actions: ttnn.Tensor) -> ttnn.Tensor:
        """Project noisy actions: (B, action_horizon, action_dim) -> (B, action_horizon, expert_width)."""
        return ttnn.linear(
            noisy_actions,
            self.tt_action_in_w,
            bias=self.tt_action_in_b,
            memory_config=_L1,
        )

    @run_on_devices(DeviceArch.P150)
    def embed_adarms_cond(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """sincos(timestep) -> time_mlp_in -> silu -> time_mlp_out -> silu -> (B, expert_width).

        The trailing silu matches the openpi/lerobot pi05 reference; without it
        the downstream scale/shift/gate modulations flip sign of model outputs.

        ``create_sinusoidal_pos_embedding`` is a host-side helper that returns a
        device tensor, so calling it here keeps the compute path pure-TTNN.
        """
        sincos = create_sinusoidal_pos_embedding(
            timestep,
            self._expert_width,
            self.device,
            min_period=4e-3,
            max_period=4.0,
        )
        x = ttnn.linear(
            sincos,
            self.tt_time_mlp_in_w,
            bias=self.tt_time_mlp_in_b,
            memory_config=_L1,
        )
        ttnn.deallocate(sincos)
        x = ttnn.silu(x, memory_config=_L1)
        x = ttnn.linear(
            x,
            self.tt_time_mlp_out_w,
            bias=self.tt_time_mlp_out_b,
            memory_config=_L1,
        )
        return ttnn.silu(x, memory_config=_L1)

    @run_on_devices(DeviceArch.P150)
    def project_output(self, expert_output: ttnn.Tensor) -> ttnn.Tensor:
        """Project expert output back to action dim: (B, action_horizon, expert_width) -> (B, action_horizon, action_dim)."""
        return ttnn.linear(
            expert_output,
            self.tt_action_out_w,
            bias=self.tt_action_out_b,
            memory_config=_L1,
        )

    @run_on_devices(DeviceArch.P150)
    def embed_suffix(self, noisy_actions: ttnn.Tensor, timestep: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Convenience: returns (suffix_embs, adarms_cond) for pi0.5 (no state token)."""
        suffix_embs = self.embed_actions(noisy_actions)
        adarms_cond = self.embed_adarms_cond(timestep)
        return suffix_embs, adarms_cond

    @run_on_devices(DeviceArch.P150)
    def forward(self, noisy_actions: ttnn.Tensor, timestep: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        return self.embed_suffix(noisy_actions, timestep)
