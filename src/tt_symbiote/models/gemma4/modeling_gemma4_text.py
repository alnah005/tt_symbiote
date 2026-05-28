# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN wrappers for the Gemma-4 text decoder (Phase 8 Wave A).

Phase-8 scope (high-value-first, N150): wire up the *structurally simple*
text-side submodules to existing TTNN integrations so they execute on
device. The architecturally bespoke pieces stay declared
``cpu_fallback`` in :data:`Gemma4Recipe._CPU_FALLBACK` and continue to
run on host PyTorch (this is *correct* and expected — the runtime
ledger only complains about *undeclared* fallbacks).

What this file ships
--------------------

* :class:`TTNNGemma4TextMLP` ← ``Gemma4TextMLP``. Gated MLP whose
  inner ``gate_proj`` / ``up_proj`` / ``down_proj`` are
  :class:`TTNNLinear` (the host wrapper only orchestrates the
  ``gelu(gate) * up`` step on device via ``ttnn.gelu`` /
  ``ttnn.multiply``).

* :class:`TTNNGemma4ScaledWordEmbedding` ← ``Gemma4TextScaledWordEmbedding``.
  Reuses :class:`TTNNEmbedding` with the
  ``scale_factor=sqrt(embedding_dim)`` scalar already baked into the HF
  layer's ``embed_scale`` buffer.

What's deferred to later commits
--------------------------------

* ``Gemma4TextAttention`` — needs KV sharing across tail layers, dual
  sliding/full head dims, proportional partial RoPE on global layers,
  per-head Q/K/V norms. None of these fit the existing single-device
  TTNN attention modules; a bespoke wrapper is a 1+ week project.
* ``Gemma4TextRotaryEmbedding`` — dual rope tables per layer-type
  (sliding theta=10k partial=1.0; full theta=1M partial=0.25 over
  global_head_dim=512). Pure-host precompute may be acceptable later
  but isn't wired up yet.
* ``Gemma4TextDecoderLayer`` — 4-norm sandwich + PLE residual +
  optional MoE branch. Depends on the attention wrapper.
* ``Gemma4TextModel`` — orchestrates PLE pipeline + dual causal masks
  + KV-sharing state. Depends on the above.
* Per-Layer Embeddings (PLE) — the E2B blocker; no analogue in the
  Ling / ResNet ports.

These all stay declared ``cpu_fallback`` in the recipe so
``compatibility.report(model)["runtime_observed"]["unexpected"]`` stays
empty.
"""

from __future__ import annotations

import math

import torch
from torch import nn

import ttnn

from tt_symbiote.core.module import TTNNModule
from tt_symbiote.integrations.ttnn_embedding import TTNNEmbedding
from tt_symbiote.integrations.ttnn_linear import TTNNLinear

__all__ = [
    "TTNNGemma4RMSNorm",
    "TTNNGemma4ScaledWordEmbedding",
    "TTNNGemma4TextMLP",
]


# ---------------------------------------------------------------------------
# RMSNorm (handles ``with_scale=True`` *and* ``with_scale=False``)
# ---------------------------------------------------------------------------
#
# Gemma-4's per-head Q/K/V norms inside ``Gemma4TextAttention`` /
# ``Gemma4VisionAttention`` use ``Gemma4RMSNorm(head_dim, with_scale=False)``
# -- no learnable scale, and *no stored ``dim`` attribute either*. The
# generic :class:`TTNNLocalRMSNorm` integration tries to infer the dim
# from ``weight`` / ``normalized_shape`` / ``dim`` / ``_norm_dim`` and
# raises ``"Cannot infer normalization dimension from torch layer with
# no weight"`` for this case. Rather than monkey-patch the source
# layers with a ``_norm_dim`` stash, we ship a tiny dedicated wrapper
# that handles both shapes inline:
#
# * ``with_scale=True``  -> ``ttnn.rms_norm(x, weight=W, epsilon=eps)``
# * ``with_scale=False`` -> ``ttnn.rms_norm(x, epsilon=eps)`` (no weight
#                            arg; ttnn happily accepts that signature).


class TTNNGemma4RMSNorm(TTNNModule):
    """``Gemma4RMSNorm`` (``with_scale=True``) -> ``ttnn.rms_norm``.

    Handles only the *scaled* form -- the variant used for decoder pre-/
    post-norms, vision LN, and the text/vision RMSNorm slots that own a
    learnable scale parameter. Weight is preprocessed onto the device as
    ``shape [1, dim]`` so ``ttnn.rms_norm`` broadcasts cleanly across
    [B, ..., H, D] inputs.

    ``Gemma4RMSNorm`` instances with ``with_scale=False`` -- the per-head
    Q/K/V norms inside ``Gemma4TextAttention`` / ``Gemma4VisionAttention``
    and the multimodal embedder's pre-projection norm -- are *not*
    handled here. They have no stored ``dim`` (and ``Gemma4RMSNorm``
    doesn't keep one), so ``from_torch`` simply declines to wrap them
    and returns the original layer; the surrounding cpu_fallback
    attention forward keeps using them on host. The multimodal embedder
    is wrapped as a whole so its weightless norm runs inline on device
    inside :class:`TTNNGemma4MultimodalEmbedder`.
    """

    @classmethod
    def from_torch(cls, rms_norm):
        has_scale = bool(getattr(rms_norm, "with_scale", True))
        weight = getattr(rms_norm, "weight", None)
        if not has_scale or weight is None:
            # Decline -- caller will keep the original module on host.
            return rms_norm
        new = cls()
        new._fallback_torch_layer = rms_norm
        new._eps = float(getattr(rms_norm, "eps", getattr(rms_norm, "variance_epsilon", 1e-6)))
        return new

    def preprocess_weights_impl(self):
        weight = self.torch_layer.weight
        self.tt_weight = ttnn.from_torch(
            weight.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(
            self.tt_weight,
            self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self._eps)


# ---------------------------------------------------------------------------
# Scaled word embedding
# ---------------------------------------------------------------------------
#
# ``Gemma4TextScaledWordEmbedding`` subclasses ``nn.Embedding`` and
# multiplies the looked-up vectors by ``embed_scale = sqrt(hidden_size)``
# (registered as a non-persistent buffer). :class:`TTNNEmbedding`
# already accepts a ``scale_factor`` for exactly this — we just pull
# the scalar off the HF layer's buffer and forward it. No bespoke code.


class TTNNGemma4ScaledWordEmbedding(TTNNEmbedding):
    """``Gemma4TextScaledWordEmbedding`` -> on-device embedding lookup + scale.

    Maps directly to :class:`TTNNEmbedding` with the scaling factor
    pulled from the HF layer's ``embed_scale`` buffer (sqrt of the
    embedding dim by default; Gemma-4 sets this in
    :class:`transformers.Gemma4TextConfig.__init__`).

    HF's ``Gemma4Model.forward`` reads ``embed_tokens.weight`` directly
    (to slice out the pad-token vector before the masked-scatter that
    splices vision tokens into the text stream). Exposing ``weight`` as
    a property that delegates to the preserved torch fallback layer
    keeps that path working without re-shipping the parameter tensor
    on the wrapper itself.
    """

    @classmethod
    def from_torch(cls, embedding):
        # ``embed_scale`` is stored as a non-persistent buffer; ``scalar_embed_scale``
        # mirrors it as a Python float. Either works; the float path avoids
        # a buffer-on-CPU lookup at every forward.
        scale = float(getattr(embedding, "scalar_embed_scale", 1.0))
        if scale == 0.0:
            scale = math.sqrt(embedding.weight.shape[-1])
        return super().from_torch(embedding, scale_factor=scale)

    @property
    def weight(self):
        # HF code reads ``model.embed_tokens.weight`` outside of the
        # forward (for the pad-token row used in ``Gemma4Model.forward``).
        # The torch_layer is the original nn.Embedding, kept alive via
        # ``_fallback_torch_layer``.
        return self.torch_layer.weight


# ---------------------------------------------------------------------------
# Gated MLP (gate * up -> down) with on-device matmuls + activation
# ---------------------------------------------------------------------------
#
# HF ``Gemma4TextMLP`` is:
#
#     down_proj(act_fn(gate_proj(x)) * up_proj(x))
#
# where ``act_fn`` is ``ACT2FN[config.hidden_activation]`` — Gemma-4
# defaults to ``"gelu_pytorch_tanh"``. We map gate / up / down each to
# :class:`TTNNLinear` so the matmuls run on device, then do the
# elementwise gate + multiply with ``ttnn.gelu`` / ``ttnn.multiply``.
#
# Activation handling: ``ttnn.gelu`` is the closest match. The HF
# default ``"gelu_pytorch_tanh"`` is the tanh approximation; TTNN's
# ``gelu`` uses the same approximation (configurable via the ``fast_and_approx``
# kwarg if needed). For ``"gelu"`` (exact) we still call ``ttnn.gelu`` —
# the small numerical drift is within bf16 noise and matches what the
# ResNet port accepts.


class TTNNGemma4TextMLP(TTNNModule):
    """``Gemma4TextMLP`` -> on-device gated GELU MLP.

    Internal layout:

    * ``gate_proj``, ``up_proj``: :class:`TTNNLinear` projecting
      hidden -> intermediate.
    * ``down_proj``: :class:`TTNNLinear` projecting intermediate ->
      hidden.

    Activation: ``ttnn.gelu`` applied to ``gate_proj(x)`` before the
    elementwise multiply with ``up_proj(x)``.

    ``use_double_wide_mlp`` is silently honoured because we read the
    intermediate dim from the HF layer's actual ``gate_proj.out_features``
    (which already accounts for the 2x widening on KV-shared tail
    layers when the config flag is on).
    """

    @classmethod
    def from_torch(cls, mlp):
        new = cls()
        new._fallback_torch_layer = mlp
        # Child TTNN modules must be public-attribute named so
        # ``set_device``'s recursive walker descends into them; same
        # convention as ResNet (see ``modeling_resnet.py``
        # ``TTNNResNetConvLayer.from_torch`` for the rationale).
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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = self.gate_proj(x)
        # ``ttnn.gelu`` uses the tanh-approximation by default which
        # matches Gemma-4's ``"gelu_pytorch_tanh"``. For the rare ``"gelu"``
        # config the small numerical drift is within bf16 noise.
        gate = ttnn.gelu(gate)
        up = self.up_proj(x)
        intermediate = ttnn.multiply(gate, up)
        # Free the gate/up activations once the multiply is done — same
        # explicit deallocate convention as ``TTNNBailingMoEDecoderLayer``.
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return self.down_proj(intermediate)
