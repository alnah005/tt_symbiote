# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN wrappers for the Gemma-4 vision tower (Phase 8 Wave A).

Phase-8 scope: same high-value-first strategy as the text companion
(see :mod:`.modeling_gemma4_text`). The vision tower's heavy compute
(attention + MLP + LN) is the matmul inside each
:class:`Gemma4ClippableLinear`; we wrap the gated MLP whose three
``ClippableLinear`` children feed it. Patch embed, 2-D RoPE, attention
itself, encoder layers, pooler, and the top-level ``Gemma4VisionModel``
stay declared ``cpu_fallback`` — they need bespoke 2-D RoPE plumbing
and the existing single-device attention modules don't fit Gemma-4's
non-causal vision SDPA shape.

What this file ships
--------------------

* :class:`TTNNGemma4VisionMLP` ← ``Gemma4VisionMLP``. Same structure
  as :class:`TTNNGemma4TextMLP`, but its gate/up/down children are
  ``Gemma4ClippableLinear`` instances (a thin wrapper around
  ``nn.Linear``). The wrapper detects ``use_clipped_linears`` and:

    * if clipping is disabled (Gemma-4 E2B vision config default),
      reaches inside each ``ClippableLinear`` and plucks the
      ``nn.Linear`` for direct :class:`TTNNLinear` wrapping;
    * if clipping is enabled (audio-only path; not exercised by the
      Phase 7/8 image-text demos), declines and lets the recipe leave
      the MLP on host. This keeps clamp correctness while still
      allowing the image path to accelerate.

* :class:`TTNNGemma4MultimodalEmbedder` ← ``Gemma4MultimodalEmbedder``.
  Two ops: RMSNorm (``with_scale=False``) followed by an unbiased
  linear from vision-hidden -> text-hidden. The norm is implemented
  inline (no scale, eps from the config) and the linear becomes a
  :class:`TTNNLinear`.

What's deferred
---------------

* ``Gemma4VisionAttention`` — 2-D rotary embedding (height + width
  inv_freq concat), non-causal SDPA, position-aware projections.
* ``Gemma4VisionRotaryEmbedding`` — 2-D position encoding precompute.
* ``Gemma4VisionPatchEmbedder``, ``Gemma4VisionEncoderLayer``,
  ``Gemma4VisionEncoder``, ``Gemma4VisionPooler``, ``Gemma4VisionModel``.
  Pooler in particular has position-dependent average-pool behaviour
  that needs custom TTNN plumbing.
"""

from __future__ import annotations

import ttnn
from torch import nn

from tt_symbiote.core.module import TTNNModule
from tt_symbiote.integrations.ttnn_linear import TTNNLinear

__all__ = [
    "TTNNGemma4MultimodalEmbedder",
    "TTNNGemma4VisionMLP",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inner_linear(clippable) -> nn.Linear:
    """Pull the underlying ``nn.Linear`` out of a ``Gemma4ClippableLinear``.

    Falls back to returning the input unchanged if it already *is* an
    ``nn.Linear`` — keeps the wrapper robust if HF ever inlines the
    helper.
    """
    if isinstance(clippable, nn.Linear):
        return clippable
    inner = getattr(clippable, "linear", None)
    if isinstance(inner, nn.Linear):
        return inner
    raise TypeError(f"_inner_linear expected Gemma4ClippableLinear or nn.Linear, got {type(clippable).__name__}")


def _clipping_active(clippable) -> bool:
    """True iff the ClippableLinear is configured to clamp at runtime."""
    return bool(getattr(clippable, "use_clipped_linears", False))


# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------


class TTNNGemma4VisionMLP(TTNNModule):
    """``Gemma4VisionMLP`` -> on-device gated GELU MLP (image-path).

    HF shape:

        down_proj(act_fn(gate_proj(x)) * up_proj(x))

    where each ``*_proj`` is a :class:`Gemma4ClippableLinear`. When
    clipping is disabled (E2B vision config default) we wrap the
    inner ``nn.Linear`` of each ``ClippableLinear`` as
    :class:`TTNNLinear`. If clipping is enabled (audio-only path; not
    exercised by the dog-photo demo) we raise so the recipe leaves
    the MLP on host — clamp correctness matters more than throughput
    for the corner case.
    """

    @classmethod
    def from_torch(cls, mlp):
        # If any ClippableLinear is in clip-active mode, decline the
        # swap. Returning the original module is the established
        # convention in TTNN integrations (see
        # ``TTNNLayerNorm.from_torch`` -> "Using standard LayerNorm").
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if _clipping_active(proj):
                return mlp

        new = cls()
        new._fallback_torch_layer = mlp
        new.gate_proj = TTNNLinear.from_torch(_inner_linear(mlp.gate_proj))
        new.up_proj = TTNNLinear.from_torch(_inner_linear(mlp.up_proj))
        new.down_proj = TTNNLinear.from_torch(_inner_linear(mlp.down_proj))
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
        gate = ttnn.gelu(gate)
        up = self.up_proj(x)
        intermediate = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return self.down_proj(intermediate)


# ---------------------------------------------------------------------------
# Multimodal embedder (vision-hidden -> text-hidden bridge)
# ---------------------------------------------------------------------------
#
# HF ``Gemma4MultimodalEmbedder`` is:
#
#     embedding_pre_projection_norm: Gemma4RMSNorm(..., with_scale=False)
#     embedding_projection:          nn.Linear(vision_hidden, text_hidden, bias=False)
#
# The RMSNorm here has *no* learnable weight (with_scale=False), so we
# implement the norm inline rather than rely on
# :class:`TTNNLocalRMSNorm` (whose ``_infer_dim`` would need a
# ``_norm_dim`` stash to handle the weightless case). The linear maps
# straight to :class:`TTNNLinear`.


class TTNNGemma4MultimodalEmbedder(TTNNModule):
    """``Gemma4MultimodalEmbedder`` -> on-device projection (norm stays on host).

    HF layout::

        embedding_pre_projection_norm: Gemma4RMSNorm(with_scale=False)
        embedding_projection:          nn.Linear(vision_hidden, text_hidden)

    The weightless RMSNorm is kept on host (the HF module is preserved
    via ``_fallback_torch_layer``) -- ``ttnn.rms_norm`` requires a
    weight tensor today, and shipping a synthetic all-ones buffer just
    for this single call site adds more friction than it saves. The
    heavy lifting is the projection (vision-hidden -> text-hidden),
    which moves to :class:`TTNNLinear`.
    """

    @classmethod
    def from_torch(cls, embedder):
        new = cls()
        new._fallback_torch_layer = embedder
        new.embedding_projection = TTNNLinear.from_torch(embedder.embedding_projection)
        return new

    def preprocess_weights_impl(self):
        self.embedding_projection.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.embedding_projection.move_weights_to_device()
        super().move_weights_to_device_impl()

    def forward(self, inputs_embeds):
        # Norm on host (weightless RMSNorm; preserves bit-exact behaviour
        # with the upstream HF reference). Projection on device.
        normed = self.torch_layer.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)
