# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma-4 VLM TTNN recipe — Phase 8 Wave A incremental port.

Targets :class:`transformers.Gemma4ForConditionalGeneration` (the
image-text-to-text head over the audio + vision + text backbones).

Phase-8 Wave A scope (high-value-first, N150)
---------------------------------------------

We move the *structurally simple* classes from ``cpu_fallback`` to
``tt_implemented`` by wrapping them onto existing TTNN integrations
(``TTNNLinear``, ``TTNNEmbedding``, ``TTNNLocalRMSNorm``). The
architecturally bespoke parts — text attention with KV-sharing, dual
RoPE tables, Per-Layer Embeddings (PLE), vision 2-D RoPE,
position-aware pooler — stay declared ``cpu_fallback`` (correct,
*expected* runtime behaviour; the runtime ledger only flags
undeclared CPU paths).

What moves to ``tt_implemented`` this commit:

* ``Gemma4RMSNorm`` -> :class:`TTNNLocalRMSNorm` (handles
  ``with_scale=True/False`` via existing logic; ``eps`` is read off
  the source layer).
* ``Gemma4TextScaledWordEmbedding`` ->
  :class:`TTNNGemma4ScaledWordEmbedding` (reuses
  :class:`TTNNEmbedding` with ``scale_factor=sqrt(embed_dim)``).
* ``Gemma4TextMLP`` -> :class:`TTNNGemma4TextMLP` (TTNNLinear x3 +
  ``ttnn.gelu`` + ``ttnn.multiply``).
* ``Gemma4VisionMLP`` -> :class:`TTNNGemma4VisionMLP` (same shape;
  reaches inside ``Gemma4ClippableLinear`` to pluck the inner
  ``nn.Linear``).
* ``Gemma4MultimodalEmbedder`` ->
  :class:`TTNNGemma4MultimodalEmbedder` (weightless RMSNorm +
  unbiased :class:`TTNNLinear`).

What stays ``cpu_fallback``:

* Text decoder: ``Gemma4TextAttention``, ``Gemma4TextRotaryEmbedding``,
  ``Gemma4TextDecoderLayer``, ``Gemma4TextModel``,
  ``Gemma4TextExperts``, ``Gemma4TextRouter``.
* Vision tower: every class except the MLP and the multimodal
  embedder above. The 2-D rotary embedding and the bespoke pooler
  block silent reuse of the existing single-device integrations.
* Shared: ``Gemma4ClippableLinear`` (when its parent MLP gets
  swapped, the inner ``nn.Linear`` is wrapped; otherwise it stays on
  host as a thin clamp orchestrator).

What's declared ``host_glue`` (added in Phase 8 alongside ``host_glue``
support in :mod:`tt_symbiote.utils.compatibility`):

* ``Gemma4Model``: the top-level fusion of vision + text. Owns
  ``masked_scatter`` to splice vision tokens into the text-embedding
  stream and the dual sliding/full attention mask construction.
  No compute beyond glue.
* ``Gemma4ForConditionalGeneration``: thin head over ``Gemma4Model``
  + ``lm_head`` + optional logit softcap.

A model with no registered recipe gets a placeholder shape so callers
that pipe :func:`compatibility.report` to JSON don't need a special
case.

Phase 7 -> Phase 8 migration notes
----------------------------------

Phase 7 shipped a *CPU-first* recipe (``build_module_dict`` returned
an empty dict). Phase 8 keeps that file shape and only adds the swaps
where existing TTNN integrations cleanly apply. The legacy 31B-dense
mechanical migration is still parked in
:file:`legacy_modeling_gemma4.py.bak`; nothing imports it.
"""

from __future__ import annotations

import torch

from tt_symbiote.auto.auto_mappings import register_recipe
from tt_symbiote.models.gemma4.configuration_gemma4 import lookup_ttnn_tuning
from tt_symbiote.models.gemma4.modeling_gemma4_text import (
    TTNNGemma4RMSNorm,
    TTNNGemma4ScaledWordEmbedding,
    TTNNGemma4TextMLP,
)
from tt_symbiote.models.gemma4.modeling_gemma4_vision import (
    TTNNGemma4MultimodalEmbedder,
    TTNNGemma4VisionMLP,
)

__all__ = ["Gemma4Recipe"]


# ---------------------------------------------------------------------------
# Design-time coverage manifests
# ---------------------------------------------------------------------------
#
# Source of truth for each list: the class definitions in
# ``transformers/src/transformers/models/gemma4/modeling_gemma4.py`` and
# its sibling ``configuration_gemma4.py``. Order within each list is
# top-down following the HF file (text -> vision -> multimodal -> top-level),
# so a diff against the upstream file is easy to read.
#
# Membership rules:
#   * ``tt_implemented``: this commit ships a TTNN wrapper that
#     ``build_module_dict`` swaps in.
#   * ``cpu_fallback``: HF class is *exercised* by the image-text-to-text
#     demo but stays as PyTorch in this commit. The runtime hook in
#     :mod:`tt_symbiote.core.run_config` will flag every entry here as
#     "expected" (not "unexpected") in :func:`compatibility.report`.
#   * ``host_glue``: HF class is intentionally host-only by policy
#     (orchestration, output dataclasses, mask building, scatter
#     fusion, index walks). Glue has no FLOPs to accelerate.
#   * ``out_of_scope``: HF class exists in the model file but is *not
#     touched* by the documented demos (audio tower, the text-only
#     ``Gemma4ForCausalLM`` head, output dataclasses).


_TT_IMPLEMENTED: list[str] = [
    "Gemma4RMSNorm",
    "Gemma4TextScaledWordEmbedding",
    "Gemma4TextMLP",
    "Gemma4VisionMLP",
    "Gemma4MultimodalEmbedder",
]


_CPU_FALLBACK: list[str] = [
    # ----- Shared building blocks (text + vision share these) -----
    "Gemma4ClippableLinear",  # Inner nn.Linear is on-device when its
                               # parent MLP gets swapped; otherwise this
                               # is a thin host clamp wrapper.
    # ----- Vision tower (Gemma4VisionModel) -----
    "Gemma4VisionPatchEmbedder",
    "Gemma4VisionRotaryEmbedding",  # 2-D RoPE precompute — deferred.
    "Gemma4VisionAttention",        # Non-causal SDPA with 2-D RoPE — deferred.
    "Gemma4VisionEncoderLayer",
    "Gemma4VisionEncoder",
    "Gemma4VisionPooler",
    "Gemma4VisionModel",
    # ----- Text decoder (Gemma4TextModel) -----
    "Gemma4TextRotaryEmbedding",   # Dual rope tables per layer-type — deferred.
    "Gemma4TextAttention",         # KV-sharing + dual RoPE + per-head norms — deferred.
    # MoE pair only used by the 26B-A4B variant; harmless for dense models
    "Gemma4TextExperts",
    "Gemma4TextRouter",
    "Gemma4TextDecoderLayer",      # PLE residual + 4-norm sandwich — deferred.
    "Gemma4TextModel",             # PLE orchestration + dual mask — deferred.
]


_HOST_GLUE: list[str] = [
    # ----- Top-level composites: orchestration only -----
    # ``Gemma4Model`` performs ``masked_scatter`` to splice vision tokens
    # into the text embedding stream and builds the sliding/full causal
    # masks. No FLOPs beyond glue once its children accelerate.
    "Gemma4Model",
    # ``Gemma4ForConditionalGeneration`` is the HF generation head; the
    # forward pass is delegate-to-Gemma4Model + lm_head + optional logit
    # softcap. The softcap is a single ``tanh`` on the logits — kept
    # host because it's the very last step before sampling, and the
    # rest of ``GenerationMixin`` (top-k, top-p, beam, etc.) is host.
    "Gemma4ForConditionalGeneration",
]


_OUT_OF_SCOPE: list[str] = [
    # ----- Output dataclasses (not torch modules) -----
    "Gemma4ModelOutputWithPast",
    "Gemma4CausalLMOutputWithPast",
    "Gemma4TextModelOutputWithPast",
    "Gemma4AudioModelOutput",
    # ----- Audio tower (Phase 7/8 demos are image+text only) -----
    "Gemma4AudioRelPositionalEncoding",
    "Gemma4AudioAttention",
    "Gemma4AudioSubSampleConvProjectionLayer",
    "Gemma4AudioSubSampleConvProjection",
    "Gemma4AudioFeedForward",
    "Gemma4AudioCausalConv1d",
    "Gemma4AudioLightConv1d",
    "Gemma4AudioLayer",
    "Gemma4AudioModel",
    # ----- Alternative top-level head not exercised in Phase 7/8 -----
    "Gemma4ForCausalLM",
]


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


@register_recipe(hf_class_name="Gemma4ForConditionalGeneration")
class Gemma4Recipe:
    """Incremental TTNN recipe for HuggingFace ``Gemma4ForConditionalGeneration``.

    Wraps the simple sub-classes onto existing TTNN integrations while
    keeping the architecturally bespoke pieces (attention, PLE, dual
    RoPE, 2-D vision RoPE) on host. Top-level orchestration (vision/text
    fusion, sliding/full mask build) is declared ``host_glue``.
    """

    tt_implemented: list[str] = _TT_IMPLEMENTED
    cpu_fallback: list[str] = _CPU_FALLBACK
    host_glue: list[str] = _HOST_GLUE
    out_of_scope: list[str] = _OUT_OF_SCOPE

    def build_module_dict(self, model):
        """Return the flat ``{torch_class: ttnn_class}`` replacement map.

        Imports the HF source classes lazily so this module is cheap to
        import even when ``transformers`` isn't installed (matches the
        ResNet recipe convention).
        """
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder,
            Gemma4RMSNorm,
            Gemma4TextMLP,
            Gemma4TextScaledWordEmbedding,
            Gemma4VisionMLP,
        )

        return {
            Gemma4RMSNorm: TTNNGemma4RMSNorm,
            Gemma4TextScaledWordEmbedding: TTNNGemma4ScaledWordEmbedding,
            Gemma4TextMLP: TTNNGemma4TextMLP,
            Gemma4VisionMLP: TTNNGemma4VisionMLP,
            Gemma4MultimodalEmbedder: TTNNGemma4MultimodalEmbedder,
        }

    def post_register(self, model):
        """Patch ``model.device`` to ``cpu`` and stash the TTNN tuning.

        ``device`` is normally a property defined on
        :class:`transformers.PreTrainedModel` that walks the parameters.
        Overriding it on the *class* (not the instance) keeps the
        ``@strict`` Gemma-4 config happy and matches the convention
        from :class:`BailingMoEV2Recipe` and :class:`ResNetRecipe`.

        ``_tt_runtime_config`` is consumed by the example scripts (mesh
        shape, ``l1_small_size``, etc.) and by downstream wrappers once
        they land.
        """
        type(model).device = property(lambda self: torch.device("cpu"))
        model._tt_runtime_config = lookup_ttnn_tuning(model)

    # ``make_kv_cache`` is intentionally not implemented. The
    # ``register_recipe`` decorator installs a no-op default which
    # leaves HF's ``DynamicCache`` in place — sufficient for the Phase 7/8
    # short-generation demos. A bespoke paged dual-cache lands with the
    # text-attention port in a follow-up commit.
