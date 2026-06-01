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

import warnings

import torch

from tt_symbiote.auto.auto_mappings import register_recipe
from tt_symbiote.models.gemma4.configuration_gemma4 import lookup_ttnn_tuning
from tt_symbiote.models.gemma4.modeling_gemma4_text import (
    TTNNGemma4RMSNorm,
    TTNNGemma4ScaledWordEmbedding,
    TTNNGemma4TextMLP,
)
from tt_symbiote.models.gemma4.modeling_gemma4_vision import TTNNGemma4MultimodalEmbedder, TTNNGemma4VisionMLP

__all__ = ["Gemma4Recipe"]


# ---------------------------------------------------------------------------
# Mesh-replicated weight budget (Wave A)
# ---------------------------------------------------------------------------
#
# All Phase 8 Wave A wrappers (``TTNNLinear``, ``TTNNEmbedding``,
# ``TTNNGemma4RMSNorm``) take the default ``ttnn.to_device(tensor, mesh)``
# path which **replicates** the tensor across every chip of the mesh
# (no ``mesh_mapper`` passed in). The footprint per chip is therefore
# independent of mesh size: a 60-layer 5376/21504 MLP stack costs the
# same ~38 GB on a single-chip N150 as it does on an 8-chip T3K.
#
# Wormhole (N150/N300/T3K building block) chips expose ~10.5 GB of
# usable DRAM per chip after fabric reserve and the trace region. We
# budget 9 GB for replicated weights, leaving the remaining ~1.5 GB for
# activations, KV cache and intermediate tensors.
#
# Variants whose replicated footprint exceeds the budget are *not* swapped
# by ``build_module_dict``: the demo runs entirely on PyTorch/CPU and the
# compatibility report records 0 TTNN modules in ``runtime_observed``.
# This is the deliberate sequencing point for the Wave B follow-up,
# which is to plumb ``TTNNLinearIColShardedWRowSharded`` and friends
# through the Gemma-4 recipe so 31B / 26B-A4B can shard MLP weights
# along the intermediate dimension across the T3K mesh.

_TTNN_PER_CHIP_BUDGET_BYTES: int = 9 * 1024**3


def _ttnn_replicated_weight_footprint_bytes(model) -> int:
    """Estimate per-chip replicated TTNN weight bytes for the Wave A wrappers.

    Dominant terms (BF16, 2 bytes/elem):

    1. Text embedding ``vocab x hidden``.
    2. Optional Per-Layer-Embedding ``vocab x hidden_size_per_layer_input``
       (E2B/E4B only; the dense 31B has ``ple_dim == 0``).
    3. Three MLP linears per text layer: ``3 x hidden x intermediate``
       times ``num_hidden_layers``.

    RMSNorm scales, vision MLP linears and the multimodal embedder
    projection together contribute <1% even on the smallest variant and
    are folded into a 5% safety margin.

    Returns 0 for any model that doesn't expose a ``text_config``
    (e.g. callers that hand us a partially-loaded stub); the budget
    check then defaults to "fits" and ``build_module_dict`` returns the
    full TTNN swap dict.
    """
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None) if config is not None else None
    if text is None:
        return 0

    hidden = int(getattr(text, "hidden_size", 0) or 0)
    intermediate = int(getattr(text, "intermediate_size", 0) or 0)
    num_layers = int(getattr(text, "num_hidden_layers", 0) or 0)
    vocab = int(getattr(text, "vocab_size", 0) or 0)
    ple_dim = int(getattr(text, "hidden_size_per_layer_input", 0) or 0)

    embed_bytes = vocab * (hidden + ple_dim) * 2
    mlp_per_layer_bytes = 3 * hidden * intermediate * 2
    return int(1.05 * (embed_bytes + num_layers * mlp_per_layer_bytes))


def _ttnn_swap_is_safe(model) -> tuple[bool, str]:
    """Return ``(is_safe, reason)`` for the Wave A swap on this model.

    Two failure modes are gated here:

    1. **Per-chip DRAM budget**. Wave A wrappers replicate weights
       across the mesh; variants whose replicated footprint exceeds
       :data:`_TTNN_PER_CHIP_BUDGET_BYTES` partially fill the device
       and then OOM on the input/activation move. Empirically
       calibrated against Wave A: E2B = 2.8 GB ✓, E4B = 7.8 GB ✓,
       31B = 43 GB ✗ (oversubscribes per-chip DRAM by ~4x even on T3K).

    2. **MoE structural mismatch**. The 26B-A4B variant interleaves a
       dense ``Gemma4TextMLP`` (which Wave A wraps) with a
       ``Gemma4TextRouter`` + ``Gemma4TextExperts`` block (which Wave A
       leaves on CPU). The MoE branch carries extra
       ``q_norm`` / ``k_norm`` / ``pre_feedforward_layernorm_2`` /
       ``post_feedforward_layernorm_2`` siblings whose head-dim sizes
       (32-352) don't match the TTNN RMSNorm tile geometry, and the
       expert ``nn.Linear`` weights are shape-incompatible with the
       wrapped dense MLP path. Wrapping these triggers hundreds of
       per-token runtime fallbacks (validation failures inside
       ``ttnn.rms_norm`` / ``ttnn.linear``) and an "unexpected"
       ``Linear`` entry in the compatibility ledger.

    Both failure modes have the same proper fix (tensor-parallel
    sharding + a Gemma4TextExperts wrapper), so for now we route both
    through the same "skip the swap, run pure-PyTorch on CPU" path.
    """
    footprint = _ttnn_replicated_weight_footprint_bytes(model)
    if footprint > _TTNN_PER_CHIP_BUDGET_BYTES:
        gb = footprint / 1024**3
        return False, (
            f"replicated weight footprint ~{gb:.1f} GB exceeds the "
            f"{_TTNN_PER_CHIP_BUDGET_BYTES / 1024**3:.0f} GB per-chip "
            f"budget (tensor-parallel sharding not yet wired in)"
        )

    text = getattr(getattr(model, "config", None), "text_config", None)
    if text is not None and bool(getattr(text, "enable_moe_block", False)):
        return False, (
            "MoE variant: Gemma4TextExperts + bespoke head-dim norms "
            "fall outside the Wave A wrapper coverage (Gemma4TextMLP "
            "wraps only the dense branch); a partial swap produces "
            "hundreds of shape-validation fallbacks at runtime"
        )

    return True, ""


def _ttnn_swap_exceeds_chip_budget(model) -> bool:
    """Compat alias used by :meth:`Gemma4Recipe.post_register`."""
    return not _ttnn_swap_is_safe(model)[0]


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
    "Gemma4VisionAttention",  # Non-causal SDPA with 2-D RoPE — deferred.
    "Gemma4VisionEncoderLayer",
    "Gemma4VisionEncoder",
    "Gemma4VisionPooler",
    "Gemma4VisionModel",
    # ----- Text decoder (Gemma4TextModel) -----
    "Gemma4TextRotaryEmbedding",  # Dual rope tables per layer-type — deferred.
    "Gemma4TextAttention",  # KV-sharing + dual RoPE + per-head norms — deferred.
    # MoE pair only used by the 26B-A4B variant; harmless for dense models
    "Gemma4TextExperts",
    "Gemma4TextRouter",
    "Gemma4TextDecoderLayer",  # PLE residual + 4-norm sandwich — deferred.
    "Gemma4TextModel",  # PLE orchestration + dual mask — deferred.
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

        For Wave A we gate the swap on the per-chip replicated weight
        budget (see :func:`_ttnn_swap_exceeds_chip_budget`). Variants that
        fit (``E2B``, ``E4B``) get the full 5-class swap dict; variants
        that don't (``31B``, ``26B-A4B``) get an empty dict and a clear
        ``UserWarning`` so the demo runs end-to-end on PyTorch/CPU
        instead of OOMing mid-``set_device``. The compatibility report
        for the skipped variants still carries the design-time intent
        (``tt_implemented`` listing the 5 Wave A classes) while
        ``runtime_observed`` is empty — the canonical signal that the
        swap was budget-skipped and that tensor-parallel sharding is
        the next-wave priority.

        Imports the HF source classes lazily so this module is cheap to
        import even when ``transformers`` isn't installed (matches the
        ResNet recipe convention).
        """
        is_safe, reason = _ttnn_swap_is_safe(model)
        if not is_safe:
            warnings.warn(
                f"Gemma4Recipe: skipping TTNN swap for "
                f"{type(model).__name__} "
                f"('{getattr(model.config, '_name_or_path', '<unknown>')}'): "
                f"{reason}. Running on PyTorch/CPU.",
                stacklevel=2,
            )
            return {}

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
        they land. We also attach two read-only diagnostic fields:

        * ``ttnn_replicated_footprint_bytes``: estimate of the per-chip
          replicated weight bytes that the Wave A wrappers *would*
          allocate if all swaps applied. Useful when the e2e script
          wants to render a "what would TTNN cost?" annotation next to
          the compatibility report.
        * ``ttnn_swap_skipped``: ``True`` iff ``build_module_dict``
          returned ``{}`` because of the budget gate above. Lets the
          demo and the docs distinguish a deliberate budget skip from
          a forgotten recipe.
        """
        type(model).device = property(lambda self: torch.device("cpu"))
        runtime = lookup_ttnn_tuning(model)
        runtime["ttnn_replicated_footprint_bytes"] = _ttnn_replicated_weight_footprint_bytes(model)
        runtime["ttnn_swap_skipped"] = _ttnn_swap_exceeds_chip_budget(model)
        model._tt_runtime_config = runtime

    # ``make_kv_cache`` is intentionally not implemented. The
    # ``register_recipe`` decorator installs a no-op default which
    # leaves HF's ``DynamicCache`` in place — sufficient for the Phase 7/8
    # short-generation demos. A bespoke paged dual-cache lands with the
    # text-attention port in a follow-up commit.
