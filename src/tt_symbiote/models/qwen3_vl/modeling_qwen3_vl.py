# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL TTNN recipe — Phase 8 Wave B incremental port.

Targets :class:`transformers.Qwen3VLForConditionalGeneration` (the
image-text-to-text head over the vision tower + text decoder).

Phase-8 Wave B scope (high-value-first, N150)
---------------------------------------------

Same shape as Gemma-4 Wave A: wrap the structurally simple compute
classes onto existing TTNN integrations, leave the bespoke parts
declared ``cpu_fallback``, and call the top-level fusion ``host_glue``.

What moves to ``tt_implemented`` this commit:

* ``Qwen3VLTextRMSNorm`` -> :class:`TTNNQwen3VLTextRMSNorm`
  (``ttnn.rms_norm`` with weight).
* ``Qwen3VLTextMLP`` -> :class:`TTNNQwen3VLTextMLP` (SwiGLU:
  ``TTNNLinear`` x3 + ``ttnn.silu`` + ``ttnn.multiply``).
* ``Qwen3VLVisionMLP`` -> :class:`TTNNQwen3VLVisionMLP` (two-layer
  ``TTNNLinear`` + ``ttnn.gelu``).
* ``Qwen3VLVisionPatchMerger`` -> :class:`TTNNQwen3VLVisionPatchMerger`
  (``TTNNLayerNorm`` + two ``TTNNLinear`` + ``ttnn.gelu``).

What stays ``cpu_fallback``:

* Text decoder: ``Qwen3VLTextAttention`` (Q/K head-norms + M-RoPE
  with 3-axis interleaving), ``Qwen3VLTextRotaryEmbedding`` (M-RoPE
  precompute), ``Qwen3VLTextDecoderLayer``, ``Qwen3VLTextModel``
  (DeepStack visual injection at sparse layers).
* Vision tower: ``Qwen3VLVisionPatchEmbed`` (Conv3d + bilinear
  positional embed), ``Qwen3VLVisionRotaryEmbedding`` (2-D RoPE
  precompute), ``Qwen3VLVisionAttention`` (varlen-packed SDPA),
  ``Qwen3VLVisionBlock``, ``Qwen3VLVisionModel``.

What's declared ``host_glue``:

* ``Qwen3VLPreTrainedModel``: pure ``PreTrainedModel`` inheritance
  scaffold; no compute.
* ``Qwen3VLModel``: top-level fusion. Owns the masked_scatter that
  injects vision tokens into the text embedding stream, DeepStack
  injection at specific layer indices, ``cu_seqlens`` + M-RoPE
  index building. No FLOPs beyond glue.
* ``Qwen3VLForConditionalGeneration``: HF generation head delegate.
"""

from __future__ import annotations

import torch

from tt_symbiote.auto.auto_mappings import register_recipe
from tt_symbiote.models.qwen3_vl.configuration_qwen3_vl import lookup_ttnn_tuning
from tt_symbiote.models.qwen3_vl.modeling_qwen3_vl_text import (
    TTNNQwen3VLTextMLP,
    TTNNQwen3VLTextRMSNorm,
)
from tt_symbiote.models.qwen3_vl.modeling_qwen3_vl_vision import (
    TTNNQwen3VLVisionMLP,
    TTNNQwen3VLVisionPatchMerger,
)

__all__ = ["Qwen3VLRecipe"]


# ---------------------------------------------------------------------------
# Design-time coverage manifests
# ---------------------------------------------------------------------------


_TT_IMPLEMENTED: list[str] = [
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLTextRMSNorm",
    "Qwen3VLTextMLP",
]


_CPU_FALLBACK: list[str] = [
    # ----- Vision tower (Qwen3VLVisionModel) -----
    "Qwen3VLVisionPatchEmbed",       # Conv3d + bilinear positional embed — deferred.
    "Qwen3VLVisionRotaryEmbedding",  # 2-D RoPE precompute — deferred.
    "Qwen3VLVisionAttention",        # varlen-packed SDPA — deferred.
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionModel",
    # ----- Text decoder (Qwen3VLTextModel) -----
    "Qwen3VLTextRotaryEmbedding",    # M-RoPE 3-axis precompute — deferred.
    "Qwen3VLTextAttention",          # Q/K head-norms + M-RoPE — deferred.
    "Qwen3VLTextDecoderLayer",       # depends on attention.
    "Qwen3VLTextModel",              # DeepStack injection — deferred.
]


_HOST_GLUE: list[str] = [
    # ----- Top-level composites: orchestration only -----
    "Qwen3VLPreTrainedModel",
    "Qwen3VLModel",
    "Qwen3VLForConditionalGeneration",
]


_OUT_OF_SCOPE: list[str] = [
    # ----- Output dataclasses (not torch modules) -----
    "BaseModelOutputWithDeepstackFeatures",
    "Qwen3VLModelOutputWithPast",
    "Qwen3VLCausalLMOutputWithPast",
]


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


@register_recipe(hf_class_name="Qwen3VLForConditionalGeneration")
class Qwen3VLRecipe:
    """Incremental TTNN recipe for HuggingFace ``Qwen3VLForConditionalGeneration``.

    Wraps the simple sub-classes onto existing TTNN integrations while
    keeping the architecturally bespoke pieces (M-RoPE, varlen vision
    SDPA, DeepStack injection) on host. Top-level orchestration
    (vision/text fusion, DeepStack sparse layer dispatch) is declared
    ``host_glue``.
    """

    tt_implemented: list[str] = _TT_IMPLEMENTED
    cpu_fallback: list[str] = _CPU_FALLBACK
    host_glue: list[str] = _HOST_GLUE
    out_of_scope: list[str] = _OUT_OF_SCOPE

    def build_module_dict(self, model):
        """Return the flat ``{torch_class: ttnn_class}`` replacement map."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextMLP,
            Qwen3VLTextRMSNorm,
            Qwen3VLVisionMLP,
            Qwen3VLVisionPatchMerger,
        )

        return {
            Qwen3VLTextRMSNorm: TTNNQwen3VLTextRMSNorm,
            Qwen3VLTextMLP: TTNNQwen3VLTextMLP,
            Qwen3VLVisionMLP: TTNNQwen3VLVisionMLP,
            Qwen3VLVisionPatchMerger: TTNNQwen3VLVisionPatchMerger,
        }

    def post_register(self, model):
        """Patch ``model.device`` to ``cpu`` and stash the TTNN tuning."""
        type(model).device = property(lambda self: torch.device("cpu"))
        model._tt_runtime_config = lookup_ttnn_tuning(model)

    # ``make_kv_cache`` is intentionally not implemented. The
    # ``register_recipe`` decorator installs a no-op default which
    # leaves HF's ``DynamicCache`` in place — sufficient for the
    # short-generation demos. A bespoke paged cache lands with the
    # text-attention port in a follow-up commit.
