# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt_symbiote port of the dots.ocr model (text decoder + vision tower + pipeline).

Vendored from ``models/experimental/tt_symbiote/`` in tt-metal at commit
``c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195``.

Layout:
  Foundational (leading underscore) -- shared sharded variants the dots.ocr
  layers depend on:
    _linear, _normalization, _rope, _attention, _embedding
  dots.ocr-specific layers:
    dots_ocr_attention, dots_ocr_mlp, dots_ocr_decoder_layer, dots_ocr_vision
  Orchestration:
    pipeline (TTNNDotsOCRPipeline + prefill/decode graphs + PipelineConfig +
              _create_paged_kv_cache), kv_cache (thin re-export shim)
"""

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"

from tt_symbiote.models.dots_ocr.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLayerStack,
)
from tt_symbiote.models.dots_ocr.dots_ocr_attention import (
    TTNNDotsOCRAttention,
    TTNNDotsOCRAttentionT3K,
)
from tt_symbiote.models.dots_ocr.dots_ocr_mlp import TTNNDotsOCRMLP
from tt_symbiote.models.dots_ocr.dots_ocr_vision import TTNNDotsOCRVisionTower
from tt_symbiote.models.dots_ocr._embedding import TTNNEmbedding
from tt_symbiote.models.dots_ocr.pipeline import (
    TTNNDotsOCRPipeline,
    PipelineConfig,
    _create_paged_kv_cache,
)

# Importing the recipe runs its @register_recipe("DotsOCRForCausalLM") decorator,
# wiring the canonical tt_symbiote.AutoModelForCausalLM + model.generate path.
from tt_symbiote.models.dots_ocr.recipe import DotsOCRRecipe

__all__ = [
    "TTNNDotsOCRDecoderLayer",
    "TTNNDotsOCRLayerStack",
    "TTNNDotsOCRAttention",
    "TTNNDotsOCRAttentionT3K",
    "TTNNDotsOCRMLP",
    "TTNNDotsOCRVisionTower",
    "TTNNEmbedding",
    "TTNNDotsOCRPipeline",
    "PipelineConfig",
    "_create_paged_kv_cache",
    "DotsOCRRecipe",
    "TT_METAL_COMMIT",
]
