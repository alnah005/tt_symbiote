# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt_symbiote port of baidu/Unlimited-OCR (DeepSeek-OCR-style VLM + MoE).

Layout:
  modeling_unlimited_ocr -- TTNN module tree (vision DeepEncoder, MLP projector,
    DeepSeek-V2 MoE decoder stack, for-causal-lm shell). SCAFFOLD: forwards stub.
  reference_loader       -- host/torch loader for the HF reference (compat shims,
    config backfill, rope standardize, eager attn, buffer rematerialization).
  recipe                 -- @register_recipe("UnlimitedOCRForCausalLM"), Pattern B.
"""

TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"

from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
    TTNNUnlimitedOcrClipAttention,
    TTNNUnlimitedOcrClipBlock,
    TTNNUnlimitedOcrClipEncoder,
    TTNNUnlimitedOcrDecoderLayer,
    TTNNUnlimitedOcrDeepEncoder,
    TTNNUnlimitedOcrDeepseekMLP,
    TTNNUnlimitedOcrDeepseekModel,
    TTNNUnlimitedOcrForCausalLM,
    TTNNUnlimitedOcrLlamaMHA,
    TTNNUnlimitedOcrMlpProjector,
    TTNNUnlimitedOcrMoE,
    TTNNUnlimitedOcrSamAttention,
    TTNNUnlimitedOcrSamBlock,
    TTNNUnlimitedOcrSamEncoder,
)
from tt_symbiote.models.unlimited_ocr.reference_loader import (
    load_reference_config,
    load_reference_model,
)

# Importing the recipe runs its @register_recipe("UnlimitedOCRForCausalLM")
# decorator, wiring tt_symbiote.AutoModelForCausalLM.
from tt_symbiote.models.unlimited_ocr.recipe import UnlimitedOCRRecipe

__all__ = [
    "TT_METAL_COMMIT",
    "TTNNUnlimitedOcrLlamaMHA",
    "TTNNUnlimitedOcrDeepseekMLP",
    "TTNNUnlimitedOcrMoE",
    "TTNNUnlimitedOcrMlpProjector",
    "TTNNUnlimitedOcrSamAttention",
    "TTNNUnlimitedOcrClipAttention",
    "TTNNUnlimitedOcrSamBlock",
    "TTNNUnlimitedOcrClipBlock",
    "TTNNUnlimitedOcrDecoderLayer",
    "TTNNUnlimitedOcrSamEncoder",
    "TTNNUnlimitedOcrClipEncoder",
    "TTNNUnlimitedOcrDeepEncoder",
    "TTNNUnlimitedOcrDeepseekModel",
    "TTNNUnlimitedOcrForCausalLM",
    "load_reference_config",
    "load_reference_model",
    "UnlimitedOCRRecipe",
]
