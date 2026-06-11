# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma (HF ``DiffusionGemmaForBlockDiffusion``) TTNN bring-up.

Block-diffusion, non-autoregressive 26B sparse-MoE. Native to ``transformers``
5.11.0 (``model_type='diffusion_gemma'``). Manual integration path -- this is
NOT a standard ``*ForCausalLM``, so the Recipe/Auto-CausalLM path does not apply.

Current scope: scaffold + Tier1/Tier2 (RMSNorm, gelu-gated MLP, MoE router
implemented in pure TTNN; attention + sparse-MoE experts scaffolded pending the
post-Tier1/2 review). See ``modeling_diffusion_gemma`` for the module map.
"""

from tt_symbiote.models.diffusion_gemma.modeling_diffusion_gemma import (
    TTNNDiffusionGemmaDecoderTextAttention,
    TTNNDiffusionGemmaDecoderTextLayer,
    TTNNDiffusionGemmaEncoderTextAttention,
    TTNNDiffusionGemmaEncoderTextLayer,
    TTNNDiffusionGemmaDecoderTextModel,
    TTNNDiffusionGemmaEncoderTextModel,
    TTNNDiffusionGemmaLayerStack,
    TTNNDiffusionGemmaLMHead,
    TTNNDiffusionGemmaLMHead,
    TTNNDiffusionGemmaRMSNorm,
    TTNNDiffusionGemmaTextExperts,
    TTNNDiffusionGemmaTextMLP,
    TTNNDiffusionGemmaTextRouter,
    TTNNDiffusionGemmaTextScaledWordEmbedding,
    TTNNDiffusionGemmaSelfConditioning,
)

__all__ = [
    "TTNNDiffusionGemmaDecoderTextAttention",
    "TTNNDiffusionGemmaDecoderTextLayer",
    "TTNNDiffusionGemmaEncoderTextAttention",
    "TTNNDiffusionGemmaEncoderTextLayer",
    "TTNNDiffusionGemmaDecoderTextModel",
    "TTNNDiffusionGemmaEncoderTextModel",
    "TTNNDiffusionGemmaLMHead",
    "TTNNDiffusionGemmaRMSNorm",
    "TTNNDiffusionGemmaTextExperts",
    "TTNNDiffusionGemmaTextMLP",
    "TTNNDiffusionGemmaTextRouter",
    "TTNNDiffusionGemmaTextScaledWordEmbedding",
    "TTNNDiffusionGemmaSelfConditioning",
]

from tt_symbiote.models.diffusion_gemma.pipeline import (  # noqa: E402
    PipelineConfig,
    TTNNDiffusionGemmaPipeline,
)

# Importing the recipe runs its @register_recipe('DiffusionGemmaForBlockDiffusion') decorator.
from tt_symbiote.models.diffusion_gemma import recipe as _recipe  # noqa: E402,F401

__all__ += [
    "TTNNDiffusionGemmaLayerStack",
    "TTNNDiffusionGemmaLMHead",
    "PipelineConfig",
    "TTNNDiffusionGemmaPipeline",
]
