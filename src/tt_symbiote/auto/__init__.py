# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Public ``tt_symbiote.auto`` namespace.

Re-exports the 43 ``AutoModel*`` classes from :mod:`tt_symbiote.auto.modeling_auto`,
the processor / tokenizer / config Auto re-exports, and the recipe-registry
API. Mirrors the layout of ``transformers.models.auto`` so existing HF users
have no surprises.
"""

from tt_symbiote.auto.auto_mappings import (
    TT_MODEL_REGISTRY,
    Recipe,
    register_recipe,
)
from tt_symbiote.auto.configuration_auto import AutoConfig
from tt_symbiote.auto.feature_extraction_auto import AutoFeatureExtractor
from tt_symbiote.auto.image_processing_auto import AutoImageProcessor
from tt_symbiote.auto.modeling_auto import (
    AutoBackbone,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioTokenization,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForDepthEstimation,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForImageTextToText,
    AutoModelForImageToImage,
    AutoModelForInstanceSegmentation,
    AutoModelForKeypointDetection,
    AutoModelForKeypointMatching,
    AutoModelForMaskedImageModeling,
    AutoModelForMaskedLM,
    AutoModelForMaskGeneration,
    AutoModelForMultimodalLM,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForObjectDetection,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTableQuestionAnswering,
    AutoModelForTableRecognition,
    AutoModelForTDT,
    AutoModelForTextEncoding,
    AutoModelForTextRecognition,
    AutoModelForTextToSpectrogram,
    AutoModelForTextToWaveform,
    AutoModelForTimeSeriesPrediction,
    AutoModelForTokenClassification,
    AutoModelForUniversalSegmentation,
    AutoModelForVideoClassification,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelForZeroShotObjectDetection,
)
from tt_symbiote.auto.processing_auto import AutoProcessor
from tt_symbiote.auto.tokenization_auto import AutoTokenizer
from tt_symbiote.auto.video_processing_auto import AutoVideoProcessor

__all__ = [
    # 43 AutoModel* classes
    "AutoBackbone",
    "AutoModel",
    "AutoModelForAudioClassification",
    "AutoModelForAudioFrameClassification",
    "AutoModelForAudioTokenization",
    "AutoModelForAudioXVector",
    "AutoModelForCausalLM",
    "AutoModelForCTC",
    "AutoModelForDepthEstimation",
    "AutoModelForDocumentQuestionAnswering",
    "AutoModelForImageClassification",
    "AutoModelForImageSegmentation",
    "AutoModelForImageTextToText",
    "AutoModelForImageToImage",
    "AutoModelForInstanceSegmentation",
    "AutoModelForKeypointDetection",
    "AutoModelForKeypointMatching",
    "AutoModelForMaskedImageModeling",
    "AutoModelForMaskedLM",
    "AutoModelForMaskGeneration",
    "AutoModelForMultimodalLM",
    "AutoModelForMultipleChoice",
    "AutoModelForNextSentencePrediction",
    "AutoModelForObjectDetection",
    "AutoModelForPreTraining",
    "AutoModelForQuestionAnswering",
    "AutoModelForSemanticSegmentation",
    "AutoModelForSeq2SeqLM",
    "AutoModelForSequenceClassification",
    "AutoModelForSpeechSeq2Seq",
    "AutoModelForTableQuestionAnswering",
    "AutoModelForTableRecognition",
    "AutoModelForTDT",
    "AutoModelForTextEncoding",
    "AutoModelForTextRecognition",
    "AutoModelForTextToSpectrogram",
    "AutoModelForTextToWaveform",
    "AutoModelForTimeSeriesPrediction",
    "AutoModelForTokenClassification",
    "AutoModelForUniversalSegmentation",
    "AutoModelForVideoClassification",
    "AutoModelForVisualQuestionAnswering",
    "AutoModelForZeroShotImageClassification",
    "AutoModelForZeroShotObjectDetection",
    # processor / config Autos
    "AutoConfig",
    "AutoFeatureExtractor",
    "AutoImageProcessor",
    "AutoProcessor",
    "AutoTokenizer",
    "AutoVideoProcessor",
    # registry API
    "Recipe",
    "TT_MODEL_REGISTRY",
    "register_recipe",
]
