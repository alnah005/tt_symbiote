# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""``tt_symbiote`` — TTNN-accelerated HuggingFace transformers.

Mirrors the user-facing surface of ``transformers`` for ``Auto*`` loading,
plus the two TTNN-specific public APIs:

- :func:`tt_symbiote.set_device` — mandatory device binding step that must
  run before any model invocation.
- :func:`tt_symbiote.register_modules` — utility for swapping PyTorch
  modules with their TTNN equivalents (the underlying mechanism that
  ``Auto*.from_pretrained`` uses when a recipe is registered).

Example::

    from tt_symbiote import AutoModelForCausalLM, set_device
    import ttnn

    model = AutoModelForCausalLM.from_pretrained(
        "inclusionAI/Ling-mini-2.0", trust_remote_code=True, dtype="auto"
    )
    set_device(model, ttnn.open_mesh_device(...))

    out = model.generate(**inputs, max_new_tokens=128)
"""

from tt_symbiote.auto import (
    AutoBackbone,
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
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
    AutoProcessor,
    AutoTokenizer,
    AutoVideoProcessor,
    Recipe,
    TT_MODEL_REGISTRY,
    register_recipe,
)
from tt_symbiote.core.run_config import DispatchManager, TracedRun
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules

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
    # TTNN-specific public API
    "DispatchManager",
    "Recipe",
    "TT_MODEL_REGISTRY",
    "TracedRun",
    "register_modules",
    "register_recipe",
    "set_device",
]
