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

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tt_symbiote")
except PackageNotFoundError:  # editable install before pip resolves metadata
    __version__ = "0.0.0+unknown"

from tt_symbiote.models.auto import (
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
from tt_symbiote.utils import compatibility
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules

# HF-style side-effect import: pulling in :mod:`tt_symbiote.models` triggers
# the per-model ``@register_recipe`` decorators, populating
# :data:`tt_symbiote.models.auto.auto_mappings.TT_MODEL_REGISTRY` before the user
# can call :meth:`AutoModelForCausalLM.from_pretrained`. Guarded with
# try/except so a broken model file degrades to "no recipe" rather than
# blowing up the whole import.
try:
    from tt_symbiote import models as _models  # noqa: F401
except Exception as _e:  # pragma: no cover - exercised on broken installs
    import warnings as _warnings

    _warnings.warn(
        f"tt_symbiote: failed to load model recipes ({type(_e).__name__}: {_e}); "
        f"AutoModel*.from_pretrained will fall back to unmodified HF models.",
        stacklevel=2,
    )

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
    "compatibility",
    "register_modules",
    "register_recipe",
    "set_device",
]
