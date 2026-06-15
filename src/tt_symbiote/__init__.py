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


def _ensure_ttnn_importable() -> None:
    """Wire a source-built ``ttnn`` onto ``sys.path`` before the eager import below.

    ``ttnn`` is intentionally NOT a PyPI dependency (see pyproject.toml): it is
    provided by a tt-metal SOURCE BUILD at ``$TT_METAL_HOME``. Importing
    ``tt_symbiote`` transitively does ``import ttnn`` at module-load time, so this
    runs first and:

    - no-ops if ``ttnn`` is already importable — a real build already on the path,
      or the ``sys.modules`` stub installed by ``tests/auto`` for software-only runs;
    - otherwise, if ``$TT_METAL_HOME`` is set, prepends its source tree
      (``$TT_METAL_HOME`` and ``$TT_METAL_HOME/ttnn``) to ``sys.path`` so the import
      resolves — the same wiring ``scripts/bootstrap_venv.sh`` writes as a ``.pth``;
    - otherwise raises a clear, actionable error.

    This can auto-WIRE an existing build; it cannot auto-PROVIDE ttnn — a built
    tt-metal checkout and ``$TT_METAL_HOME`` remain the user's responsibility.
    """
    import importlib
    import importlib.util
    import os
    import sys

    def _importable() -> bool:
        if "ttnn" in sys.modules:  # covers the tests/auto sys.modules stub
            return True
        try:
            return importlib.util.find_spec("ttnn") is not None
        except (ImportError, ValueError):
            return False

    if _importable():
        return

    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if tt_metal_home:
        for path in (tt_metal_home, os.path.join(tt_metal_home, "ttnn")):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        importlib.invalidate_caches()
        if _importable():
            return

    raise ImportError(
        "tt_symbiote requires `ttnn`, which is provided by a tt-metal SOURCE BUILD "
        "(not a PyPI wheel). Point $TT_METAL_HOME at your built tt-metal checkout so "
        "`ttnn` is importable:\n"
        "    export TT_METAL_HOME=/path/to/tt-metal\n"
        + (
            f"(current $TT_METAL_HOME={tt_metal_home!r} has no importable ttnn under it)"
            if tt_metal_home
            else "($TT_METAL_HOME is not set)"
        )
        + "\nSee the Installation section of the README, or run scripts/bootstrap_venv.sh."
    )


_ensure_ttnn_importable()

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
from tt_symbiote.core.weight_cache import clear_weight_cache
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
    "clear_weight_cache",
    "compatibility",
    "register_modules",
    "register_recipe",
    "set_device",
]
