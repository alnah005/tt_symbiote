# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Every ``Auto*`` class and core public symbol is importable from ``tt_symbiote``."""

import pytest

# The hand-listed 43 AutoModel* classes per ``docs/internal/PROJECT_PROPOSAL.md`` §4.3
# plus the processor / config autos and TTNN-specific public API.
EXPECTED_AUTO_MODEL_CLASSES = [
    "AutoModel",
    "AutoModelForPreTraining",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForMaskGeneration",
    "AutoModelForKeypointDetection",
    "AutoModelForKeypointMatching",
    "AutoModelForTextEncoding",
    "AutoModelForImageToImage",
    "AutoModelForSequenceClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForTableQuestionAnswering",
    "AutoModelForVisualQuestionAnswering",
    "AutoModelForDocumentQuestionAnswering",
    "AutoModelForTokenClassification",
    "AutoModelForMultipleChoice",
    "AutoModelForNextSentencePrediction",
    "AutoModelForImageClassification",
    "AutoModelForZeroShotImageClassification",
    "AutoModelForImageSegmentation",
    "AutoModelForSemanticSegmentation",
    "AutoModelForUniversalSegmentation",
    "AutoModelForInstanceSegmentation",
    "AutoModelForObjectDetection",
    "AutoModelForZeroShotObjectDetection",
    "AutoModelForDepthEstimation",
    "AutoModelForTextRecognition",
    "AutoModelForTableRecognition",
    "AutoModelForVideoClassification",
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForAudioClassification",
    "AutoModelForCTC",
    "AutoModelForTDT",
    "AutoModelForSpeechSeq2Seq",
    "AutoModelForAudioFrameClassification",
    "AutoModelForAudioXVector",
    "AutoModelForTextToSpectrogram",
    "AutoModelForTextToWaveform",
    "AutoModelForTimeSeriesPrediction",
    "AutoModelForMaskedImageModeling",
    "AutoModelForAudioTokenization",
    "AutoBackbone",
]

EXPECTED_PROCESSOR_AUTOS = [
    "AutoConfig",
    "AutoTokenizer",
    "AutoImageProcessor",
    "AutoFeatureExtractor",
    "AutoProcessor",
    "AutoVideoProcessor",
]

EXPECTED_TT_PUBLIC_SYMBOLS = [
    "DispatchManager",
    "Recipe",
    "TT_MODEL_REGISTRY",
    "TracedRun",
    "register_modules",
    "register_recipe",
    "set_device",
]


@pytest.mark.parametrize("name", EXPECTED_AUTO_MODEL_CLASSES)
def test_top_level_auto_model_import(name):
    import tt_symbiote

    assert hasattr(tt_symbiote, name), f"tt_symbiote.{name} is missing"
    cls = getattr(tt_symbiote, name)
    assert isinstance(cls, type), f"tt_symbiote.{name} should be a class, got {type(cls)}"
    # Every Auto* class must subclass _BaseAutoModelClass.
    from tt_symbiote.models.auto.auto_factory import _BaseAutoModelClass

    assert issubclass(cls, _BaseAutoModelClass), f"tt_symbiote.{name} must subclass _BaseAutoModelClass"
    assert cls._HF_AUTO_CLASS is not None, f"tt_symbiote.{name}._HF_AUTO_CLASS is unset"


@pytest.mark.parametrize("name", EXPECTED_PROCESSOR_AUTOS)
def test_top_level_processor_auto_import(name):
    import tt_symbiote

    assert hasattr(tt_symbiote, name), f"tt_symbiote.{name} is missing"


@pytest.mark.parametrize("name", EXPECTED_TT_PUBLIC_SYMBOLS)
def test_top_level_tt_public_symbol(name):
    import tt_symbiote

    assert hasattr(tt_symbiote, name), f"tt_symbiote.{name} is missing"


def test_count_matches_proposal():
    # Sanity check that we ship at least 43 AutoModel* classes (docs/internal/PROJECT_PROPOSAL.md §4.3).
    assert len(EXPECTED_AUTO_MODEL_CLASSES) >= 43


def test_auto_subpackage_reexports_everything():
    """`tt_symbiote.models.auto.__all__` covers every AutoModel + processor name."""
    import tt_symbiote.models.auto as auto

    for name in EXPECTED_AUTO_MODEL_CLASSES + EXPECTED_PROCESSOR_AUTOS:
        assert name in auto.__all__, f"{name} missing from tt_symbiote.models.auto.__all__"
        assert hasattr(auto, name), f"tt_symbiote.models.auto.{name} is missing"
