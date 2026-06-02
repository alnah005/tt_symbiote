# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""``tt_symbiote.AutoModel*`` classes mirroring ``transformers`` v5.9.0.

Per ``docs/internal/PROJECT_PROPOSAL.md`` §4.3 we ship full ``Auto*`` coverage even for
tasks where we have no TTNN recipe yet. Classes with no recipe just fall
back to "load HF, warn, return unmodified HF model" — that path is
implemented in :class:`tt_symbiote.models.auto.auto_factory._BaseAutoModelClass`.

The set of classes below is the v5.9.0 list extracted from
``transformers.models.auto.modeling_auto``. The file is hand-listed (not
generated) so an explicit grep for ``class AutoModelFor`` finds the same
names a developer would find in HF's source tree.
"""

from __future__ import annotations

import transformers

from tt_symbiote.models.auto.auto_factory import _BaseAutoBackboneClass, _BaseAutoModelClass

__all__ = [
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


# --- core ----------------------------------------------------------------


class AutoModel(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModel


class AutoModelForPreTraining(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForPreTraining


class AutoModelForCausalLM(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForCausalLM


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForMaskedLM


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForSeq2SeqLM


class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForMaskGeneration


class AutoModelForKeypointDetection(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForKeypointDetection


class AutoModelForKeypointMatching(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForKeypointMatching


class AutoModelForTextEncoding(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTextEncoding


class AutoModelForImageToImage(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForImageToImage


# --- classification / QA -------------------------------------------------


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForSequenceClassification


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForQuestionAnswering


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTableQuestionAnswering


class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForVisualQuestionAnswering


class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForDocumentQuestionAnswering


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTokenClassification


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForMultipleChoice


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForNextSentencePrediction


# --- vision --------------------------------------------------------------


class AutoModelForImageClassification(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForImageClassification


class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForZeroShotImageClassification


class AutoModelForImageSegmentation(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForImageSegmentation


class AutoModelForSemanticSegmentation(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForSemanticSegmentation


class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForUniversalSegmentation


class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForInstanceSegmentation


class AutoModelForObjectDetection(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForObjectDetection


class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForZeroShotObjectDetection


class AutoModelForDepthEstimation(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForDepthEstimation


class AutoModelForTextRecognition(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTextRecognition


class AutoModelForTableRecognition(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTableRecognition


class AutoModelForVideoClassification(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForVideoClassification


# --- multimodal ----------------------------------------------------------


class AutoModelForImageTextToText(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForImageTextToText


class AutoModelForMultimodalLM(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForMultimodalLM


# --- audio ---------------------------------------------------------------


class AutoModelForAudioClassification(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForAudioClassification


class AutoModelForCTC(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForCTC


class AutoModelForTDT(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTDT


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForSpeechSeq2Seq


class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForAudioFrameClassification


class AutoModelForAudioXVector(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForAudioXVector


class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTextToSpectrogram


class AutoModelForTextToWaveform(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTextToWaveform


class AutoModelForAudioTokenization(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForAudioTokenization


# --- specialty -----------------------------------------------------------


class AutoModelForTimeSeriesPrediction(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForTimeSeriesPrediction


class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _HF_AUTO_CLASS = transformers.AutoModelForMaskedImageModeling


class AutoBackbone(_BaseAutoBackboneClass):
    _HF_AUTO_CLASS = transformers.AutoBackbone
