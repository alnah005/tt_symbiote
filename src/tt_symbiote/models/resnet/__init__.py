# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""ResNet image-classification (HF ``ResNetForImageClassification``) TTNN port.

Importing this package runs the ``@register_recipe`` decorator in
:mod:`tt_symbiote.models.resnet.modeling_resnet` and inserts
:class:`ResNetRecipe` into
:data:`tt_symbiote.auto.auto_mappings.TT_MODEL_REGISTRY` under the key
``"ResNetForImageClassification"``.

Covers all five canonical Microsoft variants
(``microsoft/resnet-{18,34,50,101,152}``); see
:mod:`.configuration_resnet` for the per-variant TTNN tuning table.
"""

from tt_symbiote.models.resnet.configuration_resnet import (
    RESNET_TTNN_TUNING,
    ResNetConfig,
    lookup_ttnn_tuning,
)
from tt_symbiote.models.resnet.modeling_resnet import (
    ResNetRecipe,
    TTNNResNetAdaptiveAvgPool2dNHWC,
    TTNNResNetBasicLayer,
    TTNNResNetBottleNeckLayer,
    TTNNResNetConvLayer,
    TTNNResNetEmbeddings,
    TTNNResNetShortCut,
)

__all__ = [
    "RESNET_TTNN_TUNING",
    "ResNetConfig",
    "ResNetRecipe",
    "TTNNResNetAdaptiveAvgPool2dNHWC",
    "TTNNResNetBasicLayer",
    "TTNNResNetBottleNeckLayer",
    "TTNNResNetConvLayer",
    "TTNNResNetEmbeddings",
    "TTNNResNetShortCut",
    "lookup_ttnn_tuning",
]
