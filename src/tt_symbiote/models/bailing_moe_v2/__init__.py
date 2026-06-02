# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Ling-mini-2.0 (HF ``BailingMoeV2ForCausalLM``) TTNN port.

Importing this package runs the ``@register_recipe`` decorator in
:mod:`tt_symbiote.models.bailing_moe_v2.modeling_bailing_moe_v2` and
inserts :class:`BailingMoEV2Recipe` into
:data:`tt_symbiote.models.auto.auto_mappings.TT_MODEL_REGISTRY` under the key
``"BailingMoeV2ForCausalLM"``.
"""

from tt_symbiote.models.bailing_moe_v2.modeling_bailing_moe_v2 import (
    BailingMoEV2Recipe,
    TTNNBailingMoEDecoderLayer,
    TTNNBailingMoEDecoderLayerPadded,
    TTNNBailingMoeV2Model,
)

__all__ = [
    "BailingMoEV2Recipe",
    "TTNNBailingMoEDecoderLayer",
    "TTNNBailingMoEDecoderLayerPadded",
    "TTNNBailingMoeV2Model",
]
