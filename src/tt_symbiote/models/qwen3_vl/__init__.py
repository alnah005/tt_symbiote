# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL VLM (HF ``Qwen3VLForConditionalGeneration``) TTNN port.

Importing this package runs the ``@register_recipe`` decorator in
:mod:`tt_symbiote.models.qwen3_vl.modeling_qwen3_vl` and inserts
:class:`Qwen3VLRecipe` into
:data:`tt_symbiote.auto.auto_mappings.TT_MODEL_REGISTRY` under the key
``"Qwen3VLForConditionalGeneration"``.

First Qwen3-VL commit ships a CPU-first port: the recipe leaves the
HuggingFace reference modeling untouched (empty ``build_module_dict``)
and uses the three declared class-name lists (``tt_implemented``,
``cpu_fallback``, ``out_of_scope``) to drive
:func:`tt_symbiote.compatibility.report`. This is the first model
landed via the ``port-hf-model-to-tt-symbiote`` Cursor skill.

See :mod:`.configuration_qwen3_vl` for the per-variant TTNN tuning
table (2B / 4B / 8B / 32B dense; MoE variants noted for
discoverability).
"""

from tt_symbiote.models.qwen3_vl.configuration_qwen3_vl import (
    QWEN3_VL_TTNN_TUNING,
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    lookup_ttnn_tuning,
)
from tt_symbiote.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLRecipe

__all__ = [
    "QWEN3_VL_TTNN_TUNING",
    "Qwen3VLConfig",
    "Qwen3VLRecipe",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
    "lookup_ttnn_tuning",
]
