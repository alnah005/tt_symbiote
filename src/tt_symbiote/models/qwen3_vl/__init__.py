# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL VLM (HF ``Qwen3VLForConditionalGeneration``) TTNN port.

Importing this package runs the ``@register_recipe`` decorator in
:mod:`tt_symbiote.models.qwen3_vl.modeling_qwen3_vl` and inserts
:class:`Qwen3VLRecipe` into
:data:`tt_symbiote.models.auto.auto_mappings.TT_MODEL_REGISTRY` under the key
``"Qwen3VLForConditionalGeneration"``.

Phase 8 Wave B moves the structurally simple compute classes
(``Qwen3VLTextRMSNorm``, ``Qwen3VLTextMLP``, ``Qwen3VLVisionMLP``,
``Qwen3VLVisionPatchMerger``) to on-device wrappers built on existing
TTNN integrations. Bespoke pieces (M-RoPE, varlen-packed vision SDPA,
DeepStack injection) stay declared ``cpu_fallback``; the top-level
fusion (``Qwen3VLPreTrainedModel``, ``Qwen3VLModel``,
``Qwen3VLForConditionalGeneration``) is declared ``host_glue``
(orchestration only — no FLOPs to accelerate). The four declared
class-name lists (``tt_implemented``, ``cpu_fallback``, ``host_glue``,
``out_of_scope``) drive :func:`tt_symbiote.compatibility.report`.

This model was first landed via the ``port-hf-model-to-tt-symbiote``
Cursor skill.

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
