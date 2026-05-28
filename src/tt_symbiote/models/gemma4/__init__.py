# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma-4 VLM (HF ``Gemma4ForConditionalGeneration``) TTNN port.

Importing this package runs the ``@register_recipe`` decorator in
:mod:`tt_symbiote.models.gemma4.modeling_gemma4` and inserts
:class:`Gemma4Recipe` into
:data:`tt_symbiote.auto.auto_mappings.TT_MODEL_REGISTRY` under the key
``"Gemma4ForConditionalGeneration"``.

Phase 7 ships a CPU-first port: the recipe leaves the HuggingFace
reference modeling untouched (empty ``build_module_dict``) and uses the
three declared class-name lists (``tt_implemented``, ``cpu_fallback``,
``out_of_scope``) to drive :func:`tt_symbiote.compatibility.report`.

See :mod:`.configuration_gemma4` for the per-variant TTNN tuning table
(``google/gemma-4-E2B-it`` on N150, ``google/gemma-4-31B-it`` on T3K).
"""

from tt_symbiote.models.gemma4.configuration_gemma4 import (
    GEMMA4_TTNN_TUNING,
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    lookup_ttnn_tuning,
)
from tt_symbiote.models.gemma4.modeling_gemma4 import Gemma4Recipe

__all__ = [
    "GEMMA4_TTNN_TUNING",
    "Gemma4AudioConfig",
    "Gemma4Config",
    "Gemma4Recipe",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "lookup_ttnn_tuning",
]
