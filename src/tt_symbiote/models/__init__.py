# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Model packages.

Importing this package eagerly imports every recipe-bearing model
subpackage so the ``@register_recipe`` decorators inside their
``modeling_<model>.py`` files run and populate
:data:`tt_symbiote.models.auto.auto_mappings.TT_MODEL_REGISTRY`. This mirrors the
side-effect-import pattern used by ``transformers/models/__init__.py`` to
register Auto-class mappings at top-level import time.

Each entry is guarded with ``try/except`` so a broken subpackage warns
loudly but does not poison the rest of the registry — the user-visible
import path ``from tt_symbiote import AutoModelForCausalLM`` must never
fail because one experimental model file has a syntax error.
"""

from __future__ import annotations

import warnings

# Recipe-bearing subpackages. Add an entry here when a new model lands.
_RECIPE_BEARING_SUBPACKAGES = (
    "bailing_moe_v2",
    "resnet",
    "gemma4",
    "qwen3_vl",
    "dots_ocr",
    "unlimited_ocr",
)


def _eager_import_recipe_packages() -> None:
    """Import every recipe-bearing subpackage to trigger ``@register_recipe``."""
    for name in _RECIPE_BEARING_SUBPACKAGES:
        try:
            __import__(f"tt_symbiote.models.{name}")
        except Exception as e:  # pragma: no cover - exercised on broken installs
            warnings.warn(
                f"tt_symbiote.models: failed to import {name!r} "
                f"({type(e).__name__}: {e}); its recipe will not be registered.",
                stacklevel=2,
            )


_eager_import_recipe_packages()
