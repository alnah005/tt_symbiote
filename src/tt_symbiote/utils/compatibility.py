# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Compatibility / op-coverage reporting for ``tt_symbiote`` recipes.

Phase 7 introduced this surface to answer a simple operational question:
*"For a given loaded model, which submodules are TTNN-accelerated and which
ones are running on the CPU via the PyTorch fallback path?"*

Two complementary information sources feed the report:

1. **Design-time intent**, read from the registered recipe. Each recipe may
   declare four class-name lists describing how it *intends* to map the
   upstream HF model:

   - ``tt_implemented``: HF class names for which the recipe ships a TTNN
     wrapper in :meth:`build_module_dict`.
   - ``cpu_fallback``: HF class names that are deliberately left as
     PyTorch modules (either because no TTNN equivalent exists yet or
     because the op is cheap enough to ignore).
   - ``host_glue``: HF class names that are *intentionally* host-only by
     policy (orchestration, output dataclasses, mask building, scatter
     fusion, index walks). Distinguishing these from ``cpu_fallback``
     prevents Phase 8+ ports from being flagged for never moving glue
     code that has no compute to accelerate.
   - ``out_of_scope``: HF class names that exist in the model file but are
     never exercised by the documented demos (e.g. the Gemma-4 audio
     tower under the image-text-to-text path).

2. **Runtime observation**, populated by the small hooks in
   :mod:`tt_symbiote.core.run_config` that fire whenever a ``TTNNModule``'s
   ``forward`` raises and is rescued by the stored ``_fallback_torch_layer``.
   These events are appended to a process-global ledger keyed by module
   name (``{module_name: class_name}``); :func:`report` aggregates the ledger
   by class for compactness.

The intent of the two-track design is to make the *gap* between design and
runtime visible: an entry that shows up in the runtime ledger but not in the
recipe's ``cpu_fallback`` list is a real, actionable signal — either the
recipe is incomplete or a TTNN op silently regressed.

Phase 7's first commit ships the Gemma-4 recipe as a fully CPU-first port
(empty ``build_module_dict``), so its design-time ``cpu_fallback`` list is
the only populated channel. As TTNN wrappers are added in subsequent
commits the same surface will track the migration without further work.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict, List

__all__ = [
    "report",
    "record_runtime_fallback",
    "reset_runtime_observations",
]


# ---------------------------------------------------------------------------
# Runtime ledger
# ---------------------------------------------------------------------------
#
# A single process-wide store, guarded by a lock so concurrent forward
# passes (rare but legal for multi-device meshes) don't race on the dict.
# The ledger is keyed by the unique module name (e.g. ``"model.vision_tower
# .encoder.layers.3.self_attn"``) so multiple instances of the same class
# show up as separate entries. :func:`report` collapses them by class for
# the summary view.

_LEDGER_LOCK = threading.Lock()
_RUNTIME_LEDGER: Dict[str, str] = {}


def record_runtime_fallback(module_name: str, class_name: str) -> None:
    """Note that ``module_name`` (of type ``class_name``) hit the torch fallback path.

    Called by the warning sites in :mod:`tt_symbiote.core.run_config`.
    Safe to call from any thread. Bad / missing inputs are silently
    ignored — this is observability, not a correctness path.
    """
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        return
    with _LEDGER_LOCK:
        _RUNTIME_LEDGER[module_name] = class_name


def reset_runtime_observations() -> None:
    """Clear the runtime ledger (e.g. between tests or between demos)."""
    with _LEDGER_LOCK:
        _RUNTIME_LEDGER.clear()


def _runtime_observations() -> Dict[str, str]:
    """Snapshot of the runtime ledger; copy so callers can mutate freely."""
    with _LEDGER_LOCK:
        return dict(_RUNTIME_LEDGER)


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


def _recipe_for(model: Any):
    """Look up the registered recipe for ``model``'s class, if any."""
    try:
        from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY
    except Exception:
        return None
    return TT_MODEL_REGISTRY.get(type(model).__name__)


def _list_attr(obj: Any, name: str) -> List[str]:
    """Best-effort fetch of a list[str] attribute; never raises."""
    if obj is None:
        return []
    value = getattr(obj, name, None)
    if value is None:
        return []
    try:
        return [str(v) for v in value]
    except Exception:
        return []


def report(model: Any) -> Dict[str, Any]:
    """Return a structured op-coverage report for ``model``.

    The shape is intentionally JSON-friendly so it round-trips through
    logs, CI artefacts, and the ``docs/supported_models.md`` table:

    .. code-block:: python

        {
          "model_class": "Gemma4ForConditionalGeneration",
          "design_time": {
              "tt_implemented": [...],          # from recipe.tt_implemented
              "cpu_fallback":   [...],          # from recipe.cpu_fallback
              "host_glue":      [...],          # from recipe.host_glue
              "out_of_scope":   [...],          # from recipe.out_of_scope
          },
          "runtime_observed": {
              "by_class": {"Gemma4VisionAttention": 27, ...},  # count of unique modules per class
              "by_module": {"model.vision_tower...": "Gemma4VisionAttention", ...},
              "unexpected": ["SomeClassNotInRecipeLists"],     # runtime hits not declared in any list
          },
          "summary": {
              "tt_implemented_count": 0,
              "cpu_fallback_count": 19,
              "runtime_fallback_count": 0,
          },
        }

    A model with no registered recipe gets a placeholder shape so callers
    that pipe the output to JSON don't need a special case.
    """
    recipe = _recipe_for(model)

    tt_impl = _list_attr(recipe, "tt_implemented")
    cpu_fb = _list_attr(recipe, "cpu_fallback")
    host_glue = _list_attr(recipe, "host_glue")
    oos = _list_attr(recipe, "out_of_scope")
    declared: set = set(tt_impl) | set(cpu_fb) | set(host_glue) | set(oos)

    runtime = _runtime_observations()
    by_class: Counter = Counter(runtime.values())
    unexpected = sorted({cls for cls in runtime.values() if cls not in declared})

    return {
        "model_class": type(model).__name__,
        "design_time": {
            "tt_implemented": list(tt_impl),
            "cpu_fallback": list(cpu_fb),
            "host_glue": list(host_glue),
            "out_of_scope": list(oos),
        },
        "runtime_observed": {
            "by_class": dict(by_class),
            "by_module": dict(runtime),
            "unexpected": unexpected,
        },
        "summary": {
            "tt_implemented_count": len(tt_impl),
            "cpu_fallback_count": len(cpu_fb),
            "host_glue_count": len(host_glue),
            "out_of_scope_count": len(oos),
            "runtime_fallback_count": len(runtime),
            "runtime_unexpected_count": len(unexpected),
        },
    }
