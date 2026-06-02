# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Runtime op-coverage reporting for ``tt_symbiote`` recipes.

Phase 8.5 reframed this surface as a **pure runtime observation artefact**
rather than a serialization of the recipe's declared intent. Recipes still
ship the four design-time class-name lists (``tt_implemented`` /
``cpu_fallback`` / ``host_glue`` / ``out_of_scope``) — they're consumed by
the budget gate, the porting skill, and the migration docs — but the JSON
emitted by :func:`report` no longer echoes them. Instead it serializes
exactly *what happened on this run*:

1. **What got swapped.** Populated once by :mod:`tt_symbiote.utils.device_management`
   at the end of :func:`set_device` for every ``TTNNModule`` still bound to
   the model after the arch-support gate. Records ``module_name ->
   class_name`` of the preserved ``_fallback_torch_layer`` so the report
   surface uses HF class names everywhere.

2. **What ran successfully on device.** Populated by
   :mod:`tt_symbiote.core.run_config` from the success path of every
   ``Run.__call__`` that actually attempted a TTNN forward
   (``NormalRun``, ``NormalRunWithFallback``). A module that successfully
   ran TTNN at least once shows up in this ledger; the report aggregates
   by class.

3. **What fell back to torch.** Populated by the existing
   ``_record_runtime_fallback`` call sites in :mod:`tt_symbiote.core.run_config`
   (TTNN forward raised, no device, etc.). Same ``module_name ->
   class_name`` shape.

The recipe's ``cpu_fallback`` list is still consulted at ``report()``
time — but only to compute a single derived field, ``regressions`` =
``set(fallbacks_by_class) - set(recipe.cpu_fallback)``. That preserves
the "is this fallback expected?" signal without serializing the design-
time lists into the JSON.

See ``docs/development/cpu_vs_device_coverage.md`` for the operational write-up and
``docs/development/migration_notes.md`` for the Phase 8.5 design notes.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict, List

__all__ = [
    "report",
    "record_runtime_fallback",
    "record_runtime_success",
    "record_swapped_class",
    "reset_runtime_observations",
    "reset_swapped_registry",
]


# ---------------------------------------------------------------------------
# Runtime ledgers
# ---------------------------------------------------------------------------
#
# Three process-wide stores guarded by a single lock so concurrent forward
# passes (rare but legal for multi-device meshes) don't race. The ledgers
# are populated by hooks in ``tt_symbiote.core.run_config`` (success +
# fallback) and ``tt_symbiote.utils.device_management`` (swap walk after
# ``set_device``).
#
# Semantics intentionally differ between the two runtime ledgers:
#
#   * ``_RUNTIME_LEDGER`` (fallback): ``module_name -> class_name`` (set
#     semantics — a module that falls back N times shows up once).
#     A fallback is usually a permanent property of a module on a run
#     (the TTNN path is broken or the device is unset), so call-count
#     would just amplify noise.
#   * ``_SUCCESS_LEDGER`` (success): ``module_name -> class_name``.
#   * ``_SUCCESS_CALL_COUNT`` (success calls): ``module_name -> int``.
#     Successes accumulate per forward call; aggregating into class-level
#     totals gives a useful "this op ran on device 1560 times this run"
#     signal that distinguishes a quick smoke test from a full decode.
#   * ``_SWAPPED_REGISTRY`` (set_device walk): ``module_name -> class_name``.
#     Populated exactly once per ``set_device`` call.

_LEDGER_LOCK = threading.Lock()
_RUNTIME_LEDGER: Dict[str, str] = {}
_SUCCESS_LEDGER: Dict[str, str] = {}
_SUCCESS_CALL_COUNT: Counter = Counter()
_SWAPPED_REGISTRY: Dict[str, str] = {}


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


def record_runtime_success(module_name: str, class_name: str) -> None:
    """Note that ``module_name`` (of type ``class_name``) ran TTNN successfully.

    Fired by the success path of every TTNN-attempting ``Run.__call__``
    (``NormalRun``, ``NormalRunWithFallback``). Bumps a per-module call
    counter so :func:`report` can summarize "how many on-device forward
    calls did this run rack up" — useful for distinguishing a smoke
    test from a full decode.
    """
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        return
    with _LEDGER_LOCK:
        _SUCCESS_LEDGER[module_name] = class_name
        _SUCCESS_CALL_COUNT[module_name] += 1


def record_swapped_class(module_name: str, class_name: str) -> None:
    """Note that ``module_name`` was wrapped by a ``TTNNModule`` at set_device time.

    Called by :mod:`tt_symbiote.utils.device_management` from a post-walk
    pass over the model tree. ``class_name`` should be the HF source
    class (the class of ``_fallback_torch_layer``) so the registry is
    cross-referenceable with the recipe's design-time lists and with the
    fallback ledger.
    """
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        return
    with _LEDGER_LOCK:
        _SWAPPED_REGISTRY[module_name] = class_name


def reset_runtime_observations() -> None:
    """Clear the success + fallback ledgers (e.g. between tests or between demos)."""
    with _LEDGER_LOCK:
        _RUNTIME_LEDGER.clear()
        _SUCCESS_LEDGER.clear()
        _SUCCESS_CALL_COUNT.clear()


def reset_swapped_registry() -> None:
    """Clear the swapped-class registry (e.g. between tests or between demos)."""
    with _LEDGER_LOCK:
        _SWAPPED_REGISTRY.clear()


def _snapshot() -> Dict[str, Dict]:
    """Snapshot the three ledgers under a single lock for a consistent read."""
    with _LEDGER_LOCK:
        return {
            "swapped": dict(_SWAPPED_REGISTRY),
            "successes": dict(_SUCCESS_LEDGER),
            "success_calls": dict(_SUCCESS_CALL_COUNT),
            "fallbacks": dict(_RUNTIME_LEDGER),
        }


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


def _recipe_for(model: Any):
    """Look up the registered recipe for ``model``'s class, if any."""
    try:
        from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY
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
    """Return a structured runtime coverage report for ``model``.

    The JSON shape is intentionally observational — every field
    answers a question of the form *"what happened on this run?"* and
    none of them echoes the recipe's declared intent. ``regressions``
    is the only derived field; it's the set of HF classes observed
    in the fallback ledger that the recipe does *not* declare as an
    expected fallback.

    .. code-block:: python

        {
          "model_class": "Gemma4ForConditionalGeneration",
          "ttnn_swap_skipped": False,
          "ttnn_swap_skipped_reason": None,
          "modules_swapped": {
              "by_class": {"Gemma4TextMLP": 26, "Gemma4RMSNorm": 122, ...},
              "by_module": {"model.language_model.layers.0.mlp": "Gemma4TextMLP", ...},
          },
          "runtime_observed": {
              "successes_by_class": {"Gemma4TextMLP": 1560, ...},
              "fallbacks_by_class": {"Gemma4VisionAttention": 27, ...},
              "fallbacks_by_module": {"model.vision_tower.encoder.layers.0.self_attn":
                                      "Gemma4VisionAttention", ...},
          },
          "regressions": [],
          "summary": {
              "modules_swapped_count": 153,
              "runtime_successes": 1587,
              "runtime_fallbacks": 27,
              "regression_count": 0,
          },
        }

    A model with no registered recipe still emits the same shape (with
    ``regressions`` computed against an empty ``cpu_fallback`` list).
    The runtime ledgers are read under a single snapshot so concurrent
    forward passes can't tear the report.
    """
    recipe = _recipe_for(model)
    cpu_fb = set(_list_attr(recipe, "cpu_fallback"))

    snap = _snapshot()
    swapped_by_module: Dict[str, str] = snap["swapped"]
    successes_by_module: Dict[str, str] = snap["successes"]
    success_calls: Dict[str, int] = snap["success_calls"]
    fallbacks_by_module: Dict[str, str] = snap["fallbacks"]

    modules_swapped_by_class: Counter = Counter(swapped_by_module.values())

    successes_by_class: Counter = Counter()
    for module_name, class_name in successes_by_module.items():
        successes_by_class[class_name] += success_calls.get(module_name, 0)

    fallbacks_by_class: Counter = Counter(fallbacks_by_module.values())

    regressions = sorted({cls for cls in fallbacks_by_class if cls not in cpu_fb})

    runtime_cfg = getattr(model, "_tt_runtime_config", None) or {}
    ttnn_swap_skipped = bool(runtime_cfg.get("ttnn_swap_skipped", False))
    ttnn_swap_skipped_reason = runtime_cfg.get("ttnn_swap_skipped_reason", None)

    return {
        "model_class": type(model).__name__,
        "ttnn_swap_skipped": ttnn_swap_skipped,
        "ttnn_swap_skipped_reason": ttnn_swap_skipped_reason,
        "modules_swapped": {
            "by_class": dict(modules_swapped_by_class),
            "by_module": dict(swapped_by_module),
        },
        "runtime_observed": {
            "successes_by_class": dict(successes_by_class),
            "fallbacks_by_class": dict(fallbacks_by_class),
            "fallbacks_by_module": dict(fallbacks_by_module),
        },
        "regressions": regressions,
        "summary": {
            "modules_swapped_count": sum(modules_swapped_by_class.values()),
            "runtime_successes": sum(successes_by_class.values()),
            "runtime_fallbacks": sum(fallbacks_by_class.values()),
            "regression_count": len(regressions),
        },
    }
