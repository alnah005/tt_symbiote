# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Phase 7 Gemma-4 VLM CPU-first port.

Verifies that the recipe loads cleanly through the new public API
without needing live hardware *or* multi-GB weight downloads:

1. The recipe is registered under
   ``"Gemma4ForConditionalGeneration"``.
2. The recipe's ``build_module_dict`` is empty (Phase 7 contract).
3. The compatibility report reads the design-time lists and reports
   an empty runtime ledger.

The end-to-end "dog" demo lives in ``examples/e2e/run_gemma4_e2b.py``
(E2B on N150) and ``examples/e2e/run_gemma4_31b.py`` (31B on T3K).
Those scripts each pull ~10 GB / ~58 GB of weights and require a live
mesh device, so they intentionally do not run under pytest.

Once TTNN wrappers land in a subsequent commit the recipe's
``build_module_dict`` will become non-empty and the matching hardware
smoke test will be re-introduced here (mirroring the Phase 5
``tests/models/bailing_moe_v2/test_modeling_bailing_moe_v2.py`` and
Phase 6 ``tests/models/resnet/test_modeling_resnet.py`` patterns).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def gemma4_recipe():
    """Side-effect import: load the Gemma-4 modeling package, get the recipe."""
    import tt_symbiote.models.gemma4  # noqa: F401
    from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY

    assert "Gemma4ForConditionalGeneration" in TT_MODEL_REGISTRY, (
        "Gemma4Recipe should be registered after importing "
        "tt_symbiote.models.gemma4 (eager side-effect import in __init__.py)."
    )
    return TT_MODEL_REGISTRY["Gemma4ForConditionalGeneration"]


def test_recipe_has_phase8_wave_a_wrappers(gemma4_recipe):
    """Phase 8 contract: 5 TTNN wrappers declared, plus the design-time manifest."""
    assert set(gemma4_recipe.tt_implemented) >= {
        "Gemma4RMSNorm",
        "Gemma4TextScaledWordEmbedding",
        "Gemma4TextMLP",
        "Gemma4VisionMLP",
        "Gemma4MultimodalEmbedder",
    }
    assert len(gemma4_recipe.cpu_fallback) >= 14, (
        "cpu_fallback should still enumerate the text + vision attention "
        "stacks (deferred) exercised by the image-text-to-text demo"
    )


def test_recipe_make_kv_cache_is_noop(gemma4_recipe):
    """HF DynamicCache handles short generation; Phase 7 keeps it."""
    assert gemma4_recipe.make_kv_cache(model=None, device=None) is None


def test_top_level_compatibility_module_exposed():
    """``import tt_symbiote`` should expose the ``compatibility`` submodule."""
    import tt_symbiote

    assert hasattr(
        tt_symbiote, "compatibility"
    ), "tt_symbiote.compatibility is the top-level op-coverage report surface"
    assert callable(tt_symbiote.compatibility.report)
    assert callable(tt_symbiote.compatibility.reset_runtime_observations)
    assert callable(tt_symbiote.compatibility.reset_swapped_registry)


def test_compatibility_report_shape_for_gemma4():
    """End-to-end: ``compatibility.report`` returns the Phase 8.5 runtime shape."""
    from tt_symbiote.utils.compatibility import (
        report,
        reset_runtime_observations,
        reset_swapped_registry,
    )

    class Gemma4ForConditionalGeneration:  # noqa: N801 — match HF class name
        pass

    reset_runtime_observations()
    reset_swapped_registry()
    out = report(Gemma4ForConditionalGeneration())

    for top_key in (
        "model_class",
        "ttnn_swap_skipped",
        "ttnn_swap_skipped_reason",
        "modules_swapped",
        "runtime_observed",
        "regressions",
        "summary",
    ):
        assert top_key in out, f"report missing top-level key {top_key!r}"
    assert "design_time" not in out, "Phase 8.5 dropped the design_time block from the runtime artefact"

    for swapped_key in ("by_class", "by_module"):
        assert swapped_key in out["modules_swapped"], f"report['modules_swapped'] missing {swapped_key!r}"

    for runtime_key in ("successes_by_class", "fallbacks_by_class", "fallbacks_by_module"):
        assert runtime_key in out["runtime_observed"], f"report['runtime_observed'] missing {runtime_key!r}"

    assert out["summary"]["runtime_fallbacks"] == 0, "no forward was run; runtime ledger should be empty"
    assert out["regressions"] == []
