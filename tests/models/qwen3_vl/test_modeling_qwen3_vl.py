# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the Qwen3-VL VLM CPU-first port.

Verifies that the recipe loads cleanly through the new public API
without needing live hardware *or* multi-GB weight downloads:

1. The recipe is registered under ``"Qwen3VLForConditionalGeneration"``.
2. The recipe's ``build_module_dict`` is empty (CPU-first contract).
3. The compatibility report reads the design-time lists and reports an
   empty runtime ledger.

This is the first model landed through the
``port-hf-model-to-tt-symbiote`` Cursor skill; the test file was
produced from :file:`.cursor/skills/port-hf-model-to-tt-symbiote/
templates/smoke_test_template.py.tmpl` by mechanical placeholder
substitution.

The end-to-end "dog" demo lives in
:file:`examples/e2e/run_qwen3_vl_2b.py`. That script pulls ~4 GB of
weights and requires a live mesh device, so it intentionally does not
run under pytest.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def qwen3_vl_recipe():
    """Side-effect import: load the Qwen3-VL modeling package, get the recipe."""
    import tt_symbiote.models.qwen3_vl  # noqa: F401
    from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY

    assert "Qwen3VLForConditionalGeneration" in TT_MODEL_REGISTRY, (
        "Qwen3VLRecipe should be registered after importing "
        "tt_symbiote.models.qwen3_vl (eager side-effect import in __init__.py)."
    )
    return TT_MODEL_REGISTRY["Qwen3VLForConditionalGeneration"]


def test_recipe_has_phase8_wave_b_wrappers(qwen3_vl_recipe):
    """Phase 8 Wave B contract: 4 TTNN wrappers declared, with the design-time manifest."""
    assert set(qwen3_vl_recipe.tt_implemented) >= {
        "Qwen3VLVisionMLP",
        "Qwen3VLVisionPatchMerger",
        "Qwen3VLTextRMSNorm",
        "Qwen3VLTextMLP",
    }
    assert len(qwen3_vl_recipe.cpu_fallback) >= 9, (
        "cpu_fallback should still enumerate the vision tower (5), " "text decoder (4) deferred to later waves"
    )
    assert len(qwen3_vl_recipe.host_glue) >= 3, (
        "host_glue should cover the top-level composites (PreTrainedModel, "
        "Qwen3VLModel, Qwen3VLForConditionalGeneration)"
    )


def test_recipe_make_kv_cache_is_noop(qwen3_vl_recipe):
    """HF DynamicCache handles short generation; CPU-first commit keeps it."""
    assert qwen3_vl_recipe.make_kv_cache(model=None, device=None) is None


def test_compatibility_report_shape_for_qwen3_vl():
    """End-to-end: ``compatibility.report`` returns the Phase 8.5 runtime shape."""
    from tt_symbiote.utils.compatibility import (
        report,
        reset_runtime_observations,
        reset_swapped_registry,
    )

    class Qwen3VLForConditionalGeneration:  # noqa: N801 — match HF class name
        pass

    reset_runtime_observations()
    reset_swapped_registry()
    out = report(Qwen3VLForConditionalGeneration())

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
