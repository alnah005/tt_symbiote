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


def test_recipe_is_cpu_first_phase(qwen3_vl_recipe):
    """CPU-first contract: empty build_module_dict, populated coverage lists."""
    assert qwen3_vl_recipe.build_module_dict(None) == {}, (
        "CPU-first contract: no TTNN swaps until vision/text wrappers land "
        "in a follow-up commit."
    )
    assert qwen3_vl_recipe.tt_implemented == []
    assert len(qwen3_vl_recipe.cpu_fallback) >= 16, (
        "cpu_fallback should enumerate the vision tower (7), the text "
        "decoder (6), and top-level composites (3) exercised by the "
        "image-text-to-text demo — 16 total minimum"
    )


def test_recipe_make_kv_cache_is_noop(qwen3_vl_recipe):
    """HF DynamicCache handles short generation; CPU-first commit keeps it."""
    assert qwen3_vl_recipe.make_kv_cache(model=None, device=None) is None


def test_compatibility_report_shape_for_qwen3_vl():
    """End-to-end: ``compatibility.report`` returns the JSON-friendly Phase 7 shape."""
    from tt_symbiote.utils.compatibility import report, reset_runtime_observations

    class Qwen3VLForConditionalGeneration:  # noqa: N801 — match HF class name
        pass

    reset_runtime_observations()
    out = report(Qwen3VLForConditionalGeneration())

    for top_key in ("model_class", "design_time", "runtime_observed", "summary"):
        assert top_key in out, f"report missing top-level key {top_key!r}"

    for design_key in ("tt_implemented", "cpu_fallback", "out_of_scope"):
        assert design_key in out["design_time"], (
            f"report['design_time'] missing {design_key!r}"
        )

    for runtime_key in ("by_class", "by_module", "unexpected"):
        assert runtime_key in out["runtime_observed"], (
            f"report['runtime_observed'] missing {runtime_key!r}"
        )

    assert out["summary"]["runtime_fallback_count"] == 0, (
        "no forward was run; runtime ledger should be empty"
    )
