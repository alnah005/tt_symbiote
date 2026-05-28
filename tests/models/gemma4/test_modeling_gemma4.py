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
    from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY

    assert "Gemma4ForConditionalGeneration" in TT_MODEL_REGISTRY, (
        "Gemma4Recipe should be registered after importing "
        "tt_symbiote.models.gemma4 (eager side-effect import in __init__.py)."
    )
    return TT_MODEL_REGISTRY["Gemma4ForConditionalGeneration"]


def test_recipe_is_cpu_first_phase(gemma4_recipe):
    """Phase 7 ships a CPU-first port: empty build_module_dict, populated coverage lists."""
    assert gemma4_recipe.build_module_dict(None) == {}, (
        "Phase 7 contract: no TTNN swaps until vision/text wrappers land "
        "in a follow-up commit."
    )
    assert gemma4_recipe.tt_implemented == []
    assert len(gemma4_recipe.cpu_fallback) >= 19, (
        "cpu_fallback should enumerate the text + vision + multimodal "
        "modules exercised by the image-text-to-text demo"
    )


def test_recipe_make_kv_cache_is_noop(gemma4_recipe):
    """HF DynamicCache handles short generation; Phase 7 keeps it."""
    assert gemma4_recipe.make_kv_cache(model=None, device=None) is None


def test_top_level_compatibility_module_exposed():
    """``import tt_symbiote`` should expose the new ``compatibility`` submodule."""
    import tt_symbiote

    assert hasattr(tt_symbiote, "compatibility"), (
        "Phase 7 added tt_symbiote.compatibility as the top-level op-coverage report"
    )
    assert callable(tt_symbiote.compatibility.report)
    assert callable(tt_symbiote.compatibility.reset_runtime_observations)


def test_compatibility_report_shape_for_gemma4():
    """End-to-end: ``compatibility.report`` returns the JSON-friendly Phase 7 shape."""
    from tt_symbiote.utils.compatibility import report, reset_runtime_observations

    class Gemma4ForConditionalGeneration:  # noqa: N801 — match HF class name
        pass

    reset_runtime_observations()
    out = report(Gemma4ForConditionalGeneration())

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
