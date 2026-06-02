# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for ``tt_symbiote.utils.compatibility``.

Phase 8.5 reframed ``_coverage.json`` (and the ``report()`` surface that
feeds it) as a **pure runtime observation artefact**: design-time class
lists are still tracked on the recipe but are no longer serialized into
the report. These tests pin that contract so the JSON shape doesn't
accidentally regress to the pre-8.5 design.

The tests do not require real TTNN: they exercise the public hook API
(:func:`record_swapped_class`, :func:`record_runtime_success`,
:func:`record_runtime_fallback`) directly and stub the recipe lookup
via :data:`tt_symbiote.auto.auto_mappings.TT_MODEL_REGISTRY`.
"""

from __future__ import annotations

import pytest

from tt_symbiote.utils import compatibility


@pytest.fixture(autouse=True)
def _clean_ledgers():
    """Every test starts from a blank slate."""
    compatibility.reset_runtime_observations()
    compatibility.reset_swapped_registry()
    yield
    compatibility.reset_runtime_observations()
    compatibility.reset_swapped_registry()


class _StubRecipe:
    """Minimal recipe stand-in carrying just the field ``report()`` reads."""

    def __init__(self, cpu_fallback=None):
        self.cpu_fallback = list(cpu_fallback or [])


class _StubModel:
    """Stand-in for an HF model. ``report()`` only needs ``type(...).__name__``."""


def _install_stub_recipe(monkeypatch, model_cls, recipe):
    """Make ``compatibility._recipe_for(model)`` return ``recipe``."""
    from tt_symbiote.auto import auto_mappings

    monkeypatch.setitem(auto_mappings.TT_MODEL_REGISTRY, model_cls.__name__, recipe)


def test_clean_shape_has_no_design_time_lists(monkeypatch):
    """A blank-ledger ``report()`` matches the Phase 8.5 schema exactly.

    Specifically: none of the four design-time list names (``tt_implemented``,
    ``cpu_fallback``, ``host_glue``, ``out_of_scope``) appear *anywhere* in
    the JSON shape, even though the recipe declares ``cpu_fallback``. That
    list is used only to compute ``regressions``; the JSON itself is purely
    observational.
    """
    _install_stub_recipe(monkeypatch, _StubModel, _StubRecipe(cpu_fallback=["Foo"]))

    out = compatibility.report(_StubModel())

    expected_top_level = {
        "model_class",
        "ttnn_swap_skipped",
        "ttnn_swap_skipped_reason",
        "modules_swapped",
        "runtime_observed",
        "regressions",
        "summary",
    }
    assert set(out.keys()) == expected_top_level
    assert "design_time" not in out
    for forbidden in ("tt_implemented", "cpu_fallback", "host_glue", "out_of_scope"):
        assert forbidden not in out
        assert forbidden not in out["modules_swapped"]
        assert forbidden not in out["runtime_observed"]
        assert forbidden not in out["summary"]

    assert out["modules_swapped"] == {"by_class": {}, "by_module": {}}
    assert out["runtime_observed"] == {
        "successes_by_class": {},
        "fallbacks_by_class": {},
        "fallbacks_by_module": {},
    }
    assert out["regressions"] == []
    assert out["summary"] == {
        "modules_swapped_count": 0,
        "runtime_successes": 0,
        "runtime_fallbacks": 0,
        "regression_count": 0,
    }
    assert out["ttnn_swap_skipped"] is False
    assert out["ttnn_swap_skipped_reason"] is None


def test_modules_swapped_aggregated_by_class():
    """``record_swapped_class`` populates ``modules_swapped`` by-class + by-module.

    Two modules of class ``Bar`` plus one ``Foo`` should aggregate to
    ``{"Bar": 2, "Foo": 1}`` in ``by_class``. ``by_module`` is the raw
    mapping for traceability.
    """
    compatibility.record_swapped_class("model.layer.0", "Bar")
    compatibility.record_swapped_class("model.layer.1", "Bar")
    compatibility.record_swapped_class("model.layer.2", "Foo")

    out = compatibility.report(_StubModel())

    assert out["modules_swapped"]["by_class"] == {"Bar": 2, "Foo": 1}
    assert out["modules_swapped"]["by_module"] == {
        "model.layer.0": "Bar",
        "model.layer.1": "Bar",
        "model.layer.2": "Foo",
    }
    assert out["summary"]["modules_swapped_count"] == 3


def test_successes_aggregate_call_counts_by_class():
    """``record_runtime_success`` accumulates per-module call counts then class-aggregates.

    Two modules of class ``Foo`` (called 3 and 4 times respectively) plus a
    single ``Bar`` call should aggregate to ``{"Foo": 7, "Bar": 1}``.
    """
    for _ in range(3):
        compatibility.record_runtime_success("model.layer.0", "Foo")
    for _ in range(4):
        compatibility.record_runtime_success("model.layer.1", "Foo")
    compatibility.record_runtime_success("model.layer.2", "Bar")

    out = compatibility.report(_StubModel())

    assert out["runtime_observed"]["successes_by_class"] == {"Foo": 7, "Bar": 1}
    assert out["summary"]["runtime_successes"] == 8


def test_fallbacks_aggregated_by_class_and_module():
    """Fallbacks use set semantics (one entry per module name)."""
    compatibility.record_runtime_fallback("model.vision.0", "Gemma4VisionAttention")
    compatibility.record_runtime_fallback("model.vision.1", "Gemma4VisionAttention")
    compatibility.record_runtime_fallback("model.text.0", "Gemma4TextAttention")

    out = compatibility.report(_StubModel())

    assert out["runtime_observed"]["fallbacks_by_class"] == {
        "Gemma4VisionAttention": 2,
        "Gemma4TextAttention": 1,
    }
    assert out["runtime_observed"]["fallbacks_by_module"] == {
        "model.vision.0": "Gemma4VisionAttention",
        "model.vision.1": "Gemma4VisionAttention",
        "model.text.0": "Gemma4TextAttention",
    }
    assert out["summary"]["runtime_fallbacks"] == 3


def test_regressions_set_difference_against_cpu_fallback(monkeypatch):
    """``regressions == set(fallbacks_by_class) - set(recipe.cpu_fallback)``.

    Concretely: the recipe declares ``Foo`` as expected-fallback; at runtime
    we observe ``Foo`` (declared, ergo not a regression) and ``Bar``
    (undeclared, ergo a regression). The output is sorted for stable diffs.
    """
    _install_stub_recipe(monkeypatch, _StubModel, _StubRecipe(cpu_fallback=["Foo"]))

    compatibility.record_runtime_fallback("model.layer.0", "Foo")
    compatibility.record_runtime_fallback("model.layer.1", "Bar")
    compatibility.record_runtime_fallback("model.layer.2", "Baz")

    out = compatibility.report(_StubModel())

    assert out["regressions"] == ["Bar", "Baz"]
    assert out["summary"]["regression_count"] == 2


def test_regressions_empty_when_all_fallbacks_declared(monkeypatch):
    """All observed fallbacks declared in ``cpu_fallback`` → no regressions."""
    _install_stub_recipe(monkeypatch, _StubModel, _StubRecipe(cpu_fallback=["Foo", "Bar"]))

    compatibility.record_runtime_fallback("model.layer.0", "Foo")
    compatibility.record_runtime_fallback("model.layer.1", "Bar")

    out = compatibility.report(_StubModel())

    assert out["regressions"] == []
    assert out["summary"]["regression_count"] == 0


def test_ttnn_swap_skipped_reflected_from_runtime_config():
    """``_tt_runtime_config`` is the canonical source for the two gate fields."""
    model = _StubModel()
    model._tt_runtime_config = {
        "ttnn_swap_skipped": True,
        "ttnn_swap_skipped_reason": "Replicated weight footprint exceeds DRAM budget",
    }

    out = compatibility.report(model)

    assert out["ttnn_swap_skipped"] is True
    assert out["ttnn_swap_skipped_reason"] == ("Replicated weight footprint exceeds DRAM budget")


def test_gate_fired_produces_empty_modules_swapped(monkeypatch):
    """Acceptance criterion 3: gate fires → modules_swapped empty, regressions empty.

    Mirrors the Gemma-4 31B / 26B-A4B case where ``build_module_dict``
    returns ``{}`` because of the chip-budget gate. No TTNN modules in the
    tree → no swapped entries → no successes → no fallbacks (the model runs
    on pure PyTorch) → no regressions.
    """
    _install_stub_recipe(monkeypatch, _StubModel, _StubRecipe(cpu_fallback=["Foo"]))

    model = _StubModel()
    model._tt_runtime_config = {"ttnn_swap_skipped": True}

    out = compatibility.report(model)

    assert out["ttnn_swap_skipped"] is True
    assert out["modules_swapped"] == {"by_class": {}, "by_module": {}}
    assert out["runtime_observed"]["successes_by_class"] == {}
    assert out["runtime_observed"]["fallbacks_by_class"] == {}
    assert out["regressions"] == []


def test_reset_helpers_clear_independently():
    """``reset_runtime_observations`` and ``reset_swapped_registry`` are scoped."""
    compatibility.record_swapped_class("model.a", "Foo")
    compatibility.record_runtime_success("model.a", "Foo")
    compatibility.record_runtime_fallback("model.b", "Bar")

    compatibility.reset_runtime_observations()
    out = compatibility.report(_StubModel())
    assert out["modules_swapped"]["by_class"] == {"Foo": 1}
    assert out["runtime_observed"]["successes_by_class"] == {}
    assert out["runtime_observed"]["fallbacks_by_class"] == {}

    compatibility.reset_swapped_registry()
    out = compatibility.report(_StubModel())
    assert out["modules_swapped"] == {"by_class": {}, "by_module": {}}


def test_model_with_no_recipe_still_emits_full_shape():
    """A model whose class is not in the registry still gets the full shape.

    ``cpu_fallback`` defaults to empty, so every observed fallback becomes a
    regression. Useful so demo callers can pipe the report to JSON without
    a special case.
    """
    compatibility.record_runtime_fallback("model.layer.0", "MysteryClass")

    out = compatibility.report(_StubModel())

    assert out["regressions"] == ["MysteryClass"]
    assert set(out.keys()) >= {
        "model_class",
        "ttnn_swap_skipped",
        "modules_swapped",
        "runtime_observed",
        "regressions",
        "summary",
    }


def test_bad_inputs_are_silently_ignored():
    """The hooks are observability — never let bad inputs raise."""
    compatibility.record_swapped_class(None, "Foo")
    compatibility.record_swapped_class("model.a", None)
    compatibility.record_runtime_success(123, "Foo")
    compatibility.record_runtime_fallback("model.a", 456)

    out = compatibility.report(_StubModel())

    assert out["modules_swapped"]["by_module"] == {}
    assert out["runtime_observed"]["successes_by_class"] == {}
    assert out["runtime_observed"]["fallbacks_by_class"] == {}
