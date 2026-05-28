# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Hardware-free unit tests for the Qwen3-VL VLM recipe.

Mirrors :mod:`tests.auto.test_gemma4_recipe`. Verifies the *shape* of
:class:`Qwen3VLRecipe`: registration, CPU-first contract, populated and
disjoint design-time coverage lists, ``post_register`` behaviour,
``make_kv_cache`` no-op, and ``lookup_ttnn_tuning`` fallbacks.

This is the first model landed through the
``port-hf-model-to-tt-symbiote`` Cursor skill; the test file was
produced from :file:`.cursor/skills/port-hf-model-to-tt-symbiote/
templates/recipe_test_template.py.tmpl` by mechanical placeholder
substitution.

Actual model execution (image + text -> "dog") lives in
:file:`examples/e2e/run_qwen3_vl_2b.py`. That script downloads ~4 GB of
weights and requires a live N150 mesh device, so it intentionally does
not run under pytest.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def recipe():
    """Importing the modeling module registers the recipe; return the instance."""
    import tt_symbiote.models.qwen3_vl  # noqa: F401 — side-effect import
    from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY

    return TT_MODEL_REGISTRY["Qwen3VLForConditionalGeneration"]


@pytest.fixture()
def fake_qwen3_vl_model():
    """A minimal stand-in for an HF ``Qwen3VLForConditionalGeneration``.

    ``post_register`` patches ``type(model).device``; we use a fresh
    subclass per test so the property override does not leak between
    tests (same pattern as the Gemma-4 recipe tests).
    """

    class _FakeQwen3VLForConditionalGeneration(nn.Module):
        pass

    instance = _FakeQwen3VLForConditionalGeneration()
    instance.config = SimpleNamespace(
        _name_or_path="Qwen/Qwen3-VL-2B-Instruct",
        text_config=SimpleNamespace(num_hidden_layers=28, hidden_size=2048),
        vision_config=SimpleNamespace(hidden_size=1152),
    )
    return instance


def test_recipe_registered(recipe):
    """Importing ``tt_symbiote.models.qwen3_vl`` populates the registry."""
    assert recipe is not None
    assert callable(recipe.build_module_dict)
    assert callable(recipe.post_register)
    assert callable(recipe.make_kv_cache)


def test_build_module_dict_is_empty_cpu_first(recipe, fake_qwen3_vl_model):
    """CPU-first contract: ``build_module_dict`` returns ``{}``.

    The empty dict is *the* signal that the recipe is in its initial
    CPU-first phase: nothing gets swapped, all HF modules stay as
    PyTorch, and ``set_device`` reduces to attaching
    ``_tt_runtime_config``. Future commits that land TTNN wrappers
    will populate this dict and shrink ``cpu_fallback``.
    """
    module_dict = recipe.build_module_dict(fake_qwen3_vl_model)
    assert module_dict == {}, (
        f"CPU-first port; build_module_dict must return {{}} until "
        f"TTNN wrappers land. Got {module_dict!r}."
    )


def test_design_time_coverage_lists_populated(recipe):
    """The three class-name lists must be present and non-trivial.

    ``tt_implemented`` is empty in the CPU-first commit, but
    ``cpu_fallback`` and ``out_of_scope`` carry the design-time coverage
    manifest that :func:`compatibility.report` surfaces.
    """
    assert isinstance(recipe.tt_implemented, list)
    assert isinstance(recipe.cpu_fallback, list)
    assert isinstance(recipe.out_of_scope, list)

    assert recipe.tt_implemented == [], (
        "CPU-first commit ships zero TTNN wrappers; tt_implemented must be empty"
    )

    # Required vision + text + top-level classes — assert membership on
    # load-bearing entries instead of a brittle ``len()`` check.
    for required in (
        "Qwen3VLVisionModel",
        "Qwen3VLVisionBlock",
        "Qwen3VLVisionPatchMerger",
        "Qwen3VLTextModel",
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLForConditionalGeneration",
    ):
        assert required in recipe.cpu_fallback, (
            f"{required} must be declared in cpu_fallback so the "
            f"compatibility report flags it as 'expected' rather than "
            f"'unexpected' when the demo runs."
        )

    # Output dataclasses live in out_of_scope (they are not nn.Modules).
    for required in (
        "Qwen3VLModelOutputWithPast",
        "Qwen3VLCausalLMOutputWithPast",
        "BaseModelOutputWithDeepstackFeatures",
    ):
        assert required in recipe.out_of_scope, (
            f"{required} should be in out_of_scope (output dataclass, "
            f"not exercised by the image-text-to-text demo)."
        )


def test_design_time_lists_are_disjoint(recipe):
    """A class must not appear in two coverage lists simultaneously.

    Overlap would make ``compatibility.report`` ambiguous (does the
    class count as TT-accelerated or as a known fallback?).
    """
    impl = set(recipe.tt_implemented)
    fallback = set(recipe.cpu_fallback)
    oos = set(recipe.out_of_scope)

    assert impl.isdisjoint(fallback), (
        f"tt_implemented and cpu_fallback overlap on {impl & fallback}"
    )
    assert impl.isdisjoint(oos), (
        f"tt_implemented and out_of_scope overlap on {impl & oos}"
    )
    assert fallback.isdisjoint(oos), (
        f"cpu_fallback and out_of_scope overlap on {fallback & oos}"
    )


def test_post_register_patches_device(recipe, fake_qwen3_vl_model):
    """``post_register`` patches ``type(model).device`` to ``cpu``."""
    recipe.post_register(fake_qwen3_vl_model)

    cls = type(fake_qwen3_vl_model)
    assert isinstance(cls.device, property), "model.device should be patched to a property"
    assert fake_qwen3_vl_model.device == torch.device("cpu")


def test_post_register_attaches_runtime_config(recipe, fake_qwen3_vl_model):
    """``post_register`` stashes the resolved per-variant TTNN tuning."""
    recipe.post_register(fake_qwen3_vl_model)

    assert hasattr(fake_qwen3_vl_model, "_tt_runtime_config"), (
        "post_register should attach model._tt_runtime_config from lookup_ttnn_tuning"
    )
    cfg = fake_qwen3_vl_model._tt_runtime_config
    assert isinstance(cfg, dict)
    # 2B-Instruct targets a single-chip mesh.
    assert cfg["mesh_shape"] == (1, 1), f"2B should target (1, 1); got {cfg['mesh_shape']}"
    assert cfg["dtype"] == "bfloat16"
    assert "l1_small_size" in cfg


def test_make_kv_cache_is_noop(recipe):
    """CPU-first commit keeps HF ``DynamicCache``; the decorator should install a no-op."""
    result = recipe.make_kv_cache(model=None, device=None)
    assert result is None, (
        f"Qwen3VLRecipe.make_kv_cache should be the @register_recipe no-op "
        f"in the CPU-first commit; got {result!r}"
    )


def test_lookup_ttnn_tuning_fallbacks():
    """``lookup_ttnn_tuning`` resolves checkpoint -> shape -> default ladder."""
    from tt_symbiote.models.qwen3_vl.configuration_qwen3_vl import lookup_ttnn_tuning

    # 1) Exact checkpoint match (2B on N150).
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="Qwen/Qwen3-VL-2B-Instruct",
            text_config=SimpleNamespace(num_hidden_layers=28, hidden_size=2048),
            vision_config=SimpleNamespace(),
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["mesh_shape"] == (1, 1)

    # 2) Exact checkpoint match (32B on T3K).
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="Qwen/Qwen3-VL-32B-Instruct",
            text_config=SimpleNamespace(num_hidden_layers=64, hidden_size=5120),
            vision_config=SimpleNamespace(),
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["mesh_shape"] == (1, 8), "32B must target the full T3K mesh"

    # 3) Unknown name, but shape matches 2B (28 layers, 2048 hidden).
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="my-org/finetuned-qwen3vl-2b",
            text_config=SimpleNamespace(num_hidden_layers=28, hidden_size=2048),
            vision_config=SimpleNamespace(),
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["mesh_shape"] == (1, 1), "shape-fallback should resolve to 2B tuning"

    # 4) Unknown shape -> permissive default.
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="my-org/exotic-qwen3vl",
            text_config=SimpleNamespace(num_hidden_layers=7, hidden_size=128),
            vision_config=None,
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["hw_verified"] is False
    assert cfg["mesh_shape"] == (1, 1), "default tuning falls back to single-chip mesh"


def test_compatibility_report_consumes_recipe(recipe, fake_qwen3_vl_model):
    """End-to-end: ``compatibility.report`` reads the recipe lists.

    Builds the report against the fake model, confirms the design-time
    section matches the recipe, and confirms the runtime section is
    empty (this test never runs a forward).
    """
    from tt_symbiote.utils.compatibility import report, reset_runtime_observations

    reset_runtime_observations()
    out = report(fake_qwen3_vl_model)

    assert out["model_class"] == "_FakeQwen3VLForConditionalGeneration", (
        "the fake stand-in shadows the real class on purpose to keep the "
        "test hardware-free; ``report`` must still surface that class name"
    )
    # The registry is keyed by the *real* HF class name, so the fake
    # model resolves to no recipe and the design-time lists come back
    # empty. That's the documented behaviour for unrecognised classes.
    assert out["design_time"]["tt_implemented"] == []
    assert out["design_time"]["cpu_fallback"] == []
    assert out["design_time"]["out_of_scope"] == []
    assert out["runtime_observed"]["by_module"] == {}
    assert out["summary"]["runtime_fallback_count"] == 0


def test_compatibility_report_for_registered_class():
    """For the actual ``Qwen3VLForConditionalGeneration`` class, the report uses recipe lists.

    Constructs an object whose ``type(...).__name__`` matches the
    registry key; ``report`` should then surface the recipe's
    ``cpu_fallback`` list verbatim.
    """
    from tt_symbiote.utils.compatibility import report, reset_runtime_observations

    class Qwen3VLForConditionalGeneration:  # noqa: N801 — mirrors HF name on purpose
        pass

    reset_runtime_observations()
    out = report(Qwen3VLForConditionalGeneration())

    assert out["model_class"] == "Qwen3VLForConditionalGeneration"
    assert out["summary"]["tt_implemented_count"] == 0
    assert out["summary"]["cpu_fallback_count"] >= 16, (
        "the Qwen3-VL recipe should declare at least the vision tower "
        "(7), text decoder (6), and top-level composites (3) under "
        "cpu_fallback — 16 total minimum"
    )
    assert "Qwen3VLVisionModel" in out["design_time"]["cpu_fallback"]
    assert "Qwen3VLForConditionalGeneration" in out["design_time"]["cpu_fallback"]
