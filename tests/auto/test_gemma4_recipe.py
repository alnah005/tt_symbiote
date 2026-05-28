# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Hardware-free unit tests for the Phase 7 Gemma-4 VLM recipe.

Mirrors :mod:`tests.auto.test_resnet_recipe` and
:mod:`tests.auto.test_ling_recipe`. Verifies the *shape* of
:class:`Gemma4Recipe`: that importing the modeling package registers
the recipe under ``"Gemma4ForConditionalGeneration"``, that the
recipe's CPU-first contract holds (``build_module_dict`` returns
``{}``), that the three design-time coverage lists are populated and
disjoint, that ``post_register`` patches the device and attaches the
TTNN runtime config, and that
:func:`tt_symbiote.compatibility.report` reads the recipe correctly.

Actual model execution (image + text -> "dog") lives in
``examples/e2e/run_gemma4_e2b.py`` (E2B on N150) and
``examples/e2e/run_gemma4_31b.py`` (31B on T3K). Those scripts download
multi-GB weights and require live hardware, so they intentionally do
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
    import tt_symbiote.models.gemma4  # noqa: F401 — side-effect import
    from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY

    return TT_MODEL_REGISTRY["Gemma4ForConditionalGeneration"]


@pytest.fixture()
def fake_gemma4_model():
    """A minimal stand-in for an HF ``Gemma4ForConditionalGeneration``.

    ``post_register`` patches ``type(model).device``; we use a fresh
    subclass per test so the property override does not leak between
    tests (same pattern as the ResNet recipe tests).
    """

    class _FakeGemma4ForConditionalGeneration(nn.Module):
        pass

    instance = _FakeGemma4ForConditionalGeneration()
    instance.config = SimpleNamespace(
        _name_or_path="google/gemma-4-E2B-it",
        text_config=SimpleNamespace(num_hidden_layers=35),
        vision_config=SimpleNamespace(hidden_size=1152),
        audio_config=SimpleNamespace(hidden_size=1024),
    )
    return instance


def test_recipe_registered(recipe):
    """Importing ``tt_symbiote.models.gemma4`` populates the registry."""
    assert recipe is not None
    assert callable(recipe.build_module_dict)
    assert callable(recipe.post_register)
    assert callable(recipe.make_kv_cache)


def test_build_module_dict_is_empty_cpu_first(recipe, fake_gemma4_model):
    """Phase 7 contract: ``build_module_dict`` returns ``{}`` (CPU-first port).

    The empty dict is *the* signal that the recipe is in its initial
    CPU-first phase: nothing gets swapped, all HF modules stay as
    PyTorch, and ``set_device`` reduces to attaching
    ``_tt_runtime_config``. Future commits that land TTNN wrappers
    will populate this dict and shrink ``cpu_fallback``.
    """
    module_dict = recipe.build_module_dict(fake_gemma4_model)
    assert module_dict == {}, (
        f"Phase 7 ships a CPU-first port; build_module_dict must return {{}} "
        f"until TTNN wrappers land. Got {module_dict!r}."
    )


def test_design_time_coverage_lists_populated(recipe):
    """The three class-name lists must be present and non-trivial.

    ``tt_implemented`` is empty in Phase 7 (CPU-first), but
    ``cpu_fallback`` and ``out_of_scope`` carry the design-time
    coverage manifest that :func:`compatibility.report` surfaces.
    """
    assert isinstance(recipe.tt_implemented, list)
    assert isinstance(recipe.cpu_fallback, list)
    assert isinstance(recipe.out_of_scope, list)

    assert recipe.tt_implemented == [], (
        "Phase 7 ships zero TTNN wrappers; tt_implemented must be empty"
    )

    # The exhaustive list covers the text path, vision path, multimodal
    # projection and both top-level composites. The literal count is
    # less interesting than the requirement that key classes are
    # present, so we assert membership on the load-bearing entries
    # instead of a brittle ``len()`` check.
    for required in (
        "Gemma4VisionModel",
        "Gemma4VisionEncoderLayer",
        "Gemma4MultimodalEmbedder",
        "Gemma4TextModel",
        "Gemma4TextDecoderLayer",
        "Gemma4ForConditionalGeneration",
    ):
        assert required in recipe.cpu_fallback, (
            f"{required} must be declared in cpu_fallback so the "
            f"compatibility report flags it as 'expected' rather than "
            f"'unexpected' when the demo runs."
        )

    for required in (
        "Gemma4AudioModel",
        "Gemma4ForCausalLM",
    ):
        assert required in recipe.out_of_scope, (
            f"{required} should be in out_of_scope (not exercised by "
            f"the image-text-to-text demo)."
        )


def test_design_time_lists_are_disjoint(recipe):
    """A class must not appear in two coverage lists simultaneously.

    Overlap would make ``compatibility.report`` ambiguous (does the
    class count as TT-accelerated or as a known fallback?). Catch the
    accident in CI before it ships.
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


def test_post_register_patches_device(recipe, fake_gemma4_model):
    """``post_register`` patches ``type(model).device`` to ``cpu`` (Phase 5/6 convention)."""
    recipe.post_register(fake_gemma4_model)

    cls = type(fake_gemma4_model)
    assert isinstance(cls.device, property), "model.device should be patched to a property"
    assert fake_gemma4_model.device == torch.device("cpu")


def test_post_register_attaches_runtime_config(recipe, fake_gemma4_model):
    """``post_register`` stashes the resolved per-variant TTNN tuning."""
    recipe.post_register(fake_gemma4_model)

    assert hasattr(fake_gemma4_model, "_tt_runtime_config"), (
        "post_register should attach model._tt_runtime_config from lookup_ttnn_tuning"
    )
    cfg = fake_gemma4_model._tt_runtime_config
    assert isinstance(cfg, dict)
    # E2B targets a single-chip mesh.
    assert cfg["mesh_shape"] == (1, 1), f"E2B should target (1, 1); got {cfg['mesh_shape']}"
    assert cfg["dtype"] == "bfloat16"
    assert "l1_small_size" in cfg


def test_make_kv_cache_is_noop(recipe):
    """Phase 7 keeps HF ``DynamicCache``; the decorator should install a no-op."""
    result = recipe.make_kv_cache(model=None, device=None)
    assert result is None, (
        f"Gemma4Recipe.make_kv_cache should be the @register_recipe no-op "
        f"in Phase 7; got {result!r}"
    )


def test_lookup_ttnn_tuning_fallbacks():
    """``lookup_ttnn_tuning`` resolves checkpoint -> shape -> default ladder."""
    from tt_symbiote.models.gemma4.configuration_gemma4 import lookup_ttnn_tuning

    # 1) Exact checkpoint match (E2B on N150).
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="google/gemma-4-E2B-it",
            text_config=SimpleNamespace(num_hidden_layers=35),
            vision_config=SimpleNamespace(),
            audio_config=SimpleNamespace(),
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["mesh_shape"] == (1, 1)

    # 2) Exact checkpoint match (31B on T3K).
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="google/gemma-4-31B-it",
            text_config=SimpleNamespace(num_hidden_layers=60),
            vision_config=SimpleNamespace(),
            audio_config=None,
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["mesh_shape"] == (1, 8), "31B must target the full T3K mesh"

    # 3) Unknown name, but shape matches a canonical variant (60 text
    # layers + vision + no audio -> 31B).
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="my-org/finetuned-gemma4-31b",
            text_config=SimpleNamespace(num_hidden_layers=60),
            vision_config=SimpleNamespace(),
            audio_config=None,
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["mesh_shape"] == (1, 8), "shape-fallback should resolve to 31B tuning"

    # 4) Unknown shape -> permissive default.
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="my-org/exotic",
            text_config=SimpleNamespace(num_hidden_layers=7),
            vision_config=None,
            audio_config=None,
        )
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["hw_verified"] is False
    assert cfg["mesh_shape"] == (1, 1), "default tuning falls back to single-chip mesh"


def test_compatibility_report_consumes_recipe(recipe, fake_gemma4_model):
    """End-to-end: ``compatibility.report`` reads the recipe lists.

    Builds the report against the fake model, confirms the design-time
    section matches the recipe, and confirms the runtime section is
    empty (this test never runs a forward).
    """
    from tt_symbiote.utils.compatibility import report, reset_runtime_observations

    reset_runtime_observations()
    out = report(fake_gemma4_model)

    assert out["model_class"] == "_FakeGemma4ForConditionalGeneration", (
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
    """For the actual ``Gemma4ForConditionalGeneration`` class, the report uses recipe lists.

    Constructs an object whose ``type(...).__name__`` matches the
    registry key; ``report`` should then surface the recipe's
    ``cpu_fallback`` list verbatim.
    """
    from tt_symbiote.utils.compatibility import report, reset_runtime_observations

    class Gemma4ForConditionalGeneration:  # noqa: N801 — mirrors HF name on purpose
        pass

    reset_runtime_observations()
    out = report(Gemma4ForConditionalGeneration())

    assert out["model_class"] == "Gemma4ForConditionalGeneration"
    assert out["summary"]["tt_implemented_count"] == 0
    assert out["summary"]["cpu_fallback_count"] >= 19, (
        "the Gemma-4 recipe should declare at least the vision tower, "
        "multimodal projection, and full text decoder under cpu_fallback"
    )
    assert "Gemma4VisionModel" in out["design_time"]["cpu_fallback"]
    assert "Gemma4ForConditionalGeneration" in out["design_time"]["cpu_fallback"]
