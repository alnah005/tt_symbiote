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


def test_build_module_dict_wraps_phase8_wave_a(recipe, fake_gemma4_model):
    """Phase 8 contract: E2B build_module_dict ships the 5-class Wave A swap.

    The fake fixture targets the E2B variant (35 layers, vision tower
    present, no audio config). For that footprint the chip-budget gate
    is *not* expected to fire, so ``build_module_dict`` returns the
    full Wave A mapping: RMSNorm, scaled word embedding, text MLP,
    vision MLP, multimodal embedder. The gate-fires case (31B /
    26B-A4B) is covered separately.
    """
    module_dict = recipe.build_module_dict(fake_gemma4_model)

    assert module_dict, (
        "Phase 8 E2B should not be gated; got an empty build_module_dict, "
        "which suggests the budget/MoE gate misclassified the variant"
    )
    swapped_class_names = {cls.__name__ for cls in module_dict.keys()}
    assert swapped_class_names == {
        "Gemma4RMSNorm",
        "Gemma4TextScaledWordEmbedding",
        "Gemma4TextMLP",
        "Gemma4VisionMLP",
        "Gemma4MultimodalEmbedder",
    }


def test_design_time_coverage_lists_populated(recipe):
    """The four class-name lists must be present and non-trivial.

    Phase 8 ships five TTNN wrappers; ``cpu_fallback``, ``host_glue``,
    and ``out_of_scope`` carry the rest of the design-time coverage
    manifest that the budget gate and porting skill consume.
    """
    assert isinstance(recipe.tt_implemented, list)
    assert isinstance(recipe.cpu_fallback, list)
    assert isinstance(recipe.host_glue, list)
    assert isinstance(recipe.out_of_scope, list)

    for required in (
        "Gemma4RMSNorm",
        "Gemma4TextScaledWordEmbedding",
        "Gemma4TextMLP",
        "Gemma4VisionMLP",
        "Gemma4MultimodalEmbedder",
    ):
        assert required in recipe.tt_implemented, f"{required} must be declared in tt_implemented (Phase 8 Wave A)"

    # Load-bearing classes still deferred to torch: Phase 8 wraps the
    # element-wise compute (RMSNorm/MLP/embeddings) but leaves the
    # attention stacks and the top-level orchestration (text + vision
    # encoder layers, model containers, dual RoPE, vision attention)
    # for later waves.
    for required in (
        "Gemma4VisionModel",
        "Gemma4VisionEncoderLayer",
        "Gemma4VisionAttention",
        "Gemma4TextModel",
        "Gemma4TextDecoderLayer",
        "Gemma4TextAttention",
    ):
        assert required in recipe.cpu_fallback, (
            f"{required} must be declared in cpu_fallback so the "
            f"compatibility report flags it as 'expected' rather than "
            f"a regression when the demo runs."
        )

    for required in ("Gemma4Model", "Gemma4ForConditionalGeneration"):
        assert required in recipe.host_glue, (
            f"{required} should be in host_glue (orchestration / " f"output dataclass — no FLOPs to accelerate)."
        )

    for required in (
        "Gemma4AudioModel",
        "Gemma4ForCausalLM",
    ):
        assert required in recipe.out_of_scope, (
            f"{required} should be in out_of_scope (not exercised by " f"the image-text-to-text demo)."
        )


def test_design_time_lists_are_disjoint(recipe):
    """A class must not appear in two coverage lists simultaneously.

    Overlap would make the recipe's coverage manifest ambiguous (does
    the class count as TT-accelerated, an expected fallback, or pure
    host glue?). Catch the accident in CI before it ships.
    """
    impl = set(recipe.tt_implemented)
    fallback = set(recipe.cpu_fallback)
    host_glue = set(recipe.host_glue)
    oos = set(recipe.out_of_scope)

    assert impl.isdisjoint(fallback), f"tt_implemented and cpu_fallback overlap on {impl & fallback}"
    assert impl.isdisjoint(host_glue), f"tt_implemented and host_glue overlap on {impl & host_glue}"
    assert impl.isdisjoint(oos), f"tt_implemented and out_of_scope overlap on {impl & oos}"
    assert fallback.isdisjoint(host_glue), f"cpu_fallback and host_glue overlap on {fallback & host_glue}"
    assert fallback.isdisjoint(oos), f"cpu_fallback and out_of_scope overlap on {fallback & oos}"
    assert host_glue.isdisjoint(oos), f"host_glue and out_of_scope overlap on {host_glue & oos}"


def test_post_register_patches_device(recipe, fake_gemma4_model):
    """``post_register`` patches ``type(model).device`` to ``cpu`` (Phase 5/6 convention)."""
    recipe.post_register(fake_gemma4_model)

    cls = type(fake_gemma4_model)
    assert isinstance(cls.device, property), "model.device should be patched to a property"
    assert fake_gemma4_model.device == torch.device("cpu")


def test_post_register_attaches_runtime_config(recipe, fake_gemma4_model):
    """``post_register`` stashes the resolved per-variant TTNN tuning."""
    recipe.post_register(fake_gemma4_model)

    assert hasattr(
        fake_gemma4_model, "_tt_runtime_config"
    ), "post_register should attach model._tt_runtime_config from lookup_ttnn_tuning"
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
        f"Gemma4Recipe.make_kv_cache should be the @register_recipe no-op " f"in Phase 7; got {result!r}"
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


def test_compatibility_report_shape(recipe, fake_gemma4_model):
    """End-to-end: ``compatibility.report`` returns the Phase 8.5 runtime shape.

    The fake model never runs a forward and ``set_device`` is not
    called, so every observation ledger is empty. The test pins the
    new schema: no ``design_time`` key, ``modules_swapped`` and
    ``runtime_observed`` sub-dicts present, ``regressions`` empty.
    """
    from tt_symbiote.utils.compatibility import (
        report,
        reset_runtime_observations,
        reset_swapped_registry,
    )

    reset_runtime_observations()
    reset_swapped_registry()
    out = report(fake_gemma4_model)

    assert out["model_class"] == "_FakeGemma4ForConditionalGeneration", (
        "the fake stand-in shadows the real class on purpose to keep the "
        "test hardware-free; ``report`` must still surface that class name"
    )
    assert "design_time" not in out, "Phase 8.5 dropped the design_time block from the runtime artefact"
    assert out["modules_swapped"] == {"by_class": {}, "by_module": {}}
    assert out["runtime_observed"] == {
        "successes_by_class": {},
        "fallbacks_by_class": {},
        "fallbacks_by_module": {},
    }
    assert out["regressions"] == []
    assert out["summary"]["modules_swapped_count"] == 0
    assert out["summary"]["runtime_fallbacks"] == 0


def test_compatibility_report_for_registered_class():
    """For the registered ``Gemma4ForConditionalGeneration``, the report stays runtime-only.

    Even when the recipe lookup succeeds (so the report could in
    principle compute regressions against ``cpu_fallback``), the JSON
    must still be observational only — ``cpu_fallback`` is consumed
    internally to compute ``regressions`` but is never serialized.
    """
    from tt_symbiote.utils.compatibility import (
        report,
        reset_runtime_observations,
        reset_swapped_registry,
    )

    class Gemma4ForConditionalGeneration:  # noqa: N801 — mirrors HF name on purpose
        pass

    reset_runtime_observations()
    reset_swapped_registry()
    out = report(Gemma4ForConditionalGeneration())

    assert out["model_class"] == "Gemma4ForConditionalGeneration"
    assert "design_time" not in out
    assert out["regressions"] == []
    assert out["ttnn_swap_skipped"] is False
