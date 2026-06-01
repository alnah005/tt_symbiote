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


def test_build_module_dict_wraps_phase8_wave_b(recipe, fake_qwen3_vl_model):
    """Phase 8 Wave B contract: build_module_dict ships the 4-class swap.

    The fake fixture targets the Qwen3-VL-2B variant. Wave B wraps four
    structurally simple classes: vision MLP, vision patch merger, text
    RMSNorm, and text MLP. Heavier classes (attention with varlen RoPE,
    the text decoder layer) remain CPU fallbacks for now.
    """
    module_dict = recipe.build_module_dict(fake_qwen3_vl_model)
    swapped_class_names = {cls.__name__ for cls in module_dict.keys()}
    assert swapped_class_names == {
        "Qwen3VLVisionMLP",
        "Qwen3VLVisionPatchMerger",
        "Qwen3VLTextRMSNorm",
        "Qwen3VLTextMLP",
    }


def test_design_time_coverage_lists_populated(recipe):
    """The four class-name lists must be present and non-trivial.

    Phase 8 ships four TTNN wrappers; ``cpu_fallback``, ``host_glue``,
    and ``out_of_scope`` carry the rest of the design-time coverage
    manifest that the porting skill and migration docs consume.
    """
    assert isinstance(recipe.tt_implemented, list)
    assert isinstance(recipe.cpu_fallback, list)
    assert isinstance(recipe.host_glue, list)
    assert isinstance(recipe.out_of_scope, list)

    for required in (
        "Qwen3VLVisionMLP",
        "Qwen3VLVisionPatchMerger",
        "Qwen3VLTextRMSNorm",
        "Qwen3VLTextMLP",
    ):
        assert required in recipe.tt_implemented, (
            f"{required} must be declared in tt_implemented (Phase 8 Wave B)"
        )

    # Load-bearing classes still deferred to torch: Phase 8 wraps the
    # element-wise compute (RMSNorm/MLP/patch merger) but leaves the
    # attention stacks, DeepStack-aware text decoder, and M-RoPE
    # precompute for later waves.
    for required in (
        "Qwen3VLVisionModel",
        "Qwen3VLVisionBlock",
        "Qwen3VLVisionAttention",
        "Qwen3VLTextModel",
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLTextAttention",
    ):
        assert required in recipe.cpu_fallback, (
            f"{required} must be declared in cpu_fallback so the "
            f"compatibility report flags it as 'expected' rather than "
            f"a regression when the demo runs."
        )

    for required in ("Qwen3VLModel", "Qwen3VLForConditionalGeneration"):
        assert required in recipe.host_glue, (
            f"{required} should be in host_glue (DeepStack dispatch / "
            f"generation orchestration — no FLOPs to accelerate)."
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

    Overlap would make the recipe's coverage manifest ambiguous (does
    the class count as TT-accelerated, an expected fallback, or pure
    host glue?). Catch the accident in CI before it ships.
    """
    impl = set(recipe.tt_implemented)
    fallback = set(recipe.cpu_fallback)
    host_glue = set(recipe.host_glue)
    oos = set(recipe.out_of_scope)

    assert impl.isdisjoint(fallback), (
        f"tt_implemented and cpu_fallback overlap on {impl & fallback}"
    )
    assert impl.isdisjoint(host_glue), (
        f"tt_implemented and host_glue overlap on {impl & host_glue}"
    )
    assert impl.isdisjoint(oos), (
        f"tt_implemented and out_of_scope overlap on {impl & oos}"
    )
    assert fallback.isdisjoint(host_glue), (
        f"cpu_fallback and host_glue overlap on {fallback & host_glue}"
    )
    assert fallback.isdisjoint(oos), (
        f"cpu_fallback and out_of_scope overlap on {fallback & oos}"
    )
    assert host_glue.isdisjoint(oos), (
        f"host_glue and out_of_scope overlap on {host_glue & oos}"
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


def test_compatibility_report_shape(recipe, fake_qwen3_vl_model):
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
    out = report(fake_qwen3_vl_model)

    assert out["model_class"] == "_FakeQwen3VLForConditionalGeneration", (
        "the fake stand-in shadows the real class on purpose to keep the "
        "test hardware-free; ``report`` must still surface that class name"
    )
    assert "design_time" not in out, (
        "Phase 8.5 dropped the design_time block from the runtime artefact"
    )
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
    """For the registered ``Qwen3VLForConditionalGeneration``, the report stays runtime-only.

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

    class Qwen3VLForConditionalGeneration:  # noqa: N801 — mirrors HF name on purpose
        pass

    reset_runtime_observations()
    reset_swapped_registry()
    out = report(Qwen3VLForConditionalGeneration())

    assert out["model_class"] == "Qwen3VLForConditionalGeneration"
    assert "design_time" not in out
    assert out["regressions"] == []
    assert out["ttnn_swap_skipped"] is False
