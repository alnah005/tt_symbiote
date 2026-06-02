# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Hardware-free unit tests for the Phase 6 ResNet recipe.

Mirrors :mod:`tests.auto.test_ling_recipe`. Verifies the *shape* of
:class:`ResNetRecipe`: that importing the modeling package registers
the recipe under ``"ResNetForImageClassification"``, that
``build_module_dict`` returns the expected six-key flat dict (Option 1
contract), that ``post_register`` both patches HF's ``model.device`` to
``cpu`` and attaches the resolved per-variant tuning to
``model._tt_runtime_config``, and that ``make_kv_cache`` is the
no-op installed by the :func:`register_recipe` decorator (ResNet has no
KV cache).

Actual TTNN execution lives in
``tests/models/resnet/test_modeling_resnet.py`` (hardware).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def recipe():
    """Import the ResNet modeling module (registers recipe) and return the instance."""
    import tt_symbiote.models.resnet  # noqa: F401 — side-effect import
    from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY

    return TT_MODEL_REGISTRY["ResNetForImageClassification"]


@pytest.fixture()
def fake_resnet_model():
    """A minimal stand-in for an HF ``ResNetForImageClassification``.

    ``post_register`` mutates ``type(model)`` (via
    ``type(model).device = property(...)``), so each test uses a fresh
    subclass to avoid cross-test contamination.
    """

    class _FakeResNetForImageClassification(nn.Module):
        pass

    instance = _FakeResNetForImageClassification()
    instance.config = SimpleNamespace(
        _name_or_path="microsoft/resnet-50",
        depths=(3, 4, 6, 3),
        layer_type="bottleneck",
        num_labels=1000,
    )
    return instance


def test_recipe_registered(recipe):
    """Importing ``tt_symbiote.models.resnet`` populates the registry."""
    assert recipe is not None
    assert callable(recipe.build_module_dict)
    assert callable(recipe.post_register)
    assert callable(recipe.make_kv_cache)


def test_build_module_dict_shape(recipe, fake_resnet_model):
    """Option 1 contract: ``build_module_dict`` returns a single flat dict."""
    module_dict = recipe.build_module_dict(fake_resnet_model)

    assert isinstance(
        module_dict, dict
    ), f"build_module_dict must return a flat dict (Option 1), got {type(module_dict).__name__}"
    assert not isinstance(module_dict, list)

    for key, value in module_dict.items():
        assert isinstance(key, type), f"key {key!r} should be a class, got {type(key).__name__}"
        assert isinstance(value, type), f"value {value!r} should be a class, got {type(value).__name__}"


def test_build_module_dict_covers_required_swaps(recipe, fake_resnet_model):
    """The recipe maps the required HF building blocks to TTNN equivalents.

    Per the plan + the NHWC pooling head adjustment landed during Phase 6
    hardware bring-up: ``ResNetConvLayer``, ``ResNetShortCut``,
    ``ResNetBasicLayer``, ``ResNetBottleNeckLayer``,
    ``ResNetEmbeddings``, ``nn.AdaptiveAvgPool2d`` (HF's
    ``ResNetModel.pooler`` — it has to consume an NHWC tensor produced
    by the encoder and emit NCHW for the Flatten+Linear head), and
    ``nn.Linear`` (classifier).
    """
    from transformers.models.resnet.modeling_resnet import (
        ResNetBasicLayer,
        ResNetBottleNeckLayer,
        ResNetConvLayer,
        ResNetEmbeddings,
        ResNetShortCut,
    )

    module_dict = recipe.build_module_dict(fake_resnet_model)
    keys = set(module_dict.keys())

    assert ResNetConvLayer in keys, "ResNetConvLayer must be swapped"
    assert ResNetShortCut in keys, "ResNetShortCut must be swapped"
    assert ResNetBasicLayer in keys, "ResNetBasicLayer must be swapped (resnet-18/34)"
    assert ResNetBottleNeckLayer in keys, "ResNetBottleNeckLayer must be swapped (resnet-50/101/152)"
    assert ResNetEmbeddings in keys, "ResNetEmbeddings must be swapped (stem)"
    assert nn.AdaptiveAvgPool2d in keys, (
        "nn.AdaptiveAvgPool2d must be swapped (HF's ResNetModel.pooler — needs an "
        "NHWC-aware replacement so the global mean reduces the right axes)"
    )
    assert nn.Linear in keys, "nn.Linear must be swapped (classifier head)"
    assert len(keys) == 7, f"expected exactly 7 top-level swaps; got {len(keys)}: {keys}"


def test_post_register_patches_device(recipe, fake_resnet_model):
    """``post_register`` installs ``type(model).device = property(cpu)``.

    Same rationale as Phase 5: after replacement no ``nn.Module``
    parameters remain, so HF's default
    ``model.device = next(self.parameters()).device`` would raise
    ``StopIteration``. The recipe patches the property to a constant
    ``cpu`` so HF generation/forward code paths still work.
    """
    recipe.post_register(fake_resnet_model)

    cls = type(fake_resnet_model)
    assert isinstance(cls.device, property), "model.device should be patched to a property"
    assert fake_resnet_model.device == torch.device("cpu")


def test_post_register_attaches_runtime_config(recipe, fake_resnet_model):
    """``post_register`` stashes the per-variant TTNN tuning on the model."""
    recipe.post_register(fake_resnet_model)

    assert hasattr(
        fake_resnet_model, "_tt_runtime_config"
    ), "post_register should attach model._tt_runtime_config from lookup_ttnn_tuning"
    cfg = fake_resnet_model._tt_runtime_config
    assert isinstance(cfg, dict), f"_tt_runtime_config must be a dict; got {type(cfg).__name__}"
    assert "l1_small_size" in cfg, "_tt_runtime_config should expose l1_small_size"
    assert cfg.get("hw_verified") is True, "microsoft/resnet-50 should resolve to the hw-verified tuning entry"


def test_make_kv_cache_is_noop(recipe):
    """ResNet has no KV cache; the decorator should have installed a no-op."""
    result = recipe.make_kv_cache(model=None, device=None)
    assert result is None, (
        f"ResNetRecipe.make_kv_cache should be the @register_recipe no-op " f"(returns None); got {result!r}"
    )


def test_lookup_ttnn_tuning_fallbacks():
    """``lookup_ttnn_tuning`` resolves checkpoint -> shape -> default ladder."""
    from tt_symbiote.models.resnet.configuration_resnet import lookup_ttnn_tuning

    # 1) Exact checkpoint match.
    model = SimpleNamespace(
        config=SimpleNamespace(_name_or_path="microsoft/resnet-50", depths=(3, 4, 6, 3), layer_type="bottleneck")
    )
    assert lookup_ttnn_tuning(model)["hw_verified"] is True

    # 2) Unknown name, but shape matches a canonical variant.
    model = SimpleNamespace(
        config=SimpleNamespace(_name_or_path="my-org/finetuned-r50", depths=(3, 4, 6, 3), layer_type="bottleneck")
    )
    assert lookup_ttnn_tuning(model)["hw_verified"] is True  # resolved to resnet-50

    # 3) Unknown shape -> permissive default.
    model = SimpleNamespace(
        config=SimpleNamespace(_name_or_path="my-org/exotic", depths=(1, 1, 1, 1), layer_type="bottleneck")
    )
    cfg = lookup_ttnn_tuning(model)
    assert cfg["hw_verified"] is False, "unknown variants must not claim hw_verified"
    assert "l1_small_size" in cfg
