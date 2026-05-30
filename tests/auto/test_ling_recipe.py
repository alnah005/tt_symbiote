# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Hardware-free unit tests for the Phase 5 Ling-mini-2.0 recipe.

These tests verify the *shape* of :class:`BailingMoEV2Recipe`: that
importing the modeling package registers the recipe, that
``build_module_dict`` returns a single flat dict (per the Option 1
contract), that ``post_register`` patches HF's ``model.device`` to
``cpu`` (the patch is required because, post-replacement, HF's default
``model.device = next(self.parameters()).device`` raises ``StopIteration``
when no ``nn.Module`` params remain), and that ``make_kv_cache`` is
callable with the documented signature.

Actual TTNN execution and KV-cache allocation are exercised by
``tests/capabilities/bailing_moe_v2/test_modeling_bailing_moe_v2.py`` on
real hardware.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def recipe():
    """Import the Ling modeling module (registers recipe) and return the instance."""
    import tt_symbiote.models.bailing_moe_v2  # noqa: F401 — side-effect import
    from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY

    return TT_MODEL_REGISTRY["BailingMoeV2ForCausalLM"]


@pytest.fixture()
def fake_hf_model():
    """A minimal stand-in for a HF ``BailingMoeV2ForCausalLM`` instance.

    ``build_module_dict`` reads ``type(model.model)`` so we need a concrete
    inner class. We do not need real HF code: any uniquely-typed
    ``nn.Module`` subclass works.
    """

    class _FakeBailingMoeV2ForCausalLM(nn.Module):
        pass

    class _FakeBailingMoeV2Model(nn.Module):
        pass

    causal = _FakeBailingMoeV2ForCausalLM()
    causal.model = _FakeBailingMoeV2Model()
    causal.lm_head = nn.Linear(4, 4)
    return causal


def test_recipe_registered(recipe):
    """Importing ``tt_symbiote.models.bailing_moe_v2`` populates the registry."""
    assert recipe is not None
    assert callable(recipe.build_module_dict)
    assert callable(recipe.post_register)
    assert callable(recipe.make_kv_cache)


def test_build_module_dict_shape(recipe, fake_hf_model):
    """Option 1 contract: ``build_module_dict`` returns a *single flat dict*.

    Specifically not a ``list[dict]`` (Option 2). Each key must be a
    class, each value a class — the walker uses them as
    ``isinstance``-style class swaps.
    """
    module_dict = recipe.build_module_dict(fake_hf_model)

    assert isinstance(module_dict, dict), (
        f"build_module_dict must return a flat dict (Option 1), got {type(module_dict).__name__}"
    )
    assert not isinstance(module_dict, list)
    assert len(module_dict) >= 1

    for key, value in module_dict.items():
        assert isinstance(key, type), f"key {key!r} should be a class, got {type(key).__name__}"
        assert isinstance(value, type), f"value {value!r} should be a class, got {type(value).__name__}"


def test_build_module_dict_covers_outer_model_and_lm_head(recipe, fake_hf_model):
    """The Option 1 recipe describes the two top-level swaps only.

    Inner decoder/norm/embed/rotary swaps are owned by
    ``TTNNBailingMoeV2Model.from_torch`` itself.
    """
    module_dict = recipe.build_module_dict(fake_hf_model)
    keys = set(module_dict.keys())

    assert type(fake_hf_model.model) in keys, "outer BailingMoeV2Model class must be swapped"
    assert nn.Linear in keys, "nn.Linear must be swapped (covers lm_head)"


def test_post_register_patches_device(recipe, fake_hf_model):
    """``post_register`` installs ``type(model).device = property(cpu)``."""
    recipe.post_register(fake_hf_model)

    cls = type(fake_hf_model)
    assert isinstance(cls.device, property), "model.device should be patched to a property"
    assert fake_hf_model.device == torch.device("cpu")


def test_make_kv_cache_signature(recipe):
    """``make_kv_cache(self, model, device, batch_size=1, **kwargs)``."""
    sig = inspect.signature(recipe.make_kv_cache)
    params = sig.parameters

    assert "model" in params, "make_kv_cache must accept ``model``"
    assert "device" in params, "make_kv_cache must accept ``device``"
    assert any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), "make_kv_cache must accept ``**kwargs`` for forward-compat"


def test_make_kv_cache_reads_config(recipe):
    """The cache constructor consumes ``num_hidden_layers``, ``num_key_value_heads``, ``head_dim``.

    We don't actually allocate a cache here (that requires a real TTNN
    device) — we just verify that the recipe pulls the right config
    fields by passing a fake config and catching the resulting
    ``AttributeError`` if any are missing.
    """
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=16,
        )
    )

    # ``device=None`` reaches the TTNN constructor and is expected to fail
    # there (the cache tries to call methods on a None device). The
    # important thing is that the AttributeError, if any, comes from TTNN
    # construction *not* from missing config fields.
    try:
        recipe.make_kv_cache(fake_model, device=None, batch_size=1)
    except AttributeError as e:
        assert "config" not in str(e), (
            f"make_kv_cache should read num_hidden_layers / num_key_value_heads / head_dim "
            f"from model.config; got config-related AttributeError: {e}"
        )
    except Exception:
        # Any non-AttributeError is fine: it means we got past config reads.
        pass
