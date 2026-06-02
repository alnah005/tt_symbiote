# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""``register_recipe`` populates ``TT_MODEL_REGISTRY`` and emits override warnings."""

import warnings

import pytest

from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY, register_recipe


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Don't let one test contaminate the global registry for the next."""
    snapshot = dict(TT_MODEL_REGISTRY)
    yield
    TT_MODEL_REGISTRY.clear()
    TT_MODEL_REGISTRY.update(snapshot)


def test_register_recipe_round_trip():
    @register_recipe(hf_class_name="UnitTestFakeHFModel")
    class _UnitTestRecipe:
        def build_module_dict(self, model):
            return {}

        def post_register(self, model):
            pass

    assert "UnitTestFakeHFModel" in TT_MODEL_REGISTRY
    instance = TT_MODEL_REGISTRY["UnitTestFakeHFModel"]
    assert instance.build_module_dict(object()) == {}
    instance.post_register(object())  # callable + no-op


def test_register_recipe_default_post_register_is_inserted_when_missing():
    @register_recipe(hf_class_name="UnitTestNoPostRegister")
    class _NoPostRegister:
        def build_module_dict(self, model):
            return {"foo": "bar"}

    instance = TT_MODEL_REGISTRY["UnitTestNoPostRegister"]
    assert callable(instance.post_register)
    instance.post_register(object())  # default is a no-op


def test_register_recipe_warns_on_override():
    @register_recipe(hf_class_name="UnitTestOverridden")
    class _RecipeA:
        def build_module_dict(self, model):
            return {"impl": "A"}

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        @register_recipe(hf_class_name="UnitTestOverridden")
        class _RecipeB:
            def build_module_dict(self, model):
                return {"impl": "B"}

    messages = [str(w.message) for w in captured]
    assert any("Overriding existing tt_symbiote recipe" in m for m in messages)
    assert TT_MODEL_REGISTRY["UnitTestOverridden"].build_module_dict(object())["impl"] == "B"


def test_register_recipe_rejects_recipe_missing_build_module_dict():
    with pytest.raises(TypeError):

        @register_recipe(hf_class_name="UnitTestBadRecipe")
        class _Bad:
            pass


def test_register_recipe_rejects_empty_name():
    with pytest.raises(ValueError):
        register_recipe(hf_class_name="")


def test_recipe_protocol_runtime_checkable():
    from tt_symbiote.models.auto.auto_mappings import Recipe

    class _Conforming:
        def build_module_dict(self, model):
            return {}

        def post_register(self, model):
            pass

    assert isinstance(_Conforming(), Recipe)
