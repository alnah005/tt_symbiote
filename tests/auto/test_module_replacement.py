# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""``register_modules`` swaps a PyTorch sub-module; the deprecated alias warns."""

import warnings

import pytest
import torch
import torch.nn as nn

from tt_symbiote.core.module import StatelessTTNNModule, TTNNModule
from tt_symbiote.utils.module_replacement import register_module_replacement_dict, register_modules


class _TTNNLinear(StatelessTTNNModule):
    """Tiny TTNNModule that wraps an nn.Linear and remembers it for verification."""

    def __init__(self):
        super().__init__()

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        instance = super().from_torch(torch_layer, *args, **kwargs)
        instance.was_built_from = torch_layer
        return instance


def _build_tiny_model():
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def test_register_modules_replaces_target_class():
    model = _build_tiny_model()
    replacements = register_modules(model, {nn.Linear: _TTNNLinear})
    # Two Linears in the model => two TTNN modules returned + spliced in place.
    assert len(replacements) == 2
    assert isinstance(model[0], _TTNNLinear)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], _TTNNLinear)
    # _fallback_torch_layer points at the original PyTorch instance.
    assert isinstance(model[0]._fallback_torch_layer, nn.Linear)


def test_register_modules_no_op_when_dict_empty():
    model = _build_tiny_model()
    replacements = register_modules(model, {})
    assert replacements == {}
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[2], nn.Linear)


def test_register_modules_exclude_replacement_skips_named_modules():
    model = nn.Sequential(nn.Linear(4, 4))
    model_names = dict(model.named_modules())
    # The Linear's named_modules() name is "0".
    assert "0" in model_names
    replacements = register_modules(model, {nn.Linear: _TTNNLinear}, exclude_replacement={"0"})
    assert replacements == {}
    assert isinstance(model[0], nn.Linear)


def test_deprecated_alias_emits_warning_and_delegates():
    model = _build_tiny_model()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        replacements = register_module_replacement_dict(model, {nn.Linear: _TTNNLinear})
    # Warning was emitted and the swap still happened.
    assert any(
        issubclass(w.category, DeprecationWarning) and "register_modules" in str(w.message) for w in captured
    ), f"Expected DeprecationWarning mentioning register_modules; got {[str(w.message) for w in captured]}"
    assert len(replacements) == 2
    assert isinstance(model[0], _TTNNLinear)
    assert isinstance(model[2], _TTNNLinear)
