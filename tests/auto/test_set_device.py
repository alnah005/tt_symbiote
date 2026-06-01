# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""``set_device`` contract tests (no real hardware required).

Verifies the four pieces of the §4.4 contract:

1. ``_device`` is set on every TTNNModule after ``set_device``.
2. ``preprocess_weights`` and ``move_weights_to_device`` are called
   (subsumed loop per OQ-3).
3. Modules with ``@run_on_devices`` declarations are proactively swapped
   to their ``_fallback_torch_layer`` when the active arch is not allowed,
   and a warning is emitted.
4. Invoking a TTNNModule before ``set_device`` raises with a message that
   mentions ``set_device``.
"""

import os
import warnings

import pytest
import torch
import torch.nn as nn

from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices
from tt_symbiote.utils.device_management import set_device


class _StubMeshDevice:
    """Minimal ``ttnn.MeshDevice`` stand-in for tests."""

    def __init__(self, num_devices: int = 1):
        self._n = num_devices

    def get_num_devices(self) -> int:
        return self._n


class _CountingTTNNModule(TTNNModule):
    """TTNNModule that records preprocess/move calls and exposes a forward."""

    def __init__(self):
        super().__init__()
        self.preprocess_calls = 0
        self.move_calls = 0

    def preprocess_weights_impl(self):
        self.preprocess_calls += 1
        return super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.move_calls += 1
        return super().move_weights_to_device_impl()

    def forward(self, *args, **kwargs):
        return args[0] if args else None


class _T3KOnlyTTNNModule(TTNNModule):
    """TTNN module that declares T3K-only support via @run_on_devices."""

    def __init__(self):
        super().__init__()

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x):
        return x


class _TTNNContainer(nn.Module):
    """Tiny nn.Module that owns a TTNNModule as a plain attribute.

    ``TTNNModule`` is intentionally not an ``nn.Module`` subclass (so
    ``nn.Sequential`` rejects it). For test plumbing we hold the TTNN
    instance on a normal attribute so the attr-walking branch of
    ``set_device`` finds and binds / swaps it.
    """

    def __init__(self, child: TTNNModule):
        super().__init__()
        # Plain attribute (NOT registered in self._modules).
        self.tt_child = child


@pytest.fixture
def reset_mesh_env(monkeypatch):
    """Always clear MESH_DEVICE before each test for hermetic arch resolution."""
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    yield monkeypatch


def test_set_device_binds_device_and_runs_weight_prep(reset_mesh_env):
    mod = _CountingTTNNModule()
    mod._fallback_torch_layer = nn.Identity()  # so the contract is happy
    device = _StubMeshDevice(num_devices=1)

    set_device(mod, device, dump_visualization=False)

    assert mod._device is device
    assert mod._tt_symbiote_device_set is True
    assert mod.preprocess_calls == 1
    assert mod.move_calls == 1


def test_set_device_marker_set_on_nn_module_root(reset_mesh_env):
    child = _CountingTTNNModule()
    child._fallback_torch_layer = nn.Identity()
    parent = _TTNNContainer(child)
    device = _StubMeshDevice(num_devices=1)

    set_device(parent, device, dump_visualization=False, register_forward_hook=False)

    assert parent._tt_symbiote_device_set is True
    assert child._tt_symbiote_device_set is True


def test_set_device_no_arch_constraint_no_swap(reset_mesh_env):
    child = _CountingTTNNModule()
    child._fallback_torch_layer = nn.Identity()
    parent = _TTNNContainer(child)
    device = _StubMeshDevice(num_devices=1)
    set_device(parent, device, dump_visualization=False, register_forward_hook=False)
    assert parent.tt_child is child, "no @run_on_devices => should not swap"


def test_run_on_devices_decorator_stamps_allowed_archs():
    # The decorator stores allowed_archs on the function, so set_device can
    # introspect without invoking the wrapper.
    forward = _T3KOnlyTTNNModule.forward
    assert hasattr(forward, "__tt_allowed_archs__"), "@run_on_devices must stamp __tt_allowed_archs__"
    assert DeviceArch.T3K in forward.__tt_allowed_archs__


def test_set_device_swaps_unsupported_arch_module(reset_mesh_env):
    reset_mesh_env.setenv("MESH_DEVICE", "N150")  # not T3K

    child = _T3KOnlyTTNNModule()
    fallback = nn.Identity()
    child._fallback_torch_layer = fallback
    parent = _TTNNContainer(child)
    device = _StubMeshDevice(num_devices=1)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        set_device(parent, device, dump_visualization=False, register_forward_hook=False)

    assert parent.tt_child is fallback, "module should be swapped to fallback when arch unsupported"
    messages = [str(w.message) for w in captured]
    assert any("not supported" in m for m in messages), f"Expected 'not supported' warning; got {messages}"


def test_set_device_keeps_supported_arch_module(reset_mesh_env):
    reset_mesh_env.setenv("MESH_DEVICE", "T3K")  # matches the @run_on_devices decl

    child = _T3KOnlyTTNNModule()
    child._fallback_torch_layer = nn.Identity()
    parent = _TTNNContainer(child)
    device = _StubMeshDevice(num_devices=1)
    set_device(parent, device, dump_visualization=False, register_forward_hook=False)

    assert parent.tt_child is child, "T3K module should remain on T3K device"


def test_module_call_before_set_device_raises_with_set_device_message():
    mod = _CountingTTNNModule()
    mod._fallback_torch_layer = nn.Identity()
    with pytest.raises(AssertionError) as info:
        mod(torch.zeros(1))
    assert "set_device" in str(info.value), f"AssertionError must mention set_device; got: {info.value!r}"
