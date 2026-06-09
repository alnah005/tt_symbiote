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

from tt_symbiote.core.module import DeviceArch, StatelessTTNNModule, TTNNModule, run_on_devices
from tt_symbiote.utils.device_management import set_device


class _StubMeshDevice:
    """Minimal ``ttnn.MeshDevice`` stand-in for tests."""

    def __init__(self, num_devices: int = 1):
        self._n = num_devices

    def get_num_devices(self) -> int:
        return self._n


class _CountingTTNNModule(StatelessTTNNModule):
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


class _T3KOnlyTTNNModule(StatelessTTNNModule):
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

    set_device(mod, device)

    assert mod._device is device
    assert mod._tt_symbiote_device_set is True
    assert mod.preprocess_calls == 1
    assert mod.move_calls == 1


def test_set_device_marker_set_on_nn_module_root(reset_mesh_env):
    child = _CountingTTNNModule()
    child._fallback_torch_layer = nn.Identity()
    parent = _TTNNContainer(child)
    device = _StubMeshDevice(num_devices=1)

    set_device(parent, device)

    assert parent._tt_symbiote_device_set is True
    assert child._tt_symbiote_device_set is True


def test_set_device_no_arch_constraint_no_swap(reset_mesh_env):
    child = _CountingTTNNModule()
    child._fallback_torch_layer = nn.Identity()
    parent = _TTNNContainer(child)
    device = _StubMeshDevice(num_devices=1)
    set_device(parent, device)
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
        set_device(parent, device)

    assert parent.tt_child is fallback, "module should be swapped to fallback when arch unsupported"
    messages = [str(w.message) for w in captured]
    assert any("not supported" in m for m in messages), f"Expected 'not supported' warning; got {messages}"


def test_set_device_keeps_supported_arch_module(reset_mesh_env):
    reset_mesh_env.setenv("MESH_DEVICE", "T3K")  # matches the @run_on_devices decl

    child = _T3KOnlyTTNNModule()
    child._fallback_torch_layer = nn.Identity()
    parent = _TTNNContainer(child)
    device = _StubMeshDevice(num_devices=1)
    set_device(parent, device)

    assert parent.tt_child is child, "T3K module should remain on T3K device"


def test_module_call_before_set_device_raises_with_set_device_message():
    mod = _CountingTTNNModule()
    mod._fallback_torch_layer = nn.Identity()
    with pytest.raises(AssertionError) as info:
        mod(torch.zeros(1))
    assert "set_device" in str(info.value), f"AssertionError must mention set_device; got: {info.value!r}"


# ---------------------------------------------------------------------------
# kv_cache_kwargs resolution: from_pretrained-declared vs set_device-override
# ---------------------------------------------------------------------------
#
# The cache shape (capacity, block size, batch budget) is a model-config
# decision, so the recommended path is to pass it at load time
# (``AutoModelForCausalLM.from_pretrained(..., kv_cache_kwargs={...})``)
# which stashes it on ``model._tt_kv_cache_kwargs``. ``set_device`` reads
# that and merges any per-key override the caller passes at the bind site.
# The merge resolution (override wins) is the contract these tests pin
# down.


class _RecipeStub:
    """Recipe stand-in that records the kwargs passed to ``make_kv_cache``."""

    def __init__(self):
        self.received: dict = {}
        self.called: int = 0

    def build_module_dict(self, model):  # pragma: no cover - never called by set_device
        return {}

    def post_register(self, model):  # pragma: no cover - never called by set_device
        return None

    def make_kv_cache(self, model, device, **kwargs):
        self.called += 1
        self.received = dict(kwargs)
        # Return a sentinel so ``set_device`` attaches it as ``model._tt_kv_cache``.
        return ("stub-kv-cache", kwargs)


class _ModelStub(nn.Module):
    """Plain ``nn.Module`` whose class name is what ``set_device`` looks up in the registry."""


def _register_stub_recipe(monkeypatch):
    """Insert ``_RecipeStub`` into ``TT_MODEL_REGISTRY`` under ``_ModelStub`` for one test."""
    from tt_symbiote.models.auto import auto_mappings

    stub = _RecipeStub()
    original = dict(auto_mappings.TT_MODEL_REGISTRY)
    monkeypatch.setattr(
        auto_mappings,
        "TT_MODEL_REGISTRY",
        {**original, "_ModelStub": stub},
    )
    return stub


def test_kv_cache_kwargs_from_pretrained_flows_to_make_kv_cache(monkeypatch, reset_mesh_env):
    """``model._tt_kv_cache_kwargs`` (set by from_pretrained) should reach make_kv_cache."""
    stub_recipe = _register_stub_recipe(monkeypatch)
    model = _ModelStub()
    model._tt_kv_cache_kwargs = {"max_num_blocks": 512, "block_size": 64}
    device = _StubMeshDevice(num_devices=1)

    set_device(model, device)

    assert stub_recipe.called == 1, "make_kv_cache should be invoked exactly once"
    assert stub_recipe.received == {
        "max_num_blocks": 512,
        "block_size": 64,
    }, f"from_pretrained kwargs should propagate verbatim; got {stub_recipe.received}"
    assert hasattr(model, "_tt_kv_cache"), "set_device should attach the returned cache"


def test_set_device_rejects_bind_site_kwargs(monkeypatch, reset_mesh_env):
    """``set_device`` is strictly two-arg; any kwarg is a TypeError.

    Documents the post-refactor contract: cache shape (and every other
    runtime configuration decision) is a from_pretrained concern, not a
    bind-site concern. Passing ``kv_cache_kwargs``, ``dump_visualization``,
    or ``register_forward_hook`` here is no longer supported.
    """
    _register_stub_recipe(monkeypatch)
    model = _ModelStub()
    model._tt_kv_cache_kwargs = {"max_num_blocks": 512, "block_size": 64}
    device = _StubMeshDevice(num_devices=1)

    import pytest as _pytest

    for forbidden in (
        {"kv_cache_kwargs": {"max_num_blocks": 1024}},
        {"dump_visualization": True},
        {"register_forward_hook": True},
    ):
        with _pytest.raises(TypeError):
            set_device(model, device, **forbidden)


def test_missing_stash_falls_back_to_empty_dict(monkeypatch, reset_mesh_env):
    """A model without ``_tt_kv_cache_kwargs`` should still bind cleanly.

    ``getattr(obj, "_tt_kv_cache_kwargs", None) or {}`` in
    ``set_device`` means hand-constructed models (tests, capabilities
    probes) that bypass ``from_pretrained`` see make_kv_cache invoked
    with an empty kwargs dict.
    """
    stub_recipe = _register_stub_recipe(monkeypatch)
    model = _ModelStub()
    # Intentionally do not set ``model._tt_kv_cache_kwargs``.
    device = _StubMeshDevice(num_devices=1)

    set_device(model, device)

    assert stub_recipe.received == {}, f"missing stash should land as empty dict; got {stub_recipe.received}"


def test_set_device_with_no_kv_cache_kwargs_anywhere_uses_recipe_defaults(monkeypatch, reset_mesh_env):
    """No-op-kwarg path: from_pretrained set an empty stash."""
    stub_recipe = _register_stub_recipe(monkeypatch)
    model = _ModelStub()
    model._tt_kv_cache_kwargs = {}  # what from_pretrained sets when the kwarg was None
    device = _StubMeshDevice(num_devices=1)

    set_device(model, device)

    assert (
        stub_recipe.received == {}
    ), f"With an explicit empty stash, make_kv_cache should see {{}}; got {stub_recipe.received}"
