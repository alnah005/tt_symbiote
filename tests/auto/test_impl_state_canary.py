# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Software-only validation of the dynamic-canary gate.

A ``*_impl`` may ONLY create/replace a flat ``ttnn.Tensor`` / ``None`` slot (plus a flat
``torch.Tensor`` in preprocess) and may only ``del`` such an attr; any other ``__dict__``
write/delete on self or a touched child raises ``NonTensorStateMutationError`` at the first cold
``set_device`` / ``move_weights_to_device``. Runs with no hardware (conftest ttnn/tracy stubs +
``_FakeTTNNTensor`` sentinel). EXPECTED REAL-MODEL BREAKAGE (by design, NOT surfaced here): an
unmigrated offender raises at ``set_device`` ON HARDWARE; ``tests/auto/`` mocks ttnn so it passes.
"""

import pytest
import torch

from tt_symbiote.core.module import (
    NonTensorStateMutationError,
    StatelessTTNNModule,
)
from tt_symbiote.utils.device_management import set_device

# Shared stubs from conftest: the rich mesh-device stub (superset; the no-arg calls here read only
# get_num_devices) + the unified recursive binder.
from .conftest import _FakeTTNNTensor, _set_device_recursive, _StubMeshDevice

# ttnn is the stub installed by conftest at collection time.
import ttnn  # noqa: E402

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"


def _bind_and_prep(mod):
    """Cold weight-prep pass: set ``_device`` recursively + drive the public wrappers directly so
    the gate raise surfaces here (``set_device``'s warn-and-continue path is tested separately)."""
    _set_device_recursive(mod, _StubMeshDevice())
    mod.preprocess_weights()
    mod.move_weights_to_device()


# 1-4: legitimate cold writes -- MUST NOT raise (zero false positives)


def test_clean_module_passes():
    class _Clean(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.tt_bias = None  # None-slot lazily filled by move

        def move_weights_to_device_impl(self):
            self.tt_weight = _FakeTTNNTensor()
            self.tt_bias = _FakeTTNNTensor()  # None -> tensor
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    mod = _Clean()
    _bind_and_prep(mod)
    assert mod._weights_on_device is True


def test_biasless_linear_none_allowed():
    class _Biasless(StatelessTTNNModule):
        # NO __init__ defaults for tt_*_host / tt_* -> absent -> None on the bias-less path.
        def preprocess_weights_impl(self):
            self.tt_weight_host = _FakeTTNNTensor()
            self.tt_bias_host = None
            return super().preprocess_weights_impl()

        def move_weights_to_device_impl(self):
            self.tt_weight = _FakeTTNNTensor()
            self.tt_bias = None
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    mod = _Biasless()
    _bind_and_prep(mod)
    assert mod._weights_on_device is True


def test_absent_to_none_allowed():
    class _Slot(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            self._qkv_dram_bias = None  # absent -> None (mirrors _linear.py:953-954)
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    _bind_and_prep(_Slot())


def test_tensor_to_none_allowed():
    class _Drop(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.tt_bias = _FakeTTNNTensor()

        def move_weights_to_device_impl(self):
            self.tt_bias = None  # tensor -> None: dropping an optional tensor
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    _bind_and_prep(_Drop())


# 5-9: forbidden non-tensor writes -- MUST raise


def _make_stale_default():
    # A: a stale __init__ default int rebound in _impl (alongside a real tensor add).
    class _StaleDefault(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self._padded_vocab = 0

        def move_weights_to_device_impl(self):
            self._padded_vocab = 128256
            self.tt_weight = _FakeTTNNTensor()
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    return _StaleDefault()


def _make_created_only():
    # B: a non-tensor created only in _impl (no __init__ default -> <absent> -> object).
    class _CreatedOnly(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            self._input_shard_cfg = object()
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    return _CreatedOnly()


def _make_config_object():
    # NB: object() is non-None AND truthy; the ADD predicate must use `is None`, not bool() -- a
    # falsy-but-non-None config (whose __bool__ returns False) must STILL raise (the [config] id of
    # test_nontensor_add_raises exercises the identical `value is None -> return False` branch).
    class _Cfg(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            self.compute_kernel_config = object()
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    return _Cfg()


@pytest.mark.parametrize(
    "factory, substrings",
    [
        (
            _make_stale_default,
            ["_StaleDefault", "_padded_vocab", "0", "128256", "[on: self]", "move_weights_to_device_impl", "configure_runtime"],
        ),
        (_make_created_only, ["_input_shard_cfg", "<absent>", "move_weights_to_device_impl"]),
        (_make_config_object, ["compute_kernel_config"]),
    ],
    ids=["A", "B", "config"],
)
def test_nontensor_add_raises(factory, substrings):
    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(factory())
    msg = str(info.value)
    for sub in substrings:
        assert sub in msg


def test_tensor_container_rebind_raises():
    class _Chunks(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.tt_weight_chunks = []

        def move_weights_to_device_impl(self):
            self.tt_weight_chunks = [_FakeTTNNTensor(), _FakeTTNNTensor()]
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_Chunks())
    msg = str(info.value)
    assert "tt_weight_chunks" in msg
    assert "container of tensors" in msg


# 10-12: container in-place backstop + blind spots


def test_in_place_container_append_raises():
    class _Append(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.tt_weight_chunks = []

        def move_weights_to_device_impl(self):
            self.tt_weight_chunks.append(_FakeTTNNTensor())  # same id, len changed
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_Append())
    assert "tt_weight_chunks" in str(info.value)


def test_inplace_constant_length_is_blind_spot():
    """Documented intentional gap: constant-length in-place element replacement (same id, same len,
    no tensor-element transition) is invisible to the id-diff."""

    class _ConstLen(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self._buf = [_FakeTTNNTensor()]

        def move_weights_to_device_impl(self):
            self._buf[0] = _FakeTTNNTensor()  # same id, same len, tensor->tensor
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    _bind_and_prep(_ConstLen())  # NO raise -- documented blind spot


def test_stable_nontensor_list_not_flagged():
    class _Stable(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self._some_ints = [1, 2, 3]

        def move_weights_to_device_impl(self):
            self.tt_weight = _FakeTTNNTensor()  # does NOT touch _some_ints
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    _bind_and_prep(_Stable())


# 13-15: transitive (parent-sets-child) coverage + the load-bearing name filter


class _CleanChild(StatelessTTNNModule):
    """A direct child with NO ``_impl`` override (mirrors TTNNSDPAAttention)."""

    def __init__(self):
        super().__init__()
        self.program_config = None

    def forward(self, x):
        return x


def test_parent_sets_child_config_raises():
    class _Parent(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.child = _CleanChild()

        def move_weights_to_device_impl(self):
            super().move_weights_to_device_impl()  # recurses -> child flag flips inside window
            self.child.program_config = object()  # None -> config on the child
            return self

        def forward(self, x):
            return x

    parent = _Parent()
    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(parent)
    msg = str(info.value)
    assert "program_config" in msg
    assert "child child" in msg  # "[on: child child]" -- attributed to the direct child
    assert "move_weights_to_device_impl" in msg
    # B1 regression guard: the child's lifecycle flag flip must NOT be reported as a violation.
    assert "_weights_on_device" not in msg
    assert "_preprocessed_weight" not in msg


def test_child_own_nontensor_raises_before_parent():
    class _OffendingChild(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            self._bad = object()  # child's OWN forbidden self-write
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    class _Parent(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.child = _OffendingChild()

        def move_weights_to_device_impl(self):
            super().move_weights_to_device_impl()  # child raises here, before parent body
            return self

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_Parent())
    msg = str(info.value)
    assert "_OffendingChild" in msg  # child named as self
    assert "_bad" in msg
    assert "[on: self]" in msg


def test_multilevel_nested_subtree():
    class _Grandchild(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.slot = None

        def forward(self, x):
            return x

    class _Mid(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.grandchild = _Grandchild()

        def move_weights_to_device_impl(self):
            super().move_weights_to_device_impl()  # grandchild flags flip here
            self.grandchild.slot = object()  # mid-parent writes a DIRECT child (the grandchild)
            return self

        def forward(self, x):
            return x

    class _Root(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.mid = _Mid()

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_Root())
    msg = str(info.value)
    # Exactly one violation, attributed to the grandchild, blamed on _Mid's _impl.
    assert "slot" in msg
    assert "child grandchild" in msg
    assert "_Mid" in msg
    # No lifecycle-flag false positive at any level.
    assert "_weights_on_device" not in msg
    assert "_preprocessed_weight" not in msg
    assert msg.count("- slot:") == 1


# 16-17: preprocess vs move torch-tensor asymmetry


def test_preprocess_torch_tensor_allowed():
    class _PreTorch(StatelessTTNNModule):
        def preprocess_weights_impl(self):
            self.tt_weight_host = torch.zeros(4)  # torch tensor allowed in preprocess
            return super().preprocess_weights_impl()

        def forward(self, x):
            return x

    _bind_and_prep(_PreTorch())

    class _PreNonTensor(StatelessTTNNModule):
        def preprocess_weights_impl(self):
            self._derived_shape = (4, 4)  # non-tensor state -- forbidden even in preprocess
            return super().preprocess_weights_impl()

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_PreNonTensor())
    assert "_derived_shape" in str(info.value)
    assert "preprocess_weights_impl" in str(info.value)


def test_move_torch_tensor_forbidden():
    class _MoveTorch(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            self.tt_weight = torch.zeros(4)  # torch in move -- forbidden (must be ttnn)
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_MoveTorch())
    assert "tt_weight" in str(info.value)


# 18: spoof resistance


def test_mock_config_is_rejected():
    # Under the stub, ttnn.WormholeComputeKernelConfig(...) returns an _Anything; it must NOT
    # be treated as a ttnn.Tensor (proves positive type identity, not hasattr duck-typing).
    forbidden = ttnn.WormholeComputeKernelConfig()
    assert not isinstance(forbidden, _FakeTTNNTensor)

    class _Spoof(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            self.compute_kernel_config = forbidden
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_Spoof())
    assert "compute_kernel_config" in str(info.value)


# 19: bookkeeping / _fallback_torch_layer never flagged


def test_bookkeeping_and_fallback_not_flagged():
    class _Conv(StatelessTTNNModule):
        def preprocess_weights_impl(self):
            self._fallback_torch_layer = torch.nn.Identity()  # mirrors ttnn_conv.py:308
            self.tt_weight = _FakeTTNNTensor()
            return super().preprocess_weights_impl()

        def forward(self, x):
            return x

    _bind_and_prep(_Conv())


# 20-21: warm-path and trace-path bypass the gate


def test_warm_path_no_gate():
    counter = {"n": 0}

    class _OnceOffender(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            counter["n"] += 1
            self.tt_weight = _FakeTTNNTensor()
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    mod = _OnceOffender()
    mod._device = _StubMeshDevice()
    mod.preprocess_weights()
    mod.move_weights_to_device()  # cold -- runs _impl once
    mod.move_weights_to_device()  # warm -- early-returns, _impl NOT re-run
    assert counter["n"] == 1


def test_trace_running_bypasses_canary(monkeypatch):
    from tt_symbiote.core import run_config

    class _WouldOffend(StatelessTTNNModule):
        def move_weights_to_device_impl(self):
            self._bad = object()  # would raise if reached
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    mod = _WouldOffend()
    mod._device = _StubMeshDevice()
    mod._preprocessed_weight = True
    mod._weights_on_device = True  # trace path asserts this and returns BEFORE _impl
    monkeypatch.setattr(run_config, "_TRACE_RUNNING", True)
    mod.move_weights_to_device()  # asserts-and-returns; no snapshot, no _impl, no raise


# 22-23: delete verdict (cleanup ALLOW vs non-tensor delete FORBID)


def test_del_torch_source_weight_allowed():
    class _Cleanup(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self.torch_w1_proj = torch.zeros(4)
            self.tt_old = _FakeTTNNTensor()

        def preprocess_weights_impl(self):
            self.tt_w1_proj = _FakeTTNNTensor()
            del self.torch_w1_proj  # pre was a torch.Tensor -> ALLOW
            return super().preprocess_weights_impl()

        def move_weights_to_device_impl(self):
            del self.tt_old  # pre was a ttnn tensor -> ALLOW
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    _bind_and_prep(_Cleanup())


def test_del_nontensor_state_forbidden():
    class _DelState(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self._padded_vocab = 0

        def move_weights_to_device_impl(self):
            del self._padded_vocab  # pre was an int -> FORBID
            self.tt_weight = _FakeTTNNTensor()
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    with pytest.raises(NonTensorStateMutationError) as info:
        _bind_and_prep(_DelState())
    msg = str(info.value)
    assert "_padded_vocab" in msg
    assert "<absent>" in msg
    assert "deleted non-tensor state" in msg


# 24: the surgical re-raise -- set_device must NOT downgrade to a warning


def test_set_device_does_not_swallow_violation(monkeypatch):
    monkeypatch.delenv("MESH_DEVICE", raising=False)

    class _OffendingLeaf(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self._fallback_torch_layer = torch.nn.Identity()

        def move_weights_to_device_impl(self):
            self._bad = object()
            return super().move_weights_to_device_impl()

        def forward(self, x):
            return x

    leaf = _OffendingLeaf()
    with pytest.raises(NonTensorStateMutationError):
        set_device(leaf, _StubMeshDevice())
