# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test-only fixtures + shared stubs for ``tests/auto/``.

These tests do not need real hardware, but the ``tt_symbiote`` package
unconditionally imports ``ttnn`` and ``tracy``. We install lightweight
stand-ins for both at collection time so the tests run on machines that
do not have either C extension built. The weight-cache / canary / set_device
feature files share the mesh-device stub, the recursive device-binder, the
serialization shims, and the cache-isolation fixture defined here.
"""

import pickle
import sys
import types

import pytest


class _Anything:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Anything()

    def __getattr__(self, _):
        return _Anything()

    def __getitem__(self, _):
        return _Anything()

    def __setitem__(self, *a, **k):
        pass

    def __iter__(self):
        return iter([])

    def __mul__(self, _):
        return _Anything()

    def __rmul__(self, _):
        return _Anything()

    def __or__(self, _):
        return _Anything()

    def __ror__(self, _):
        return _Anything()

    def __bool__(self):
        return False


class _StubMod(types.ModuleType):
    def __getattr__(self, name):
        return _Anything()


for name in (
    "ttnn",
    "ttnn.model_preprocessing",
    "ttnn.distributed",
    "ttnn.operations",
    "ttnn.operations.transformer",
    "tracy",
    "tracy.signpost",
):
    if name not in sys.modules:
        mod = _StubMod(name)
        mod.__file__ = f"<stub:{name}>"
        sys.modules[name] = mod
sys.modules["tracy"].signpost = _Anything()


class _FakeTTNNTensor:
    """Concrete sentinel type the canary treats as ``ttnn.Tensor`` (the stub makes ``ttnn.Tensor`` a
    non-type ``_Anything``). A registered concrete class cannot be spoofed by ``_Anything`` (which
    only spoofs ``hasattr``-style duck-typing, never ``isinstance``)."""


# Register the sentinel at collection time (deferred import: after the ttnn stub is installed above).
from tt_symbiote.core.module import _register_canary_fake_tensor_type  # noqa: E402

_register_canary_fake_tensor_type(_FakeTTNNTensor)


# Deferred package imports (after the ttnn stub is installed): aliased ``_core_module`` (the
# stub-loop above binds a local ``mod``; alias to avoid confusion) for the cache-driver reset, and
# ``TTNNModule`` for the recursive binder's isinstance walk.
from tt_symbiote.core import module as _core_module  # noqa: E402
from tt_symbiote.core.module import TTNNModule  # noqa: E402


# Shared mesh-device stub (RICH superset serving all 4 feature files): the extra grid/arch methods
# are inert for the thin canary/set_device callers and present for the weight_cache key/scope digests.
class _Grid:
    def __init__(self, x=8, y=8):
        self.x = x
        self.y = y


class _StubMeshDevice:
    def __init__(self, shape=(1, 1), num_devices=1):
        self.shape = list(shape)
        self._n = num_devices

    def get_num_devices(self):
        return self._n

    def compute_with_storage_grid_size(self):
        return _Grid(8, 8)

    def dram_grid_size(self):
        return _Grid(12, 1)

    def arch(self):
        return "wormhole_b0"


# Serialization shims + sentinels (the weight_cache / warm_parity dump/load/StorageType stubs).
class _DeviceSentinel:
    """The sentinel ``ttnn.StorageType.DEVICE`` is monkeypatched to and that
    ``_FakeDeviceTensor.storage_type()`` returns -- so ``_is_device_resident`` is non-vacuously True."""

    def __repr__(self):
        return "DEVICE"


_DEVICE = _DeviceSentinel()


class _HostSentinel:
    def __repr__(self):
        return "HOST"


_HOST = _HostSentinel()


class _MemCfg:
    """Comparable memory-config sentinel (round-trips through pickle for the dump/load shim)."""

    def __init__(self, name="dram_interleaved"):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _MemCfg) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"MemCfg({self.name})"


class _TensorSpec:
    def __init__(self, memcfg):
        self._memcfg = memcfg

    def memory_config(self):
        return self._memcfg


class _FakeDeviceTensor(_FakeTTNNTensor):
    """Concrete DEVICE-resident fake tensor the cache captures + serializes."""

    def __init__(self, payload=0, shape=(1, 1), memcfg=None, host=False):
        self._payload = payload
        self._shape = tuple(shape)
        self._memcfg = memcfg if memcfg is not None else _MemCfg()
        self._host = host

    @property
    def shape(self):
        return self._shape

    def storage_type(self):
        return _HOST if self._host else _DEVICE

    def tensor_spec(self):
        return _TensorSpec(self._memcfg)


# Register the device-tensor sentinel so the canary treats it as a ttnn.Tensor too (additive +
# idempotent; ``_FakeDeviceTensor`` IS-A ``_FakeTTNNTensor`` so registration order is irrelevant).
_register_canary_fake_tensor_type(_FakeDeviceTensor)


def fake_dump_tensor(path, t):
    with open(path, "wb") as f:
        pickle.dump({"payload": t._payload, "shape": t._shape, "memcfg": t._memcfg}, f)


def fake_load_tensor(path, *, device=None):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return _FakeDeviceTensor(payload=d["payload"], shape=d["shape"], memcfg=d["memcfg"])


def fake_deallocate(t):
    return None


# Unified recursive device-binder (canary superset: TTNNModule + dict + list/tuple; a strict
# superset of the weight_cache StatelessTTNNModule-only variant -- the extra branches are inert
# for wc/parity fixtures, which have no Stateful module or dict-of-children).
def _set_device_recursive(module, device):
    module._device = device
    for value in list(module.__dict__.values()):
        if isinstance(value, TTNNModule):
            _set_device_recursive(value, device)
        elif isinstance(value, dict):
            for sub in value.values():
                if isinstance(sub, TTNNModule):
                    _set_device_recursive(sub, device)
        elif isinstance(value, (list, tuple)):
            for sub in value:
                if isinstance(sub, TTNNModule):
                    _set_device_recursive(sub, device)


# Cache-isolation autouse fixture (conftest-level autouse -> applies dir-wide). Safe: the 22 errors
# are setup-time (immune to fixtures) and the non-feature tests own no cacheable tensors (inert
# under the cache redirect). ``wc``/``ttnn`` are lazy imports here so conftest import cost stays off
# the non-feature tests.
@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the cache at a fresh temp dir + reset stats/memoization + patch the ttnn stub."""
    import ttnn

    from tt_symbiote.core import weight_cache as wc

    cache_dir = tmp_path / "wc"
    monkeypatch.setenv("TT_SYMBIOTE_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("TT_SYMBIOTE_WEIGHT_CACHE", raising=False)
    monkeypatch.delenv("TT_SYMBIOTE_RUN_MODE", raising=False)
    monkeypatch.setattr(ttnn, "dump_tensor", fake_dump_tensor, raising=False)
    monkeypatch.setattr(ttnn, "load_tensor", fake_load_tensor, raising=False)
    monkeypatch.setattr(ttnn, "deallocate", fake_deallocate, raising=False)
    # StorageType.DEVICE must be the SAME sentinel _FakeDeviceTensor.storage_type() returns.
    storage_type = type("StorageType", (), {"DEVICE": _DEVICE})
    monkeypatch.setattr(ttnn, "StorageType", storage_type, raising=False)
    monkeypatch.setattr(ttnn, "TILE_SIZE", 32, raising=False)  # dots_ocr chunk math reads this
    wc._reset_backend_for_tests()
    wc.STATS.reset()
    wc._warn_once_seen.clear()
    # reset the deferred-configure driver globals
    _core_module._CONFIGURE_DEPTH = 0
    _core_module._PENDING_CONFIGURE = []
    _core_module._CONFIGURE_IN_PROGRESS = False
    yield
    wc._reset_backend_for_tests()
