# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Software-only WARM-LOAD-PARITY + FAILURE-B regression.

FAILURE-B: a bias-less linear sets ``self.tt_bias = None`` ONLY inside
``move_weights_to_device_impl``. On a WARM HIT that ``_impl`` is skipped, the cache (device tensors
only) never re-materializes the slot, so ``configure_runtime``/``forward`` crash reading
``self.tt_bias`` plain. Fix: record the cold-``None`` slot KEYS in the manifest (``none_slots``,
schema 2) and re-materialize them as ``None`` on warm BEFORE ``configure_runtime``.
"""

import json
import os

import pytest

from tt_symbiote.core import module as mod
from tt_symbiote.core import weight_cache as wc

# Shared (non-fixture) helpers from conftest (the _isolate_cache autouse fixture is conftest-level).
from .conftest import (
    _FakeDeviceTensor,
    _StubMeshDevice,
    _set_device_recursive,
)

# ``_bind`` stays LOCAL to the sibling file (wc-specific cold-prep binder).
from tests.auto.test_weight_cache import _bind
from tt_symbiote.core.module import StatelessTTNNModule

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"


# ---------------------------------------------------------------------------
# Synthetic proxies
# ---------------------------------------------------------------------------
class _BiaslessLeaf(StatelessTTNNModule):
    """Reproduces FAILURE B: ``tt_bias`` set ONLY in move_impl, None (bias-less); read in
    configure_runtime AND a stand-in forward. Owns >=1 real cacheable tensor so it takes the
    warm-read branch + writes a manifest. NO __init__ default for tt_bias (mirrors the real
    o_proj / QKV)."""

    def __init__(self):
        super().__init__()

    def preprocess_weights_impl(self):
        self.tt_weight_host = _FakeDeviceTensor(payload=1, shape=(4, 4), host=True)
        self.tt_bias_host = None

    def move_weights_to_device_impl(self):
        self.tt_weight = _FakeDeviceTensor(payload=1, shape=(4, 4))  # cacheable device tensor
        self.tt_bias = (
            None if self.tt_bias_host is None else _FakeDeviceTensor(payload=2, shape=(4, 4))
        )  # -> None (bias-less)

    def configure_runtime(self):
        # mirrors _linear.py:986: reads self.tt_bias PLAIN -> AttributeError if absent on warm.
        self._bias_fused = bool(self.tt_bias is not None)
        self.compute_kernel_config = ("HiFi2", 5)

    def read_bias(self):  # stand-in for the forward PLAIN read at _linear.py:1202
        return self.tt_bias

    def forward(self, x):
        return x


class _CondSlotLeaf(StatelessTTNNModule):
    """A conditional slot: a real device tensor in one config, None in another. The cache key folds
    the branch selector (weight_cache_variant), so the two branches never share a manifest."""

    def __init__(self, use_dram):
        super().__init__()
        self._use_dram = use_dram

    def weight_cache_variant(self):
        return f"dram{int(self._use_dram)}"

    def preprocess_weights_impl(self):
        self.tt_weight_host = _FakeDeviceTensor(payload=1, shape=(4, 4), host=True)

    def move_weights_to_device_impl(self):
        self.tt_weight = _FakeDeviceTensor(payload=1, shape=(4, 4))
        self._dram_weight = _FakeDeviceTensor(payload=2, shape=(4, 4)) if self._use_dram else None

    def configure_runtime(self):
        self.compute_kernel_config = ("HiFi2", 1)

    def read_dram(self):  # stand-in plain forward read of the conditional slot
        return self._dram_weight

    def forward(self, x):
        return x


class _ChunkedLeaf(StatelessTTNNModule):
    """A container-flatten leaf modeling the LM-head: N flattened chunk tensors + a None bias slot
    + a configure-derived scalar (_num_chunks). Exercises parity over a multi-tensor module."""

    def __init__(self, n_chunks=2):
        super().__init__()
        self._n = n_chunks

    def preprocess_weights_impl(self):
        for i in range(self._n):
            setattr(self, f"tt_weight_host_chunk_{i}", _FakeDeviceTensor(payload=i, shape=(4, 4), host=True))

    def move_weights_to_device_impl(self):
        for i in range(self._n):
            setattr(self, f"tt_weight_chunk_{i}", _FakeDeviceTensor(payload=i, shape=(4, 4)))
        self.tt_bias = None  # bias-less -> none_slot

    def configure_runtime(self):
        self._num_chunks = self._n  # configure product, recomputed every load
        self.compute_kernel_config = ("HiFi2", 2)

    def read_bias(self):
        return self.tt_bias

    def forward(self, x):
        return x


class _EmptyParent(StatelessTTNNModule):
    """A 0-cacheable-tensor 0-none container -- proves the fix does NOT make such a parent write a
    manifest / flip owns_cached_tensors."""

    def __init__(self):
        super().__init__()
        self.child = _BiaslessLeaf()
        self._parent_cfg = None

    def preprocess_weights_impl(self):
        self.child.preprocess_weights()
        return self

    def move_weights_to_device_impl(self):
        self.child.move_weights_to_device()
        return self

    def configure_runtime(self):
        self._parent_cfg = "set"

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# The general warm-load-parity guard
# ---------------------------------------------------------------------------
def assert_warm_load_parity(make_module, name, dev):
    cold = make_module()
    _bind(cold, device=dev, name=name)
    cold.preprocess_weights()
    cold.move_weights_to_device()
    assert cold._configure_runtime_done
    cold_keys = {k for k in cold.__dict__ if k not in mod.FRAMEWORK_BOOKKEEPING_KEYS}
    warm = make_module()
    _bind(warm, device=dev, name=name)
    warm.preprocess_weights()
    warm.move_weights_to_device()
    assert warm._weights_from_cache
    warm_keys = {k for k in warm.__dict__ if k not in mod.FRAMEWORK_BOOKKEEPING_KEYS}
    missing = cold_keys - warm_keys
    assert not missing, f"warm load missing forward/configure-read attrs for {name}: {sorted(missing)}"


# ---------------------------------------------------------------------------
# FAILURE-B regression: configure_runtime AND forward read
# ---------------------------------------------------------------------------
def test_failure_b_warm_load_bias_slot_present():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    cold = _BiaslessLeaf()
    _bind(cold, device=dev, name="model.biasless")
    cold.preprocess_weights()
    cold.move_weights_to_device()
    assert cold._configure_runtime_done is True
    assert cold.tt_bias is None and cold.read_bias() is None and cold._bias_fused is False

    warm = _BiaslessLeaf()
    _bind(warm, device=dev, name="model.biasless")
    warm.preprocess_weights()
    warm.move_weights_to_device()  # WARM HIT; configure runs INLINE at the top-level
    assert warm._weights_from_cache is True
    # WITHOUT the fix: configure_runtime crashes (AttributeError at self.tt_bias) DURING
    # move_weights_to_device. WITH the fix: present (None) before configure runs.
    assert "tt_bias" in warm.__dict__ and warm.tt_bias is None
    assert warm._bias_fused is False and warm.read_bias() is None


def test_failure_b_repro_without_none_slots(monkeypatch):
    """Negative control: a PRE-FIX manifest (no none_slots) -> warm move-leg MUST raise
    AttributeError inside configure_runtime. STRONGER than a forward-only repro; matches the live
    T3K crash; would NOT raise if the ttnn stubbing were inert (so the test also guards inertness)."""
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    orig = wc.save_module_weights
    # Force none_slots=[] to reproduce a pre-fix manifest that records no None slots.
    monkeypatch.setattr(
        wc, "save_module_weights", lambda module, none_slots=None: orig(module, none_slots=[])
    )
    cold = _BiaslessLeaf()
    _bind(cold, device=dev, name="model.biasless2")
    cold.preprocess_weights()
    cold.move_weights_to_device()
    warm = _BiaslessLeaf()
    _bind(warm, device=dev, name="model.biasless2")
    warm.preprocess_weights()
    with pytest.raises(AttributeError):
        warm.move_weights_to_device()  # configure_runtime hits self.tt_bias -> absent -> AttributeError


# ---------------------------------------------------------------------------
# conditional-slot no-clobber
# ---------------------------------------------------------------------------
def test_conditional_slot_not_clobbered():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    # TRUE branch: warm must restore the REAL tensor (NOT clobbered to None).
    cold_t = _CondSlotLeaf(use_dram=True)
    _bind(cold_t, device=dev, name="model.cond")
    cold_t.preprocess_weights()
    cold_t.move_weights_to_device()
    assert isinstance(cold_t.read_dram(), _FakeDeviceTensor)
    # COLD-MANIFEST membership (folded from C1): TRUE branch records _dram_weight as a TENSOR,
    # never a none_slot.
    mt = _read_manifest(cold_t)
    assert "_dram_weight" in {a["name"] for a in mt["tensor_attrs"]}
    assert "_dram_weight" not in mt["none_slots"]
    warm_t = _CondSlotLeaf(use_dram=True)
    _bind(warm_t, device=dev, name="model.cond")
    warm_t.preprocess_weights()
    warm_t.move_weights_to_device()
    assert warm_t._weights_from_cache is True
    assert isinstance(warm_t.read_dram(), _FakeDeviceTensor), "TRUE-branch tensor must NOT be clobbered to None"

    # FALSE branch: distinct variant -> distinct manifest -> warm restores None.
    cold_f = _CondSlotLeaf(use_dram=False)
    _bind(cold_f, device=dev, name="model.cond")
    cold_f.preprocess_weights()
    cold_f.move_weights_to_device()
    assert cold_f.read_dram() is None
    # COLD-MANIFEST membership (folded from C1): FALSE branch records _dram_weight as a none_slot,
    # never a tensor.
    mf = _read_manifest(cold_f)
    assert "_dram_weight" not in {a["name"] for a in mf["tensor_attrs"]}
    assert "_dram_weight" in mf["none_slots"]
    warm_f = _CondSlotLeaf(use_dram=False)
    _bind(warm_f, device=dev, name="model.cond")
    warm_f.preprocess_weights()
    warm_f.move_weights_to_device()
    assert warm_f._weights_from_cache is True
    assert "_dram_weight" in warm_f.__dict__ and warm_f.read_dram() is None


# ---------------------------------------------------------------------------
# warm-load-parity guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "make_module,name",
    [
        (lambda: _BiaslessLeaf(), "model.parity.biasless"),
        (lambda: _CondSlotLeaf(use_dram=True), "model.parity.cond_t"),
        (lambda: _CondSlotLeaf(use_dram=False), "model.parity.cond_f"),
        (lambda: _ChunkedLeaf(n_chunks=2), "model.parity.chunked"),
    ],
)
def test_warm_load_parity(make_module, name):
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    assert_warm_load_parity(make_module, name, dev)


# ---------------------------------------------------------------------------
# manifest-content unit cases
# ---------------------------------------------------------------------------
def _read_manifest(module):
    key = wc._build_key(module)
    return wc.get_backend().read_manifest(key.scope_digest(), key.module_digest())


def test_manifest_records_none_slots():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    cold = _BiaslessLeaf()
    _bind(cold, device=dev, name="model.manifest")
    cold.preprocess_weights()
    cold.move_weights_to_device()
    manifest = _read_manifest(cold)
    assert manifest is not None
    assert manifest["schema"] == 2
    assert manifest["none_slots"] == ["tt_bias"]
    tensor_names = {a["name"] for a in manifest["tensor_attrs"]}
    assert "tt_bias" not in tensor_names  # never both a tensor AND a none_slot
    assert "tt_weight" in tensor_names


def test_schema_v1_back_compat():
    """A pre-fix schema-1 manifest (no `none_slots`, no `schema` field) must be treated as a re-MISS
    in the LOAD path (the schema guard), and the reader must not KeyError on the missing key."""
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    # cold build to populate the on-disk tensor + manifest
    cold = _BiaslessLeaf()
    _bind(cold, device=dev, name="model.v1compat")
    cold.preprocess_weights()
    cold.move_weights_to_device()
    key = wc._build_key(cold)
    backend = wc.get_backend()
    module_dir = backend._module_dir(key.scope_digest(), key.module_digest())
    manifest_path = os.path.join(module_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Rewrite as a schema-1-style manifest: drop `schema` and `none_slots`.
    manifest.pop("schema", None)
    manifest.pop("none_slots", None)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, default=str)
    # has_complete stays schema-agnostic -> still True.
    assert backend.has_complete(key.scope_digest(), key.module_digest()) is True
    # The LOAD path treats it as a re-MISS (returns False, no KeyError on the missing none_slots).
    fresh = _BiaslessLeaf()
    _bind(fresh, device=dev, name="model.v1compat")
    fresh.preprocess_weights()
    assert wc.try_load_module_weights(fresh) is False


def test_owns_cached_tensors_unchanged_for_empty_parent():
    """Non-regression mirroring test_weight_cache.py:571 -- a 0-cacheable 0-none container writes NO
    manifest and owns_cached_tensors stays False (so it ALWAYS re-runs _impl + recurses on warm).
    The none_slots fix must NOT change this (the `if not members: return` early-return is KEPT)."""
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    cold = _EmptyParent()
    _set_device_recursive(cold, dev)  # set device on the parent AND the child
    cold._unique_name = "model.ep"
    cold.child._unique_name = "model.ep.child"
    cold.preprocess_weights()
    cold.move_weights_to_device()
    # parent owns 0 cacheable tensors AND records no none_slots manifest -> stays False.
    assert wc.owns_cached_tensors(cold) is False
    # the child (a tensor-owner) DID write a manifest with the none_slot.
    assert wc.owns_cached_tensors(cold.child) is True
    # the 0-cacheable parent is counted as a PASSTHROUGH (not a miss); the child is a real cold MISS.
    cold_stats = wc.STATS.as_dict()
    assert cold_stats["passthrough"] >= 1
    assert cold_stats["misses"] >= 1

    # warm: a fresh tree -> parent re-runs _impl (recurses); child HITs and restores tt_bias=None.
    warm = _EmptyParent()
    _set_device_recursive(warm, dev)
    warm._unique_name = "model.ep"
    warm.child._unique_name = "model.ep.child"
    warm.preprocess_weights()
    warm.move_weights_to_device()
    assert warm.child._weights_from_cache is True
    assert warm.child.tt_bias is None
