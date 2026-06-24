# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Software-only validation of the disk weight-cache + depth-tracked deferred configure driver.

Covers the cache-layer + driver cases, the per-class dots_ocr canary proxy over all 17 migrated
classes, and the composite-recursion / PatchEmbed exercises.

Runs under the ``tests/auto`` ttnn stub (``ttnn.Tensor``/``ttnn.StorageType`` are non-type
``_Anything``s); conftest registers a concrete ``_FakeDeviceTensor`` and the ``_isolate_cache``
autouse fixture monkeypatches ``ttnn.StorageType.DEVICE`` so the DEVICE-residency filter fires.
"""

import json
import os
import re

import pytest
import torch

from tt_symbiote.core import module as mod
from tt_symbiote.core import weight_cache as wc
from tt_symbiote.core.module import (
    NonTensorStateMutationError,
    StatelessTTNNModule,
)

# Shared stubs + the conftest-level autouse cache-isolation fixture (rich mesh-device, dump/load
# shims, sentinels, unified recursive binder). ``_bind`` stays LOCAL below (wc-specific).
from .conftest import (
    _FakeDeviceTensor,
    _set_device_recursive,
    _StubMeshDevice,
    fake_dump_tensor,
)

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------
def _bind(module, device=None, name=None):
    """Bind a module for a cold weight-prep pass (no real set_device machinery). Does NOT flip
    ``_preprocessed_weight`` -- callers invoke ``preprocess_weights()`` themselves."""
    device = device or _StubMeshDevice()
    module._device = device
    if name is not None:
        module._unique_name = name


# Mock TTNNModules for the cache-layer + driver cases
class _Leaf(StatelessTTNNModule):
    """A weight-bearing leaf: one cacheable device tensor + a recomputed config scalar."""

    def __init__(self, payload=1):
        super().__init__()
        self._payload = payload
        self._src = torch.zeros(4, 4)  # torch source weight (for src_fingerprint)
        self.tt_weight = None
        self.compute_kernel_config = None
        self._derived = 0

    def preprocess_weights_impl(self):
        self.tt_weight = _FakeDeviceTensor(payload=self._payload, shape=(4, 4))  # host-ish; allowed in preprocess

    def move_weights_to_device_impl(self):
        self.tt_weight = _FakeDeviceTensor(payload=self._payload, shape=(4, 4))

    def configure_runtime(self):
        self.compute_kernel_config = ("HiFi4", self._payload)
        self._derived = 100 + self._payload

    def forward(self, x):
        return x


class _Parent(StatelessTTNNModule):
    """A composite owning ZERO cacheable tensors + N child leaves (records configure order)."""

    def __init__(self, n_children=2, order_list=None):
        super().__init__()
        self._order = order_list if order_list is not None else []
        self.children_list = [_Leaf(payload=i + 1) for i in range(n_children)]
        self._parent_cfg = None

    def move_weights_to_device_impl(self):
        for c in self.children_list:
            c.move_weights_to_device()
        return self

    def preprocess_weights_impl(self):
        for c in self.children_list:
            c.preprocess_weights()
        return self

    def configure_runtime(self):
        self._parent_cfg = "set"
        self._order.append(("parent", id(self)))

    def forward(self, x):
        return x


class _OrderLeaf(_Leaf):
    def configure_runtime(self):
        super().configure_runtime()
        self._order_ref.append(("leaf", id(self)))


# Core O5 round-trip: cold save -> warm HIT load + configure recompute
def test_round_trip_restores_tensors_and_configure_runtime():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)

    # COLD build.
    cold = _Leaf(payload=7)
    _bind(cold, device=dev, name="model.leaf")
    cold.preprocess_weights()
    cold.move_weights_to_device()
    assert cold._configure_runtime_done is True
    assert cold.compute_kernel_config == ("HiFi4", 7)
    assert cold._derived == 107
    stats = wc.STATS.as_dict()
    assert stats["stores"] >= 1, "cold leg must store at least one tensor (no vacuous pass)"
    assert stats["misses"] >= 1
    # idempotent configure-once: a re-call is a no-op (folded from D1).
    before = cold._derived
    cold._configure_runtime_once()
    assert cold._derived == before

    # WARM HIT load (fresh instance, same identity).
    warm = _Leaf(payload=7)
    _bind(warm, device=dev, name="model.leaf")
    warm.preprocess_weights()
    warm.move_weights_to_device()
    assert warm._weights_from_cache is True
    assert isinstance(warm.tt_weight, _FakeDeviceTensor)
    assert warm.tt_weight._payload == 7
    assert warm.compute_kernel_config == ("HiFi4", 7)  # recomputed by configure_runtime on warm
    assert warm._derived == 107
    assert warm._configure_runtime_done is True  # configure ran on the warm path too (folded from D1)
    assert wc.STATS.as_dict()["hits"] >= 1


# N1 PROOF: configure fires via the lazy move() path with NO set_device
def test_configure_fires_via_lazy_module_run_without_set_device(monkeypatch):
    # Disable the cache so this is purely the cold MISS depth-flush path.
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")
    order = []
    parent = _Parent(n_children=2, order_list=order)
    for c in parent.children_list:
        c.__class__ = _OrderLeaf
        c._order_ref = order
    dev = _StubMeshDevice()
    _set_device_recursive(parent, dev)
    parent._preprocessed_weight = True
    for c in parent.children_list:
        c._preprocessed_weight = True

    # Drive the PARENT's move directly (simulating the lazy module_run entry).
    parent.move_weights_to_device()

    assert parent._configure_runtime_done is True
    for c in parent.children_list:
        assert c._configure_runtime_done is True
    # children-before-parents
    assert order[-1][0] == "parent"
    assert all(o[0] == "leaf" for o in order[:-1])
    assert mod._CONFIGURE_DEPTH == 0
    assert mod._PENDING_CONFIGURE == []


def test_configure_children_before_parents(monkeypatch):
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")
    order = []

    class _Mid(_Parent):
        def configure_runtime(self):
            self._order.append(("mid", id(self)))

    root = _Parent(n_children=0, order_list=order)
    mid = _Mid(n_children=2, order_list=order)
    for c in mid.children_list:
        c.__class__ = _OrderLeaf
        c._order_ref = order
    root.children_list = [mid]

    dev = _StubMeshDevice()
    _set_device_recursive(root, dev)
    for m in (root, mid, *mid.children_list):
        m._preprocessed_weight = True
    root.move_weights_to_device()
    kinds = [o[0] for o in order]
    # leaves first, then mid, then root (parent)
    assert kinds[-1] == "parent"
    assert kinds[-2] == "mid"
    assert all(k == "leaf" for k in kinds[:-2])


# Key sensitivity (incl. preprocess_source_fp + PatchMerger-style variant)
def test_key_sensitivity():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    a = _Leaf(payload=1)
    _bind(a, device=dev, name="model.leaf")
    ka = wc._build_key(a)
    # identical -> identical digests
    b = _Leaf(payload=1)
    _bind(b, device=dev, name="model.leaf")
    kb = wc._build_key(b)
    assert ka.module_digest() == kb.module_digest()
    assert ka.scope_digest() == kb.scope_digest()
    # different name -> different module digest
    c = _Leaf(payload=1)
    _bind(c, device=dev, name="model.other")
    assert wc._build_key(c).module_digest() != ka.module_digest()
    # scope keys ONLY on device_arch/mesh_shape/num_devices -> a different mesh -> different scope
    dev2 = _StubMeshDevice(shape=(4, 1), num_devices=4)
    f = _Leaf(payload=1)
    _bind(f, device=dev2, name="model.leaf")
    assert wc._build_key(f).scope_digest() != ka.scope_digest()
    # variant flip -> different module digest
    e = _Leaf(payload=1)
    _bind(e, device=dev, name="model.leaf")
    e.weight_cache_variant = lambda: "bf1"
    assert wc._build_key(e).module_digest() != ka.module_digest()

    # preprocess_source_fp tracks configure_runtime body edits: a FRESH _LeafEdited -> different fp.
    fp_a = wc._build_key(a).preprocess_source_fp

    class _LeafEdited(_Leaf):
        def configure_runtime(self):
            self.compute_kernel_config = ("LoFi", self._payload)  # edited body
            self._derived = 999

    g = _LeafEdited(payload=1)
    _bind(g, device=dev, name="model.leaf")
    fp_b = wc._build_key(g).preprocess_source_fp
    assert fp_a != fp_b, "editing configure_runtime must change preprocess_source_fp"


def test_determinism_guard():
    dev = _StubMeshDevice()
    # None name -> DISABLED, _unique_name unchanged
    m = _Leaf()
    m._device = dev
    m._unique_name = None
    assert wc.module_cache_state(m) == wc.CacheState.DISABLED
    assert m._unique_name is None  # the module_name property was NOT triggered

    # id-fallback form -> DISABLED
    m2 = _Leaf()
    m2._device = dev
    m2._unique_name = "TTNNLeaf_140234567"
    m2._tt_cache_state = None
    assert wc.module_cache_state(m2) == wc.CacheState.DISABLED


def test_mesh_device_unset_arch_unknown(monkeypatch):
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    assert wc._device_arch() == "unknown"


# DEVICE-resident-only capture (C2)
def test_device_resident_only_capture():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    m = _Leaf()
    m._device = dev
    m.tt_weight = _FakeDeviceTensor(payload=1, shape=(4, 4), host=False)
    m.tt_weight_host = _FakeDeviceTensor(payload=2, shape=(4, 4), host=True)  # *_host excluded
    m.host_tensor = _FakeDeviceTensor(payload=3, shape=(4, 4), host=True)  # host -> skipped
    m.torch_w = torch.zeros(4, 4)  # not a ttnn tensor -> skipped
    caps = wc.cacheable_tensor_attrs(m)
    assert "tt_weight" in caps
    assert "tt_weight_host" not in caps
    assert "host_tensor" not in caps
    assert "torch_w" not in caps


def test_container_flatten_round_trip():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)

    class _Chunked(StatelessTTNNModule):
        def __init__(self):
            super().__init__()
            self._src = torch.zeros(8, 8)
            self._num_chunks = 0

        def move_weights_to_device_impl(self):
            for i in range(2):
                setattr(self, f"tt_weight_chunk_{i}", _FakeDeviceTensor(payload=10 + i, shape=(4, 4)))

        def configure_runtime(self):
            self._num_chunks = 2

        def forward(self, x):
            return x

    cold = _Chunked()
    _bind(cold, device=dev, name="model.lmhead")
    cold.preprocess_weights()
    cold.move_weights_to_device()
    caps = wc.cacheable_tensor_attrs(cold)
    assert set(caps) == {"tt_weight_chunk_0", "tt_weight_chunk_1"}

    warm = _Chunked()
    _bind(warm, device=dev, name="model.lmhead")
    warm.preprocess_weights()
    warm.move_weights_to_device()
    assert warm._weights_from_cache is True
    assert getattr(warm, "tt_weight_chunk_0")._payload == 10
    assert getattr(warm, "tt_weight_chunk_1")._payload == 11


# Partial reuse: a cold-built leaf -> fresh same-identity instance HITs; a different name MISSes.
def test_partial_reuse_after_kill():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    # build two leaves cold (both stored)
    for name, pl in (("model.l0", 1), ("model.l1", 2)):
        leaf = _Leaf(payload=pl)
        _bind(leaf, device=dev, name=name)
        leaf.preprocess_weights()
        leaf.move_weights_to_device()
    # simulate the Nth being killed mid-dump: leave an orphan tmp dir for l1
    backend = wc.get_backend()
    # l0 is HIT, a fresh "l2" is MISS
    l0 = _Leaf(payload=1)
    _bind(l0, device=dev, name="model.l0")
    assert wc.module_cache_state(l0) == wc.CacheState.HIT
    l2 = _Leaf(payload=9)
    _bind(l2, device=dev, name="model.l2")
    assert wc.module_cache_state(l2) == wc.CacheState.MISS


# A composite owning ZERO device tensors does NOT short-circuit
def test_container_parent_no_short_circuit():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    order = []
    # cold: build + store children
    parent = _Parent(n_children=2, order_list=order)
    parent.children_list[0]._unique_name = "model.p.c0"
    parent.children_list[1]._unique_name = "model.p.c1"
    _set_device_recursive(parent, dev)
    parent._unique_name = "model.p"
    for m in (parent, *parent.children_list):
        m._preprocessed_weight = True
    parent.preprocess_weights()
    parent.move_weights_to_device()

    # parent owns 0 cacheable tensors
    assert not wc.owns_cached_tensors(parent)

    # warm: a fresh tree -> parent runs _impl (recurses), children HIT
    order2 = []
    warm = _Parent(n_children=2, order_list=order2)
    warm.children_list[0]._unique_name = "model.p.c0"
    warm.children_list[1]._unique_name = "model.p.c1"
    _set_device_recursive(warm, dev)
    warm._unique_name = "model.p"
    for m in (warm, *warm.children_list):
        m._preprocessed_weight = True
    warm.preprocess_weights()
    warm.move_weights_to_device()
    for c in warm.children_list:
        assert c._weights_from_cache is True  # children restored from cache


def test_run_mode_gating(monkeypatch):
    # NORMAL
    monkeypatch.delenv("TT_SYMBIOTE_RUN_MODE", raising=False)
    assert wc.caching_enabled_for_reads()
    assert wc.caching_enabled_for_writes()
    # TRACED
    monkeypatch.setenv("TT_SYMBIOTE_RUN_MODE", "TRACED")
    assert wc.caching_enabled_for_reads()
    # DPL/SEL read-bypass (writes still on)
    for m in ("DPL", "DPL_NO_ERROR_PROP", "SEL"):
        monkeypatch.setenv("TT_SYMBIOTE_RUN_MODE", m)
        assert not wc.caching_enabled_for_reads()
        assert wc.caching_enabled_for_writes()
    # LIGHTWEIGHT / CPU off
    for m in ("LIGHTWEIGHT", "CPU"):
        monkeypatch.setenv("TT_SYMBIOTE_RUN_MODE", m)
        assert not wc.caching_enabled_for_reads()
        assert not wc.caching_enabled_for_writes()
    # env kill switch
    monkeypatch.setenv("TT_SYMBIOTE_RUN_MODE", "NORMAL")
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")
    assert not wc.caching_enabled_for_reads()
    assert not wc.caching_enabled_for_writes()


# Backend atomicity (tmpdir sibling; commit os.replace; complete:true LAST)
def test_backend_atomicity():
    backend = wc.LocalFsWeightCacheBackend(root=os.environ["TT_SYMBIOTE_CACHE_DIR"])
    handle = backend.begin_write("scope0", "mod0")
    # tmp dir is a SIBLING of the final dir (same parent), with a .tmp. marker
    final = backend._module_dir("scope0", "mod0")
    assert handle._tmp_dir.startswith(final + ".tmp.")
    assert os.path.dirname(handle._tmp_dir) == os.path.dirname(final)
    p = handle.tensor_path_for("tt_weight")
    fake_dump_tensor(p, _FakeDeviceTensor(payload=1, shape=(4, 4)))
    handle.write_manifest({"complete": True, "owned_tensor_count": 1, "tensor_attrs": []})
    handle.commit()
    assert backend.has_complete("scope0", "mod0")
    manifest = backend.read_manifest("scope0", "mod0")
    assert manifest["complete"] is True


# Post-load memcfg guard falls back (B2-adjacent)
def test_post_load_memcfg_guard_falls_back():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    cold = _Leaf(payload=4)
    _bind(cold, device=dev, name="model.guard")
    cold.preprocess_weights()
    cold.move_weights_to_device()

    # corrupt the manifest's recorded memcfg so the post-load guard mismatches
    key = wc._build_key(cold)
    backend = wc.get_backend()
    mpath = os.path.join(backend._module_dir(key.scope_digest(), key.module_digest()), "manifest.json")
    with open(mpath) as f:
        manifest = json.load(f)
    for ta in manifest["tensor_attrs"]:
        ta["memory_layout"] = "MemCfg(WRONG)"
    with open(mpath, "w") as f:
        json.dump(manifest, f)

    warm = _Leaf(payload=4)
    _bind(warm, device=dev, name="model.guard")
    warm.preprocess_weights()
    # host weight present so the fallback _impl does not AttributeError (B2)
    assert warm.tt_weight is not None  # preprocess ran
    ok = wc.try_load_module_weights(warm)
    assert ok is False
    assert wc.STATS.as_dict()["fell_back"] >= 1


# New cache bookkeeping flags are NOT flagged by the canary
def test_cache_bookkeeping_not_flagged():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    m = _Leaf(payload=1)
    _bind(m, device=dev, name="model.bk")
    # set all bookkeeping flags before the cold _impl runs
    m._weights_from_cache = False
    m._tt_cache_state = None
    m._configure_runtime_done = False
    m.preprocess_weights()
    m.move_weights_to_device()  # must not raise NonTensorStateMutationError


# LM-head bias signal (C3)
def test_lm_head_bias_signal():
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)

    class _BiasChunked(StatelessTTNNModule):
        def __init__(self, has_bias):
            super().__init__()
            self._src = torch.zeros(8, 8)
            self._has_bias = has_bias
            self._num_chunks = 0
            self._lm_head_has_bias = False

        def move_weights_to_device_impl(self):
            for i in range(2):
                setattr(self, f"tt_weight_chunk_{i}", _FakeDeviceTensor(payload=i, shape=(4, 4)))
                if self._has_bias:
                    setattr(self, f"tt_bias_chunk_{i}", _FakeDeviceTensor(payload=100 + i, shape=(1, 4)))

        def configure_runtime(self):
            self._num_chunks = 2
            self._lm_head_has_bias = self._has_bias

        def forward(self, x):
            return x

    cold = _BiasChunked(has_bias=True)
    _bind(cold, device=dev, name="model.lmbias")
    cold.preprocess_weights()
    cold.move_weights_to_device()
    assert cold._lm_head_has_bias is True

    warm = _BiasChunked(has_bias=True)
    _bind(warm, device=dev, name="model.lmbias")
    warm.preprocess_weights()
    warm.move_weights_to_device()
    assert warm._weights_from_cache is True
    assert getattr(warm, "tt_bias_chunk_0")._payload == 100

    # no-bias variant: different module digest (variant differs only if folded; here just sanity)
    nob = _BiasChunked(has_bias=False)
    _bind(nob, device=dev, name="model.lmnobias")
    nob.preprocess_weights()
    nob.move_weights_to_device()
    assert nob._lm_head_has_bias is False
    assert not hasattr(nob, "tt_bias_chunk_0")


# N2 discard on top-level exception
def test_discard_pending_on_top_level_exception(monkeypatch):
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")

    class _BadChild(_Leaf):
        def move_weights_to_device_impl(self):
            raise RuntimeError("boom")

    class _Composite(_Parent):
        pass

    order = []
    comp = _Composite(n_children=0, order_list=order)
    good = _Leaf(payload=1)
    bad = _BadChild(payload=2)
    comp.children_list = [good, bad]
    dev = _StubMeshDevice()
    _set_device_recursive(comp, dev)
    for m in (comp, good, bad):
        m._preprocessed_weight = True

    with pytest.raises(RuntimeError):
        comp.move_weights_to_device()
    # the good (enqueued) sibling was NOT configured; queue is empty; depth back to 0
    assert good._configure_runtime_done is False
    assert mod._PENDING_CONFIGURE == []
    assert mod._CONFIGURE_DEPTH == 0


# Byte-identical-when-disabled + clear_weight_cache
def test_disabled_is_byte_identical(monkeypatch):
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    m = _Leaf(payload=1)
    _bind(m, device=dev, name="model.disabled")
    m.preprocess_weights()
    m.move_weights_to_device()
    assert wc.module_cache_state(m) == wc.CacheState.DISABLED
    assert wc.STATS.as_dict()["stores"] == 0
    assert m._configure_runtime_done is True  # configure still runs (cold MISS depth-flush)


def test_clear_weight_cache(monkeypatch):
    dev = _StubMeshDevice(shape=(8, 1), num_devices=8)
    m = _Leaf(payload=1)
    _bind(m, device=dev, name="model.clear")
    m.preprocess_weights()
    m.move_weights_to_device()
    assert os.path.isdir(os.environ["TT_SYMBIOTE_CACHE_DIR"])
    wc.clear_weight_cache()
    assert not os.path.isdir(os.environ["TT_SYMBIOTE_CACHE_DIR"])


# O3: per-class dots_ocr migration verification (LANGUAGE + VISION, 17 classes)
import importlib  # noqa: E402
import inspect  # noqa: E402

_LINEAR = importlib.import_module("tt_symbiote.models.dots_ocr._linear")
_ATTN = importlib.import_module("tt_symbiote.models.dots_ocr.dots_ocr_attention")
_NORM = importlib.import_module("tt_symbiote.models.dots_ocr._normalization")
_MLP = importlib.import_module("tt_symbiote.models.dots_ocr.dots_ocr_mlp")
_DEC = importlib.import_module("tt_symbiote.models.dots_ocr.dots_ocr_decoder_layer")
_VIS = importlib.import_module("tt_symbiote.models.dots_ocr.dots_ocr_vision")

# (class, module, [config attr names that MUST be set by configure_runtime, NOT _impl])
_MIGRATED_LANGUAGE = [
    (_LINEAR.TTNNLinearInputShardedWeightSharded, ["compute_kernel_config"]),
    (_LINEAR.TTNNLinearInputReplicatedWeightSharded, ["compute_kernel_config"]),
    (_LINEAR.TTNNLinearLLamaIColShardedWAllReduced, ["compute_kernel_config", "_qkv_dram_input_shard_cfg",
                                                     "_bias_fused_into_matmul"]),
    (_LINEAR.TTNNLinearLLamaIColShardedWAllReducedFusedGateUp, ["compute_kernel_config",
                                                                "_gate_up_dram_input_shard_cfg",
                                                                "_gate_up_decode_compute_kernel_config"]),
    (_LINEAR.TTNNLinearLLamaIReplicatedWColSharded, ["_decode_input_shard_cfg", "compute_kernel_config"]),
    (_LINEAR.TTNNDotsOCRDRAMShardedLMHead, ["_num_tp", "_padded_vocab", "_size_per_device",
                                            "_chunk_n_cols_per_device", "_chunk_program_configs",
                                            "_input_shard_cfg", "compute_kernel_config", "_num_chunks",
                                            "_lm_head_has_bias"]),
    (_ATTN._TTNNDotsOCROProjPrefillLinear, ["_prefill_in0_mem", "_prefill_out_mem", "_prefill_pc"]),
    (_ATTN.TTNNDotsOCRAttention, ["core_grid", "_rotary_setup"]),
    (_NORM.TTNNDistributedRMSNorm, ["compute_kernel_config"]),
    (_MLP.TTNNDotsOCRFusedGateUpRowSharded, ["compute_kernel_config"]),
    (_MLP.TTNNDotsOCRRowShardedNoAllGather, ["compute_kernel_config", "_down_proj_dram_input_shard_cfg"]),
]

_MIGRATED_VISION = [
    (_VIS.TTNNDotsVisionRMSNorm, ["compute_kernel_config"]),
    (_VIS.TTNNDotsVisionMLP, ["compute_kernel_config"]),
    (_VIS.TTNNDotsVisionPatchEmbed, ["vision_matmul_compute_kernel_config",
                                     "vision_norm_compute_kernel_config", "_proj_k_padded"]),
    (_VIS.TTNNDotsVisionAttention, ["compute_kernel_config", "sdpa_compute_kernel_config"]),
    (_VIS.TTNNDotsPatchMerger, ["compute_kernel_config"]),
    (_VIS.TTNNDotsOCRVisionTower, ["rope"]),
]

_ALL_MIGRATED = _MIGRATED_LANGUAGE + _MIGRATED_VISION


@pytest.mark.parametrize("cls,cfg_attrs", _ALL_MIGRATED,
                         ids=[c.__name__ for c, _ in _ALL_MIGRATED])
def test_migrated_config_moved_out_of_impl(cls, cfg_attrs):
    """Every migrated class OWN-OVERRIDES configure_runtime, and the migrated NON-TENSOR config
    attrs are SET there, NOT in the _impl bodies.

    Source-level proof that the migration moved the offending writes (so the cold _impl is
    gate-clean and warm load recomputes the config). ``self.<attr> = `` must appear in
    configure_runtime and NOT in move_weights_to_device_impl / preprocess_weights_impl.
    """
    # Own-override (not just resolved via MRO) -- the base no-op is not enough.
    assert "configure_runtime" in cls.__dict__, f"{cls.__name__} must override configure_runtime"
    cfg_src = inspect.getsource(cls.configure_runtime)
    impl_src = ""
    if "move_weights_to_device_impl" in cls.__dict__:
        impl_src += inspect.getsource(cls.move_weights_to_device_impl)
    if "preprocess_weights_impl" in cls.__dict__:
        impl_src += inspect.getsource(cls.preprocess_weights_impl)
    for attr in cfg_attrs:
        # special case: rope is set via self.rope = cache[...]
        write_re = re.compile(rf"self\.{re.escape(attr)}\s*=")
        assert write_re.search(cfg_src), f"{cls.__name__}.configure_runtime must set self.{attr}"
        assert not write_re.search(impl_src), (
            f"{cls.__name__}: self.{attr} must NOT be written in an _impl (gate offender)"
        )


def test_layerstack_exclude_hook():
    """The language LayerStack excludes _shared_decode_cur_pos (no configure_runtime)."""
    cls = _DEC.TTNNDotsOCRLayerStack
    assert "weight_cache_excluded_attrs" in cls.__dict__
    # Build a bare instance to read the frozenset (no device needed).
    inst = cls.__new__(cls)
    assert "_shared_decode_cur_pos" in inst.weight_cache_excluded_attrs()
    assert "configure_runtime" not in cls.__dict__


def test_attention_exclude_hook_and_b2_single_owner():
    """Attention excludes _decode_cur_pos; B2: it does NOT write qkv_proj.compute_kernel_config."""
    cls = _ATTN.TTNNDotsOCRAttention
    inst = cls.__new__(cls)
    assert "_decode_cur_pos" in inst.weight_cache_excluded_attrs()
    cfg_src = inspect.getsource(cls.configure_runtime)
    # B2: parent must not WRITE qkv_proj.compute_kernel_config (a comment mentioning it is fine).
    assert not re.search(r"self\.qkv_proj\.compute_kernel_config\s*=", cfg_src), (
        "B2: parent must not set qkv_proj config"
    )
    # parent IS the single owner of sdpa.*
    assert "self.sdpa.program_config" in cfg_src
    assert "self.sdpa.compute_kernel_config" in cfg_src


def test_patch_merger_variant_folds_batched_full_hidden():
    cls = _VIS.TTNNDotsPatchMerger
    inst = cls.__new__(cls)
    inst._batched_full_hidden = False
    assert inst.weight_cache_variant() == "bf0"
    inst._batched_full_hidden = True
    assert inst.weight_cache_variant() == "bf1"


def test_lm_head_chunk_layout_helper_and_configure_recompute():
    """LM-head: pure chunk-math helper return contract (M1) + configure_runtime's
    _padded_vocab/_num_chunks/_lm_head_has_bias/_chunk_program_configs recompute."""
    cls = _LINEAR.TTNNDotsOCRDRAMShardedLMHead
    assert hasattr(cls, "_compute_chunk_layout")
    lm = cls()
    lm._device = _StubMeshDevice(shape=(8, 1), num_devices=8)
    lm.out_features = 152064
    lm.in_features = 1536
    lm._weight_torch = torch.zeros(152064, 1536)
    lm._bias_torch = None

    # Pure chunk-math helper: no self-write, returns the layout tuple.
    chunk_sizes, weight_t, num_tp, size_per_device, bias_t = lm._compute_chunk_layout()
    assert num_tp == 1
    assert size_per_device > 0
    assert sum(chunk_sizes) == size_per_device
    assert bias_t is None

    # configure_runtime recompute of the padded-vocab / chunk products.
    lm._configure_runtime_once()
    assert lm._padded_vocab >= lm.out_features
    assert lm._num_chunks >= 1
    assert lm._lm_head_has_bias is False
    assert len(lm._chunk_program_configs) == lm._num_chunks


# O3 / B1 / N1: composite-recursion canary exercises (live dots_ocr composites)
class _StubChildLeaf(StatelessTTNNModule):
    """A stub child that owns a cacheable tensor and a configure_runtime (like qkv/o_proj)."""

    def __init__(self):
        super().__init__()
        self._src = torch.zeros(4, 4)
        self.tt_weight = None
        self.compute_kernel_config = None

    def move_weights_to_device_impl(self):
        self.tt_weight = _FakeDeviceTensor(payload=1, shape=(4, 4))

    def configure_runtime(self):
        self.compute_kernel_config = ("child-HiFi2",)

    def forward(self, x):
        return x


class _StubSDPA(StatelessTTNNModule):
    """A stub SDPA child: config attrs default None in __init__ (parent is single owner)."""

    def __init__(self):
        super().__init__()
        self.program_config = None
        self.decode_program_config = None
        self.compute_kernel_config = None
        self.decode_compute_kernel_config = None

    def move_weights_to_device_impl(self):
        return self  # no own weights

    def forward(self, x):
        return x


def test_attention_composite_recursion_no_raise(monkeypatch):
    """TTNNDotsOCRAttention: driving the parent's move at top level must NOT raise (the deferred
    driver keeps every configure OUT of the parent's canary window); after the flush sdpa.* is set
    (parent single-owner) + qkv_proj owns its own compute_kernel_config (B2)."""
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")
    cls = _ATTN.TTNNDotsOCRAttention
    attn = cls.__new__(cls)
    StatelessTTNNModule.__init__(attn)
    dev = _StubMeshDevice(shape=(1, 1), num_devices=1)
    attn._device = dev
    attn.head_dim = 128
    attn.core_grid = None
    # children
    attn.qkv_proj = _StubChildLeaf()
    attn.o_proj = _StubChildLeaf()
    attn.sdpa = _StubSDPA()
    for c in (attn.qkv_proj, attn.o_proj, attn.sdpa):
        c._device = dev
        c._preprocessed_weight = True
    attn._preprocessed_weight = True
    # _fallback_torch_layer.config for the rotary setup
    cfg = type("C", (), {"rope_parameters": {}, "rope_theta": 1000000.0})()
    attn._fallback_torch_layer = type("L", (), {"config": cfg})()
    # patch BailingRotarySetup so configure_runtime does not need real device work
    monkeypatch.setattr(_ATTN, "BailingRotarySetup", lambda **k: ("rope-setup",), raising=True)

    attn.move_weights_to_device()  # MUST NOT raise

    assert attn._configure_runtime_done is True
    assert attn.sdpa.program_config is not None  # parent set sdpa.* (single owner)
    assert attn.sdpa.compute_kernel_config is not None
    assert attn.qkv_proj.compute_kernel_config == ("child-HiFi2",)  # child owns its own (B2)
    assert attn.core_grid is not None
    assert mod._CONFIGURE_DEPTH == 0


def test_vision_tower_composite_recursion_no_raise(monkeypatch):
    """TTNNDotsOCRVisionTower (N1): driving the tower's move at top level must NOT raise (the
    deferred driver keeps every vision leaf's configure out of the tower/block/stack canary
    windows); after the flush self.rope is rebuilt + the merger's build-time _batched_full_hidden
    is preserved through _impl."""
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")
    monkeypatch.setattr(_VIS, "TTNNDotsVision2DRoPE", lambda **k: ("rope2d",), raising=True)
    cls = _VIS.TTNNDotsOCRVisionTower
    tower = cls.__new__(cls)
    StatelessTTNNModule.__init__(tower)
    dev = _StubMeshDevice(shape=(1, 1), num_devices=1)
    tower._device = dev
    tower.head_dim = 128
    tower.spatial_merge_size = 2
    # children: patch_embed, block_stack(None -> use blocks list), post_trunk_norm, patch_merger
    tower.patch_embed = _StubChildLeaf()
    tower.block_stack = None
    tower.blocks = [_StubChildLeaf()]
    tower.post_trunk_norm = _StubChildLeaf()
    merger = _StubChildLeaf()
    merger._batched_full_hidden = False  # relayed at build time
    tower.patch_merger = merger
    for c in (tower.patch_embed, *tower.blocks, tower.post_trunk_norm, tower.patch_merger):
        c._device = dev
        c._preprocessed_weight = True
    tower._preprocessed_weight = True

    tower.move_weights_to_device()  # MUST NOT raise

    assert tower._configure_runtime_done is True
    assert tower.rope == ("rope2d",)
    # merger build-time flag preserved through _impl (tower no longer writes it)
    assert tower.patch_merger._batched_full_hidden is False
    assert mod._CONFIGURE_DEPTH == 0


def test_patch_embed_preprocess_is_gate_clean(monkeypatch):
    """TTNNDotsVisionPatchEmbed: preprocess no longer writes the _proj_k_padded int (gate-clean);
    configure_runtime recomputes it to the correct padded value."""
    monkeypatch.setenv("TT_SYMBIOTE_WEIGHT_CACHE", "0")
    cls = _VIS.TTNNDotsVisionPatchEmbed
    pe = cls()
    dev = _StubMeshDevice(shape=(1, 1), num_devices=1)
    pe._device = dev
    # proj weight with unpadded K (e.g. 588 = 3*14*14) -> pads to 608
    pe._proj_weight = torch.zeros(1536, 588)
    pe._proj_bias = None
    pe._norm_weight = None
    pe._preprocessed_weight = True

    pe.preprocess_weights()  # MUST NOT raise (int write removed from preprocess)
    assert pe.__dict__.get("_proj_k_padded") is None
    # configure_runtime recomputes the padded int from the surviving source weight shape
    pe._configure_runtime_once()
    assert pe._proj_k_padded == 608  # ((588 + 31)//32)*32
    assert pe.vision_matmul_compute_kernel_config is not None
    assert pe.vision_norm_compute_kernel_config is not None
