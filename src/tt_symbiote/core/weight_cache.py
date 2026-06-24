# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Disk weight-cache for TTNN modules.

Persists each module's preprocessed, on-device, FLAT ``ttnn.Tensor`` weight attributes to disk
(``ttnn.dump_tensor`` / ``ttnn.load_tensor``) so a warm load skips the torch->TTNN preprocessing
(tilize / pad / shard) inside ``move_weights_to_device_impl``.

Design: only flat ``ttnn.Tensor`` attrs are cached; NON-tensor state is recomputed every load by
``configure_runtime()`` (driven from ``core/module.py``); containers flatten to indexed attrs
(``tt_weight_chunk_0``/...); fail-open / byte-identical when disabled. ``none_slots`` (manifest
schema 2) records the keys the cold ``_impl`` left ``None``; the loader re-materializes them AFTER
restoring device tensors and BEFORE ``configure_runtime`` (LOAD-BEARING:
``configure_runtime``/forward read e.g. ``self.tt_bias`` plain).

Cache key (``WeightCacheKey``) = two SHA1 digests; a warm HIT requires both to match what was stored.
``scope_digest`` (shared per run, identifies the hardware target): 1. ``device_arch`` -- ``MESH_DEVICE``
env; 2. ``mesh_shape`` -- ``module.device.shape``; 3. ``num_devices`` -- ``device.get_num_devices()``.
``module_digest`` (per module, identifies weights + how built): 4. ``module_name_path`` --
``module._unique_name``; 5. ``module_subclass`` -- fq ``module.__class__``; 6. ``variant`` --
``module.weight_cache_variant()`` (``""`` default); 7. ``src_fingerprint`` -- shapes+dtypes of ALL
``torch.Tensor`` attrs, snapshotted at preprocess-start (naming-independent; see
``capture_src_fingerprint``); 8. ``preprocess_source_fp`` -- SHA1 of ``inspect.getsource`` of
``preprocess_weights_impl`` + ``move_weights_to_device_impl`` + ``configure_runtime``;
9. ``model_config_fingerprint`` -- SHA1 of ``_model_config[_unique_name]``.

Cache layout (``$TT_SYMBIOTE_CACHE_DIR``, default ``~/.cache/tt_symbiote/weights``):
``<scope_digest>/{scope.json, <module_digest>/{manifest.json, *.tensorbin}}``; writes go to a
transient ``<module_digest>.tmp.<pid>.<rand>/`` dir ``os.replace()``d on commit (atomic).

Env vars: ``TT_SYMBIOTE_WEIGHT_CACHE=0`` fully disables; ``TT_SYMBIOTE_CACHE_DIR`` sets the root.

Run-mode matrix (read LIVE from ``run_config``): NORMAL / NORMAL_WITH_FALLBACK / TRACED ->
read+write; DPL / DPL_NO_ERROR_PROP / SEL -> read-bypass (write opt-in); LIGHTWEIGHT / CPU -> off.
"""

import dataclasses
import enum
import hashlib
import inspect
import json
import os
import re
import threading

from loguru import logger

# Records which tt-metal version this cache was developed against.
TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"

_DEFAULT_CACHE_DIR = os.path.join("~", ".cache", "tt_symbiote", "weights")
_MANIFEST_SCHEMA = 2  # bumped 1->2 for the `none_slots` field
_TENSORBIN_EXT = ".tensorbin"

# Debug-comparison modes BYPASS reads (may still write, opt-in); OFF modes disable the cache.
_READ_BYPASS_MODES = frozenset({"DPL", "DPL_NO_ERROR_PROP", "SEL"})
_OFF_MODES = frozenset({"LIGHTWEIGHT", "CPU"})

# Non-deterministic id-fallback name form produced by ``module_name`` when ``_unique_name`` is
# unset: ``<ClassName>_<id>`` (determinism guard).
_ID_FALLBACK_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*_\d+$")

_warn_once_seen: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _warn_once_seen:
        return
    _warn_once_seen.add(key)
    logger.warning(msg)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class CacheStats:
    """Thread-safe process-global counters. ``passthrough`` counts modules owning ZERO cacheable
    tensors (pure composites: neither load nor store, only run ``_impl`` to recurse) -- NOT misses."""

    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.bypass = 0
        self.fell_back = 0
        self.passthrough = 0

    def record_hit(self):
        with self._lock:
            self.hits += 1

    def record_miss(self):
        with self._lock:
            self.misses += 1

    def record_store(self):
        with self._lock:
            self.stores += 1

    def record_bypass(self):
        with self._lock:
            self.bypass += 1

    def record_fell_back(self):
        with self._lock:
            self.fell_back += 1

    def record_passthrough(self):
        with self._lock:
            self.passthrough += 1

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
                "bypass": self.bypass,
                "fell_back": self.fell_back,
                "passthrough": self.passthrough,
            }

    def reset(self):
        with self._lock:
            self.hits = self.misses = self.stores = self.bypass = self.fell_back = self.passthrough = 0


STATS = CacheStats()


# ---------------------------------------------------------------------------
# Run-mode + env gating (stdlib only; reads LIVE from run_config)
# ---------------------------------------------------------------------------
def _current_run_mode() -> str:
    """Resolve the active run mode name (LIVE from env + run_config)."""
    mode = os.environ.get("TT_SYMBIOTE_RUN_MODE")
    if mode:
        return mode
    try:
        from tt_symbiote.core import run_config as rc

        return getattr(rc, "_current_run_mode", None) or "NORMAL"
    except Exception:
        return "NORMAL"


def _cache_env_disabled() -> bool:
    return os.environ.get("TT_SYMBIOTE_WEIGHT_CACHE", "1") == "0"


def caching_enabled_for_reads() -> bool:
    if _cache_env_disabled():
        return False
    mode = _current_run_mode()
    if mode in _OFF_MODES or mode in _READ_BYPASS_MODES:
        if mode in _READ_BYPASS_MODES:
            STATS.record_bypass()
        return False
    return True


def caching_enabled_for_writes() -> bool:
    if _cache_env_disabled():
        return False
    mode = _current_run_mode()
    if mode in _OFF_MODES:
        return False
    # Writes on for all non-off modes (DPL/SEL too, so a debug run warms the cache).
    return True


# ---------------------------------------------------------------------------
# Tensor identity (REUSE the gate predicate) + device-residency capture
# ---------------------------------------------------------------------------
def _is_real_ttnn_tensor(v) -> bool:
    # REUSE the gate's positive-type-identity predicate (stub-safe via the registered fakes).
    from tt_symbiote.core.module import _is_real_ttnn_tensor as _gate_pred

    return _gate_pred(v)


def _is_device_resident(v) -> bool:
    """DEVICE-residency filter; stub-safe fallback (SW test monkeypatches ``StorageType.DEVICE``)."""
    try:
        import ttnn

        st = v.storage_type()
        return st == ttnn.StorageType.DEVICE
    except Exception:
        return True  # name-based host exclusion is applied by the caller


def cacheable_tensor_attrs(module) -> dict:
    """``{attr: tensor}`` for flat device-resident ttnn.Tensor attrs, minus ``*_host`` and the
    module's ``weight_cache_excluded_attrs()``. Containers are never traversed (flattened upstream)."""
    excluded = set(module.weight_cache_excluded_attrs())
    out = {}
    for k, v in list(module.__dict__.items()):
        if k in excluded:
            continue
        if k.endswith("_host"):  # host stash, never cached (re-derived in preprocess)
            continue
        if not _is_real_ttnn_tensor(v):
            continue
        if not _is_device_resident(v):
            continue
        out[k] = v
    return out


def owns_cached_tensors(module) -> bool:
    """True iff this module owns >=1 cacheable tensor (live) OR a manifest with owned_tensor_count>0."""
    if cacheable_tensor_attrs(module):
        return True
    try:
        key = _build_key(module)
        if key is None:
            return False
        manifest = get_backend().read_manifest(key.scope_digest(), key.module_digest())
        return bool(manifest) and int(manifest.get("owned_tensor_count", 0)) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cache key + fingerprints
# ---------------------------------------------------------------------------
class CacheState(enum.Enum):
    HIT = "hit"
    MISS = "miss"
    DISABLED = "disabled"


@dataclasses.dataclass(frozen=True)
class WeightCacheKey:
    device_arch: str
    mesh_shape: tuple
    num_devices: int
    module_subclass: str
    module_name_path: str
    variant: str
    src_fingerprint: str
    preprocess_source_fp: str
    model_config_fingerprint: str

    def scope_digest(self) -> str:
        payload = {
            "device_arch": self.device_arch,
            "mesh_shape": list(self.mesh_shape),
            "num_devices": self.num_devices,
        }
        return hashlib.sha1(_canonical_json(payload).encode()).hexdigest()[:16]

    def module_digest(self) -> str:
        payload = {
            "module_name_path": self.module_name_path,
            "module_subclass": self.module_subclass,
            "variant": self.variant,
            "src_fingerprint": self.src_fingerprint,
            "preprocess_source_fp": self.preprocess_source_fp,
            "model_config_fingerprint": self.model_config_fingerprint,
        }
        return hashlib.sha1(_canonical_json(payload).encode()).hexdigest()[:16]


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _device_arch() -> str:
    try:
        from tt_symbiote.core.module import MeshShapeToDeviceArch

        arch = MeshShapeToDeviceArch.get(os.environ.get("MESH_DEVICE"))
        if arch is None:
            return "unknown"
        return getattr(arch, "value", str(arch))
    except Exception:
        return "unknown"


def _mesh_shape_and_count(module):
    dev = getattr(module, "device", None)
    mesh_shape = ()
    num_devices = 1
    try:
        shape = getattr(dev, "shape", None)
        if shape is not None:
            mesh_shape = tuple(int(x) for x in shape)
    except Exception:
        mesh_shape = ()
    try:
        num_devices = int(dev.get_num_devices())
    except Exception:
        num_devices = 1
    return mesh_shape, num_devices


def _torch_src_digest(module) -> str:
    """Naming-independent shapes/dtypes hash over EVERY torch.Tensor attr on the module."""
    import torch

    parts = []
    for k, v in sorted(module.__dict__.items()):
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}:{tuple(v.shape)}:{v.dtype}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def capture_src_fingerprint(module) -> None:
    """Snapshot the torch source-weight fingerprint at preprocess-start, BEFORE
    ``preprocess_weights_impl`` creates the transient ``*_host`` stash. SOLE basis for the key's
    ``src_fingerprint``, so the digest is identical at the warm read-check (stash live) and the cold
    save (stash freed)."""
    try:
        module._tt_src_fp = _torch_src_digest(module)
    except Exception:
        pass


def _src_fingerprint(module) -> str:
    """Source-weight fingerprint for the cache key: the preprocess-start snapshot from
    ``capture_src_fingerprint``; live-scan fallback only when no snapshot exists."""
    fp = getattr(module, "_tt_src_fp", None)
    return fp if fp is not None else _torch_src_digest(module)


def _getsource_or_cocode(fn) -> str:
    try:
        return inspect.getsource(fn)
    except Exception:
        try:
            return repr(getattr(fn, "__code__", fn).co_code)
        except Exception:
            return repr(fn)


def _preprocess_source_fp(module) -> str:
    t = type(module)
    src = (
        _getsource_or_cocode(t.preprocess_weights_impl)
        + _getsource_or_cocode(t.move_weights_to_device_impl)
        + _getsource_or_cocode(t.configure_runtime)
    )
    return hashlib.sha1(src.encode()).hexdigest()[:16]


def _model_config_fingerprint(module) -> str:
    cfg = getattr(module, "_model_config", {}) or {}
    name = getattr(module, "_unique_name", None)
    sub = cfg.get(name, {}) if isinstance(cfg, dict) else {}
    return hashlib.sha1(_canonical_json(sub).encode()).hexdigest()[:12]


def _build_key(module):
    """Construct a :class:`WeightCacheKey`; returns ``None`` if the module name is non-deterministic."""
    name = getattr(module, "_unique_name", None)  # DIRECT read (never the module_name property)
    if name is None or _ID_FALLBACK_NAME_RE.match(name):
        return None
    mesh_shape, num_devices = _mesh_shape_and_count(module)
    return WeightCacheKey(
        device_arch=_device_arch(),
        mesh_shape=mesh_shape,
        num_devices=num_devices,
        module_subclass=f"{type(module).__module__}.{type(module).__qualname__}",
        module_name_path=name,
        variant=str(module.weight_cache_variant()),
        src_fingerprint=_src_fingerprint(module),
        preprocess_source_fp=_preprocess_source_fp(module),
        model_config_fingerprint=_model_config_fingerprint(module),
    )


# ---------------------------------------------------------------------------
# Local-filesystem weight cache backend
# ---------------------------------------------------------------------------
def _cache_root() -> str:
    root = os.environ.get("TT_SYMBIOTE_CACHE_DIR") or _DEFAULT_CACHE_DIR
    return os.path.expanduser(root)


class _LocalFsWriteHandle:
    def __init__(self, final_dir, tmp_dir):
        self._final_dir = final_dir
        self._tmp_dir = tmp_dir
        os.makedirs(self._tmp_dir, exist_ok=True)

    def tensor_path_for(self, member) -> str:
        return os.path.join(self._tmp_dir, f"{member}{_TENSORBIN_EXT}")

    def write_manifest(self, manifest: dict) -> None:
        # complete:true is written LAST so a crash mid-dump leaves no reader-addressable dir.
        with open(os.path.join(self._tmp_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, default=str)

    def commit(self) -> None:
        if os.path.isdir(self._final_dir):
            import shutil

            shutil.rmtree(self._final_dir, ignore_errors=True)
        os.replace(self._tmp_dir, self._final_dir)  # atomic dir rename (same filesystem)

    def abort(self) -> None:
        import shutil

        shutil.rmtree(self._tmp_dir, ignore_errors=True)


class LocalFsWeightCacheBackend:
    """Local-filesystem weight cache under ``$TT_SYMBIOTE_CACHE_DIR``."""

    def __init__(self, root=None):
        self._root = root or _cache_root()

    def _module_dir(self, scope_digest, module_digest) -> str:
        return os.path.join(self._root, scope_digest, module_digest)

    def _scope_dir(self, scope_digest) -> str:
        return os.path.join(self._root, scope_digest)

    def has_complete(self, scope_digest, module_digest) -> bool:
        manifest = self.read_manifest(scope_digest, module_digest)
        return bool(manifest) and manifest.get("complete") is True

    def read_manifest(self, scope_digest, module_digest):
        path = os.path.join(self._module_dir(scope_digest, module_digest), "manifest.json")
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def begin_write(self, scope_digest, module_digest) -> "_LocalFsWriteHandle":
        final_dir = self._module_dir(scope_digest, module_digest)
        os.makedirs(self._scope_dir(scope_digest), exist_ok=True)
        rand = os.urandom(4).hex()
        tmp_dir = f"{final_dir}.tmp.{os.getpid()}.{rand}"
        # Write a minimal scope.json (informational).
        try:
            scope_json = os.path.join(self._scope_dir(scope_digest), "scope.json")
            if not os.path.exists(scope_json):
                with open(scope_json, "w") as f:
                    json.dump({"scope_digest": scope_digest, "schema": _MANIFEST_SCHEMA}, f)
        except Exception:
            pass
        return _LocalFsWriteHandle(final_dir, tmp_dir)

    def purge(self, scope_digest, module_digest) -> None:
        import shutil

        shutil.rmtree(self._module_dir(scope_digest, module_digest), ignore_errors=True)


_backend_singleton = None
_backend_lock = threading.Lock()


def get_backend():
    global _backend_singleton
    with _backend_lock:
        if _backend_singleton is None:
            _backend_singleton = LocalFsWeightCacheBackend()
        return _backend_singleton


def _reset_backend_for_tests():
    global _backend_singleton
    with _backend_lock:
        _backend_singleton = None


# ---------------------------------------------------------------------------
# State resolution (memoized + DISABLED-fast)
# ---------------------------------------------------------------------------
def _cache_dir_writable() -> bool:
    root = _cache_root()
    try:
        os.makedirs(root, exist_ok=True)
        return os.access(root, os.W_OK)
    except Exception:
        return False


def module_cache_state(module) -> CacheState:
    """Resolve HIT|MISS|DISABLED (memoized on ``module._tt_cache_state``). DISABLED-fast:
    short-circuits before any fingerprinting when both gates off / name non-deterministic."""
    cached = getattr(module, "_tt_cache_state", None)
    if cached is not None:
        return cached

    # DISABLED-fast: no fingerprinting at all when fully off.
    if not caching_enabled_for_reads() and not caching_enabled_for_writes():
        module._tt_cache_state = CacheState.DISABLED
        return CacheState.DISABLED

    name = getattr(module, "_unique_name", None)
    if name is None or _ID_FALLBACK_NAME_RE.match(name):
        _warn_once(
            f"nondet-{type(module).__qualname__}",
            f"weight_cache: DISABLED for {type(module).__qualname__} (non-deterministic _unique_name={name!r})",
        )
        module._tt_cache_state = CacheState.DISABLED
        return CacheState.DISABLED

    if not _cache_dir_writable():
        _warn_once("unwritable", f"weight_cache: cache dir {_cache_root()!r} unwritable; DISABLED")
        module._tt_cache_state = CacheState.DISABLED
        return CacheState.DISABLED

    key = _build_key(module)
    if key is None:
        module._tt_cache_state = CacheState.DISABLED
        return CacheState.DISABLED
    try:
        complete = get_backend().has_complete(key.scope_digest(), key.module_digest())
    except Exception:
        complete = False
    state = CacheState.HIT if complete else CacheState.MISS
    # miss vs passthrough is classified at save time (ownership known only after _impl runs).
    module._tt_cache_state = state
    return state


# ---------------------------------------------------------------------------
# Dump / load + post-load guard
# ---------------------------------------------------------------------------
def _memcfg_repr(tensor):
    try:
        return repr(tensor.tensor_spec().memory_config())
    except Exception:
        return None


def _shape_tuple(tensor):
    try:
        return list(int(x) for x in tensor.shape)
    except Exception:
        return None


def save_module_weights(module, none_slots=None) -> None:
    """Persist flat cacheable device tensors + the cold-``None`` slot KEYS (schema 2); best-effort /
    fail-open. ``none_slots`` (key names, not tensors) ride ONLY on a manifest written because the
    module owns >=1 cacheable tensor (the ``if not members: return`` early-return is KEPT)."""
    import ttnn  # noqa: F401  (lazy; matches the gate's lazy ttnn import)

    members = cacheable_tensor_attrs(module)
    if not members:
        # 0-cacheable-tensor module (pure composite): no manifest, only ran _impl to recurse into
        # children -> a PASSTHROUGH, not a miss.
        STATS.record_passthrough()
        return
    key = _build_key(module)
    if key is None:
        return
    # A tensor-owner reached the cold materialize path -> no usable cache entry was loaded -> MISS.
    STATS.record_miss()
    backend = get_backend()
    mesh_shape, _num = _mesh_shape_and_count(module)
    handle = backend.begin_write(key.scope_digest(), key.module_digest())
    try:
        tensor_attrs = []
        for attr, tensor in members.items():
            path = handle.tensor_path_for(attr)
            logger.debug(f"weight_cache MISS {key.module_name_path}: dumping {attr} -> {path} (shape {_shape_tuple(tensor)})")
            ttnn.dump_tensor(path, tensor)
            tensor_attrs.append(
                {
                    "name": attr,
                    "file": f"{attr}{_TENSORBIN_EXT}",
                    "shape": _shape_tuple(tensor),
                    "memory_layout": _memcfg_repr(tensor),
                }
            )
        # none_slots: cold-None keys, minus any also-cached tensor name (a slot is None XOR a tensor).
        tensor_names = {a["name"] for a in tensor_attrs}
        recorded_none = sorted(k for k in (none_slots or []) if k not in tensor_names)
        manifest = {
            "schema": _MANIFEST_SCHEMA,
            "owned_tensor_count": len(tensor_attrs),
            "key": dataclasses.asdict(key),
            "mesh_shape": list(mesh_shape),
            "tensor_attrs": tensor_attrs,
            "none_slots": recorded_none,  # NEW (schema 2): cold-None forward/configure-read slot KEYS
            "complete": True,  # written LAST inside write_manifest
        }
        handle.write_manifest(manifest)
        handle.commit()
        STATS.record_store()
    except Exception as e:
        handle.abort()
        _warn_once(f"save-{type(module).__qualname__}", f"weight_cache: save failed for {key.module_name_path}: {e!r}")


def try_load_module_weights(module) -> bool:
    """Load cached device tensors; True on success. Post-load guard (memcfg/shape/mesh-shape): on
    any mismatch deallocate, record FELL-BACK, return False (-> cold recompute)."""
    import ttnn

    key = _build_key(module)
    if key is None:
        return False
    backend = get_backend()
    manifest = backend.read_manifest(key.scope_digest(), key.module_digest())
    if not manifest or manifest.get("complete") is not True:
        return False
    # Load-path-only schema guard: a stale schema-1 manifest -> re-MISS; has_complete stays schema-agnostic.
    if int(manifest.get("schema", 0)) != _MANIFEST_SCHEMA:
        return False

    loaded = {}
    try:
        cur_mesh, _num = _mesh_shape_and_count(module)
        if list(cur_mesh) != list(manifest.get("mesh_shape", [])):
            raise RuntimeError(
                f"mesh-shape mismatch: cur {list(cur_mesh)} vs manifest {manifest.get('mesh_shape')}"
            )
        for entry in manifest.get("tensor_attrs", []):
            path = os.path.join(
                backend._module_dir(key.scope_digest(), key.module_digest()), entry["file"]
            )
            t = ttnn.load_tensor(path, device=module.device)
            # post-load memcfg + shape guard
            got_memcfg = _memcfg_repr(t)
            want_memcfg = entry.get("memory_layout")
            if want_memcfg is not None and got_memcfg is not None and got_memcfg != want_memcfg:
                raise RuntimeError(f"memcfg mismatch for {entry['name']}: {got_memcfg} != {want_memcfg}")
            got_shape = _shape_tuple(t)
            want_shape = entry.get("shape")
            if want_shape is not None and got_shape is not None and got_shape != want_shape:
                raise RuntimeError(f"shape mismatch for {entry['name']}: {got_shape} != {want_shape}")
            logger.debug(f"weight_cache HIT {key.module_name_path}: loaded {entry['name']} <- {path} (shape {got_shape})")
            loaded[entry["name"]] = t
        for name, t in loaded.items():
            setattr(module, name, t)
        # LOAD-BEARING order: device tensors FIRST (above), then none_slots HERE, then the driver
        # runs configure_runtime. `if k not in loaded` never clobbers a just-restored REAL tensor.
        for k in manifest.get("none_slots", []):
            if k not in loaded:
                setattr(module, k, None)
        STATS.record_hit()
        return True
    except Exception as e:
        for t in loaded.values():
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        STATS.record_fell_back()
        _warn_once(
            f"load-{type(module).__qualname__}",
            f"weight_cache: load fell back for {key.module_name_path}: {e!r}",
        )
        return False


def purge(module) -> None:
    key = _build_key(module)
    if key is None:
        return
    try:
        get_backend().purge(key.scope_digest(), key.module_digest())
    except Exception:
        pass
    # Force a re-resolve (a purged module should re-MISS on the same process).
    module._tt_cache_state = CacheState.MISS


def clear_weight_cache() -> None:
    """Delete the entire on-disk cache root + reset process-global stats/memoization."""
    import shutil

    root = _cache_root()
    try:
        shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
    STATS.reset()
    _warn_once_seen.clear()
