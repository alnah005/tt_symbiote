# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""D2DBridge -- a trace-enabled TTNNModule connecting two TTNNModules across two devices.

Every device tensor in ``module_a``'s output is moved chip->chip over an on-device fabric socket
(no host bounce) to ``module_b``'s device. Each module must already be ``set_device``-bound to its
own device and the two devices must differ. The bridge reports its OUTPUT mesh as its own ``.device``
so it can be chained as another bridge's ``module_a`` across N meshes. NORMAL = eager pytree D2D;
TRACED = two-mesh coordinated capture/replay.
"""
from __future__ import annotations

import ttnn

from tt_symbiote.core.d2d_transport import _L1, SocketTransport, _as_l1
from tt_symbiote.core.module import StatelessTTNNModule
from tt_symbiote.core.run_config import (
    TracedRun,
    get_tensor_run_implementation,
    is_trace_enabled,
    trace_enabled,
    trace_running,
)

TT_METAL_COMMIT = "2475f8f0cab858663cebccfad11a1728604c3ece"


def _device_ttnn(x):
    """Raw on-device ttnn.Tensor backing ``x``, or None if ``x`` is not a device tensor."""
    if isinstance(x, ttnn.Tensor):
        return x
    return getattr(x, "ttnn_tensor", None)


def _rewrap_like(template, recv):
    """Re-wrap a received raw ttnn.Tensor to match ``template``'s leaf type (best effort)."""
    if isinstance(template, ttnn.Tensor):
        return recv
    try:
        return type(template)(recv)  # e.g. TorchTTNNTensor(recv)
    except Exception:
        return recv


def _flatten(obj):
    """Flatten tuple/list/dict/namedtuple nesting into (ordered leaves, rebuild fn)."""
    if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):  # namedtuple
        nt = type(obj)
        return _join([_flatten(v) for v in obj], lambda parts: nt(*parts))
    if isinstance(obj, (tuple, list)):
        ctor = type(obj)
        return _join([_flatten(v) for v in obj], lambda parts: ctor(parts))
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return _join([_flatten(obj[k]) for k in keys], lambda parts: {k: p for k, p in zip(keys, parts)})
    return [obj], (lambda new: new[0])


def _join(kids, assemble):
    leaves = [leaf for sub, _ in kids for leaf in sub]
    sizes = [len(sub) for sub, _ in kids]
    rebuilds = [rb for _, rb in kids]

    def rebuild(new):
        parts, i = [], 0
        for sz, rb in zip(sizes, rebuilds):
            parts.append(rb(new[i : i + sz]))
            i += sz
        return assemble(parts)

    return leaves, rebuild


@trace_enabled
class D2DBridge(StatelessTTNNModule):
    """Connect two TTNNModules across two devices via on-device D2D sockets (see module docstring)."""

    def __init__(self, module_a, module_b, *, transport=None, tag=None, feed="auto", sync_on_return=False):
        super().__init__()
        if feed not in ("auto", "splat", "kwargs", "single"):
            raise ValueError(f"D2DBridge: bad feed={feed!r}")

        mesh_a = getattr(module_a, "device", None)
        mesh_b = getattr(module_b, "device", None)
        if mesh_a is None or mesh_b is None:
            raise ValueError(
                "D2DBridge: both modules must already be bound to a device before being passed to "
                "the bridge -- call set_device(module, mesh) on each first "
                f"(module_a.device={mesh_a!r}, module_b.device={mesh_b!r})."
            )
        if mesh_a is mesh_b:
            raise ValueError("D2DBridge: module_a and module_b must be on TWO DIFFERENT devices")

        # underscore attrs: a set_device() walk skips _-prefixed attrs and won't rebind pre-bound children.
        self._module_a = module_a
        self._module_b = module_b
        self._mesh_a = mesh_a
        self._mesh_b = mesh_b
        self._tag = tag
        self._feed = feed
        self._sync_on_return = sync_on_return
        self._transport = transport or SocketTransport()
        self._owns_transport = transport is None
        self._traces = {}
        self._warmed = set()
        self._bypass_tensor_wrapping = True
        # expose OUTPUT mesh as .device so a bridge can chain as another's module_a
        self._device = self._mesh_b

    def to_device(self, device):
        """No-op: children are pre-bound by the caller; the bridge takes no device of its own."""
        return self

    def set_device_state(self, device_state=None):
        # drain submeshes (never the parent): the bridge holds no single-mesh distributed state.
        return self

    @property
    def mesh_a(self):
        return self._mesh_a

    @property
    def mesh_b(self):
        return self._mesh_b

    # Edge accessors: a D2DBridge is a directed dependency edge (producer -> consumer).
    @property
    def producer(self):
        return self._module_a

    @property
    def consumer(self):
        return self._module_b

    @property
    def transport(self):
        return self._transport

    @property
    def tag(self):
        return self._tag

    def forward(self, *args, **kwds):
        raise RuntimeError("D2DBridge overrides call(); forward() should never be invoked.")

    def call(self, *args, **kwds):
        impl = get_tensor_run_implementation()  # fresh each call (avoid stale import snapshot)
        if impl is TracedRun and is_trace_enabled(self):
            return self._call_traced(args, kwds)
        return self._call_eager(args, kwds)

    __call__ = call

    def _feed_into(self, target, obj):
        if self._feed == "single":
            return target(obj)
        if self._feed == "kwargs" or (self._feed == "auto" and isinstance(obj, dict)):
            return target(**obj)
        if self._feed == "splat" or (self._feed == "auto" and isinstance(obj, (tuple, list))):
            return target(*obj)
        return target(obj)

    def _call_eager(self, args, kwds):
        out_a = self._module_a(*args, **kwds)
        leaves, rebuild = _flatten(out_a)
        new_leaves, ti = [], 0
        for leaf in leaves:
            dt = _device_ttnn(leaf)
            if dt is None:
                new_leaves.append(leaf)  # non-device leaf -> passthrough
                continue
            src = dt
            if src.memory_config().buffer_type != ttnn.BufferType.L1:
                src = ttnn.to_memory_config(src, _L1)  # socket needs L1 source (hidden from modules)
            recv = self._transport.send(src, self._mesh_b, tag=(self._tag, ti))
            new_leaves.append(_rewrap_like(leaf, recv))
            ti += 1
        out_b = self._feed_into(self._module_b, rebuild(new_leaves))
        if self._sync_on_return:
            ttnn.synchronize_device(self._mesh_b)
        return out_b

    def _call_traced(self, args, kwds):
        if not args or _device_ttnn(args[0]) is None:
            raise NotImplementedError(
                "D2DBridge TRACED mode requires the first positional arg to be the varying device "
                "tensor input. For other signatures, run in NORMAL (eager) mode."
            )
        key = TracedRun._make_cache_key(self.module_name, args)
        if key not in self._warmed:  # phase 1: warm-up (JIT + populate caches)
            out = self._traced_warmup(args, kwds)
            self._warmed.add(key)
            return out
        entry = self._traces.get(key)
        if entry is None:  # phase 2: capture, then replay once
            entry = self._capture(args, kwds, key)
            self._traces[key] = entry
        return self._replay(entry, args)  # phase 3+: replay

    def _traced_warmup(self, args, kwds):
        with trace_running():
            out_a = self._module_a.forward(*args, **kwds)
            leaves, rebuild = _flatten(out_a)
            new_leaves, ti = [], 0
            for leaf in leaves:
                dt = _device_ttnn(leaf)
                if dt is None:
                    new_leaves.append(leaf)
                    continue
                src = self._as_l1(dt)
                recv = self._transport.send(src, self._mesh_b, tag=(self._tag, ti))
                new_leaves.append(_rewrap_like(leaf, recv))
                ti += 1
            out_b = self._feed_into(self._module_b.forward, rebuild(new_leaves))
        ttnn.synchronize_device(self._mesh_a)
        ttnn.synchronize_device(self._mesh_b)
        return out_b

    @staticmethod
    def _as_l1(t):
        return _as_l1(t)

    def _capture(self, args, kwds, key):
        x = args[0]
        with trace_running():
            tid_a = ttnn.begin_trace_capture(self._mesh_a, cq_id=0)
            out_a = self._module_a.forward(x, *args[1:], **kwds)
            leaves, rebuild = _flatten(out_a)
            plan, ti = [], 0
            for leaf in leaves:
                dt = _device_ttnn(leaf)
                if dt is None:
                    plan.append(("host", leaf))
                    continue
                # trace records only -- no device alloc inside; pair/buffer caches pre-warmed
                assert dt.memory_config().buffer_type == ttnn.BufferType.L1, (
                    "D2DBridge TRACED mode requires module_a's device outputs to be L1-resident "
                    "(DRAM->L1 coercion would allocate inside the trace). Output L1 from module_a."
                )
                send_sock, recv_sock, out_buf = self._transport.prepare(dt, self._mesh_b, tag=(self._tag, ti))
                self._transport.send_only(dt, send_sock)
                plan.append(("dev", recv_sock, out_buf, leaf))
                ti += 1
            ttnn.end_trace_capture(self._mesh_a, tid_a)
            tid_b = ttnn.begin_trace_capture(self._mesh_b, cq_id=0)
            new_leaves = []
            for item in plan:
                if item[0] == "host":
                    new_leaves.append(item[1])
                else:
                    _, recv_sock, out_buf, tmpl = item
                    self._transport.recv_only(out_buf, recv_sock)
                    new_leaves.append(_rewrap_like(tmpl, out_buf))
            out_b = self._feed_into(self._module_b.forward, rebuild(new_leaves))
            ttnn.end_trace_capture(self._mesh_b, tid_b)
        return {"tid_a": tid_a, "tid_b": tid_b, "in": x, "out_b": out_b}

    def _replay(self, entry, args):
        new_x = args[0]
        if new_x is not entry["in"]:
            ttnn.copy(new_x, entry["in"])  # refresh the captured input buffer
        # blocking=False on BOTH is mandatory: a blocking send replay alone deadlocks (no receiver).
        ttnn.execute_trace(self._mesh_a, entry["tid_a"], cq_id=0, blocking=False)
        ttnn.execute_trace(self._mesh_b, entry["tid_b"], cq_id=0, blocking=False)
        if self._sync_on_return:
            ttnn.synchronize_device(self._mesh_a)
            ttnn.synchronize_device(self._mesh_b)
        return entry["out_b"]

    def release_traces(self):
        for entry in self._traces.values():
            try:
                ttnn.release_trace(self._mesh_a, entry["tid_a"])
            except Exception:
                pass
            try:
                ttnn.release_trace(self._mesh_b, entry["tid_b"])
            except Exception:
                pass
        self._traces.clear()
        self._warmed.clear()

    def close(self):
        # Drain both submeshes so no pending d2d work leaves cq 0 "in use" at submesh/parent teardown.
        for m in (self._mesh_a, self._mesh_b):
            if m is not None:
                try:
                    ttnn.synchronize_device(m)
                except Exception:
                    pass
        self.release_traces()
        if self._owns_transport:
            self._transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
