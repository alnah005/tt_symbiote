# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pipeline -- run a chain of stages whose dependencies are encoded as D2DBridge edges.

Each D2DBridge ``(producer, consumer)`` is a directed dependency edge; ``Pipeline(bridges)``
assembles them into a dataflow graph and runs each stage once in topological order (derived from the
edges, not list position). TRACED mode captures ONE trace per DISTINCT physical mesh (a revisited
mesh merges into its single trace). Deadlock-free: every per-mesh trace replays ``blocking=False``
with one joint ``synchronize_device`` after all are launched, so each send finds its queued recv.
"""
from __future__ import annotations

import ttnn

from tt_symbiote.core.d2d_bridge import _device_ttnn, _rewrap_like
from tt_symbiote.core.d2d_transport import _as_l1
from tt_symbiote.core.module import StatelessTTNNModule
from tt_symbiote.core.run_config import (
    TracedRun,
    get_tensor_run_implementation,
    is_trace_enabled,
    trace_enabled,
    trace_running,
)

TT_METAL_COMMIT = "2475f8f0cab858663cebccfad11a1728604c3ece"


@trace_enabled
class Pipeline(StatelessTTNNModule):
    """Chain of stages whose dependencies are encoded as D2DBridge edges (see module docstring)."""

    # Central registries (mirror TracedRun._trace_cache): every captured trace and every socket
    # transport is tracked here so release_all() frees them in one shot, independent of per-instance
    # close(). release_trace()/transport.close() need a live device -> call release_all() BEFORE the
    # mesh is closed (e.g. in the test fixture teardown, alongside TracedRun.release_all()).
    _live_traces: list = []
    _live_transports: list = []

    @classmethod
    def _track_traces(cls, tids):
        cls._live_traces.extend(tids or ())

    @classmethod
    def _track_transport(cls, transport):
        if transport is not None and all(transport is not t for t in cls._live_transports):
            cls._live_transports.append(transport)

    @classmethod
    def _untrack_traces(cls, tids):
        drop = {id(e) for e in (tids or ())}
        cls._live_traces = [e for e in cls._live_traces if id(e) not in drop]

    @classmethod
    def release_all(cls):
        """Release every tracked trace and close every tracked socket transport (mirrors
        TracedRun.release_all). The catch-all for resources a test/process did not release
        explicitly; idempotent with the per-instance release_loop()/close() paths."""
        for m, tid in cls._live_traces:
            try:
                ttnn.release_trace(m, tid)
            except Exception:
                pass
        cls._live_traces = []
        seen = set()
        for t in cls._live_transports:
            if id(t) in seen:
                continue
            seen.add(id(t))
            try:
                t.close()
            except Exception:
                pass
        cls._live_transports = []

    def __init__(self, bridges, *, sync_on_return=False):
        super().__init__()
        bridges = list(bridges)
        if not bridges:
            raise ValueError("Pipeline needs at least one D2DBridge edge")

        mods = {}
        out_edge = {}
        indeg = {}
        for b in bridges:
            p, c = b.producer, b.consumer
            mods[id(p)] = p
            mods[id(c)] = c
            if id(p) in out_edge:
                raise ValueError(
                    "Pipeline v1 supports a LINEAR chain; a stage fans out to >1 edge "
                    f"({type(p).__name__}). Branching DAGs are a future extension."
                )
            out_edge[id(p)] = (c, b)
            indeg[id(c)] = indeg.get(id(c), 0) + 1
            indeg.setdefault(id(p), indeg.get(id(p), 0))
        if any(v > 1 for v in indeg.values()):
            raise ValueError(
                "Pipeline v1 supports a LINEAR chain; a stage has >1 incoming edge "
                "(multi-input join). Branching DAGs are a future extension."
            )

        sources = [m for mid, m in mods.items() if indeg.get(mid, 0) == 0]
        if len(sources) != 1:
            raise ValueError(
                f"Pipeline: expected exactly one source stage, found {len(sources)} "
                "(the edges must form a single connected chain)."
            )

        # topological order: follow the unique out-edge from the source.
        order, hop_in = [sources[0]], [None]
        cur = sources[0]
        while id(cur) in out_edge:
            nxt, b = out_edge[id(cur)]
            order.append(nxt)
            hop_in.append(b)
            cur = nxt
        if len(order) != len(mods):
            raise ValueError(
                "Pipeline: the bridges do not form a single connected chain " "(disconnected component or cycle)."
            )

        self._stages = order
        self._hop_in = hop_in
        self._meshes = [getattr(s, "device", None) for s in order]
        for i, d in enumerate(self._meshes):
            if d is None:
                raise ValueError(
                    f"Pipeline: stage[{i}] ({type(order[i]).__name__}) is not bound; "
                    "set_device(stage, mesh) on every stage before wiring the bridges."
                )
        self._sync_on_return = sync_on_return
        self._bypass_tensor_wrapping = True
        self._device = self._meshes[-1]  # output mesh (so a Pipeline can chain like a stage)
        self._warmed = set()
        self._x_dev = None
        self._hop_sock = None
        self._tids = None
        self._out_last = None
        for b in self._hop_in:  # track each hop's socket transport for central release
            if b is not None:
                Pipeline._track_transport(b.transport)

    @property
    def stages(self):
        return list(self._stages)

    @property
    def meshes(self):
        return list(self._meshes)

    def to_device(self, device):
        return self

    def set_device_state(self, device_state=None):
        return self

    def forward(self, *a, **k):
        raise RuntimeError("Pipeline overrides call(); forward() should never be invoked.")

    def call(self, x):
        impl = get_tensor_run_implementation()
        if impl is TracedRun and is_trace_enabled(self):
            return self._traced(x)
        return self._eager(x)

    __call__ = call

    def _as_l1(self, t):
        return _as_l1(t)

    def _src1(self, out, where):
        src = _device_ttnn(out)
        if src is None:
            raise TypeError(
                f"Pipeline ({where}): a stage returned a non/multi-tensor output. Pipeline "
                "hands off ONE tensor per edge; use D2DBridge directly for multi-tensor hops."
            )
        return src

    def _eager(self, x):
        out = self._stages[0](x)
        for i in range(1, len(self._stages)):
            b = self._hop_in[i]
            recv = b.transport.send(self._as_l1(self._src1(out, "eager")), b.mesh_b, tag=b.tag)
            out = self._stages[i](_rewrap_like(out, recv))
        if self._sync_on_return:
            ttnn.synchronize_device(self._meshes[-1])
        return out

    def _traced(self, x):
        key = TracedRun._make_cache_key(self.module_name, (x,))
        if key not in self._warmed:
            out = self._warmup(x)
            self._warmed.add(key)
            return out
        if self._tids is None:
            self._capture()
        return self._replay(x)

    def _warmup(self, x_dev):
        self._x_dev = x_dev
        self._hop_sock = [None]
        with trace_running():
            out = self._stages[0].forward(x_dev)
            for i in range(1, len(self._stages)):
                b = self._hop_in[i]
                src = self._as_l1(self._src1(out, "traced warm-up"))
                ss, rs, buf = b.transport.prepare(src, b.mesh_b, tag=b.tag)
                b.transport.send_only(src, ss)
                b.transport.recv_only(buf, rs)
                self._hop_sock.append({"ss": ss, "rs": rs, "buf": buf})
                out = self._stages[i].forward(_rewrap_like(out, buf))
        for m in self._meshes:
            ttnn.synchronize_device(m)
        return out

    def _emit_stage(self, gi, n):
        """Record stage gi's ops ([recv] + forward + [send]) into its mesh's open trace."""
        if gi == 0:
            inp = self._x_dev
        else:
            b = self._hop_in[gi]
            b.transport.recv_only(self._hop_sock[gi]["buf"], self._hop_sock[gi]["rs"])
            inp = self._hop_sock[gi]["buf"]
        o = self._stages[gi].forward(inp)
        if gi < n - 1:
            src = self._src1(o, "traced capture")
            assert src.memory_config().buffer_type == ttnn.BufferType.L1, (
                "Pipeline TRACED: a crossing stage output must be L1-resident (a DRAM->L1 coercion "
                "would allocate inside the trace). Make the stage output L1."
            )
            self._hop_in[gi + 1].transport.send_only(src, self._hop_sock[gi + 1]["ss"])
        else:
            self._out_last = o

    def _capture(self):
        # ONE trace per DISTINCT physical mesh (merges revisits -> minimum #traces); each mesh's
        # trace holds its stages' ops in topological order, so replay is deadlock-free.
        n = len(self._stages)
        seen, mesh_order = set(), []
        for st in self._stages:
            if id(st.device) not in seen:
                seen.add(id(st.device))
                mesh_order.append(st.device)
        tids = []
        with trace_running():
            for mesh in mesh_order:
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for gi in range(n):
                    if self._stages[gi].device is mesh:
                        self._emit_stage(gi, n)
                ttnn.end_trace_capture(mesh, tid)
                tids.append((mesh, tid))
        self._tids = tids
        Pipeline._track_traces(tids)

    def _replay(self, x):
        if x is not self._x_dev:
            ttnn.copy(x, self._x_dev)
        for mesh, tid in self._tids:  # blocking=False on ALL stages (else a send deadlocks)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        if self._sync_on_return:
            for mesh, _ in self._tids:
                ttnn.synchronize_device(mesh)
        return self._out_last

    @staticmethod
    def _distinct_meshes(submeshes):
        seen, order = set(), []
        for m in submeshes:
            if id(m) not in seen:
                seen.add(id(m))
                order.append(m)
        return order

    def capture_loop(self, submeshes, body_fn, n_steps):
        meshes = self._distinct_meshes(submeshes)
        tids = []
        with trace_running():
            for m in meshes:
                tids.append((m, ttnn.begin_trace_capture(m, cq_id=0)))
            for i in range(n_steps):
                body_fn(i)
            for m, tid in tids:
                ttnn.end_trace_capture(m, tid)
        Pipeline._track_traces(tids)
        return tids

    @staticmethod
    def replay_loop(loop_tids, *, drain="all", drain_mesh=None):
        for m, tid in loop_tids:
            ttnn.execute_trace(m, tid, cq_id=0, blocking=False)
        if drain == "stage0":
            ttnn.synchronize_device(drain_mesh if drain_mesh is not None else loop_tids[0][0])
        else:
            seen = set()
            for m, _ in loop_tids:
                if id(m) not in seen:
                    seen.add(id(m))
                    ttnn.synchronize_device(m)

    @classmethod
    def release_loop(cls, loop_tids):
        for m, tid in loop_tids or ():
            try:
                ttnn.release_trace(m, tid)
            except Exception:
                pass
        cls._untrack_traces(loop_tids)

    def release_traces(self):
        if self._tids:
            for mesh, tid in self._tids:
                try:
                    ttnn.release_trace(mesh, tid)
                except Exception:
                    pass
            Pipeline._untrack_traces(self._tids)
        self._tids = None
        self._warmed = set()

    def close(self):
        for m in self._meshes:  # drain submeshes (never the parent) before teardown
            try:
                ttnn.synchronize_device(m)
            except Exception:
                pass
        self.release_traces()
        seen = set()
        for b in self._hop_in:  # close each unique bridge transport once
            if b is not None and id(b.transport) not in seen:
                seen.add(id(b.transport))
                try:
                    b.transport.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
