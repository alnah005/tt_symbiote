# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device device-to-device (d2d) socket transport for tt_symbiote.

Wraps the fabric direct-write socket ops into a cacheable transport used by D2DBridge to move
tensors between two submeshes with NO host bounce. Socket payload must be L1-resident.
"""
from __future__ import annotations

from typing import Dict, Tuple

import ttnn

# Import-time guard: the socket ops must exist in the active ttnn build.
assert hasattr(ttnn.experimental, "send_direct_async"), (
    f"d2d socket ops absent from this ttnn ({ttnn.__file__}); a tt-metal build with "
    f"send_direct_async/recv_direct_async (PR #45765) is required for D2DBridge."
)

_HANDSHAKE_PAGE_SIZE = 4096  # direct mode uses the FIFO only for handshake
_L1 = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)


def _as_l1(t):
    if t.memory_config().buffer_type != ttnn.BufferType.L1:
        return ttnn.to_memory_config(t, _L1)
    return t


def carve_two_submeshes(parent):
    """Carve a parent mesh into two submeshes (1x2 each by default; 1x1 fallback)."""
    shape = tuple(int(d) for d in parent.shape)
    if shape[0] >= 2 and shape[1] >= 2:
        a = parent.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(0, 0))
        b = parent.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(1, 0))
    else:
        a = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        b = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
    return a, b


class SocketTransport:
    """On-device tensor handoff over fabric direct-write sockets.

    ``send`` lazily caches one ``(send_socket, recv_socket)`` pair + one receiver buffer per
    ``(src_mesh, dst_mesh, tag)``, then issues send/recv. No host bounce; no synchronize_device
    (trace-compatible). For TRACED capture the send and recv must land in SEPARATE per-mesh traces,
    so :meth:`send_only` / :meth:`recv_only` expose the two halves (pair+buffer must be cached first).
    """

    def __init__(self, *, num_connections: int = 1, page_size: int = _HANDSHAKE_PAGE_SIZE, send_op=None, recv_op=None):
        self._num_connections = num_connections
        self._page_size = page_size
        self._pairs: Dict[Tuple[int, int, object], Tuple] = {}
        self._recv_bufs: Dict[Tuple[int, int, object], object] = {}
        self._send_op = send_op  # default resolved at call time
        self._recv_op = recv_op

    def _connections(self, src_mesh):
        sender = [ttnn.CoreCoord(i, 0) for i in range(self._num_connections)]
        recv = [ttnn.CoreCoord(i, 1) for i in range(self._num_connections)]
        conns = []
        for coord in ttnn.MeshCoordinateRange(src_mesh.shape):
            for s, r in zip(sender, recv):
                conns.append(ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, s), ttnn.MeshCoreCoord(coord, r)))
        return conns

    def _pair(self, src_mesh, dst_mesh, tag):
        key = (id(src_mesh), id(dst_mesh), tag)
        pair = self._pairs.get(key)
        if pair is None:
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, self._page_size * 4)  # L1 storage (op requirement)
            cfg = ttnn.SocketConfig(self._connections(src_mesh), mem)
            pair = ttnn.create_socket_pair(src_mesh, dst_mesh, cfg)
            self._pairs[key] = pair
        return pair

    def allocate_recv_buffer(self, src_template, dst_mesh, *, tag=None):
        key = (id(src_template.device()), id(dst_mesh), tag)
        buf = self._recv_bufs.get(key)
        if buf is None:
            buf = ttnn.allocate_tensor_on_device(src_template.spec, dst_mesh)  # matches spec
            self._recv_bufs[key] = buf
        return buf

    def _check_payload(self, src_tensor):
        # Direct-write reads the source from L1 (a DRAM source silently corrupts the transfer).
        assert (
            src_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), f"d2d send() payload must be L1-resident, got {src_tensor.memory_config().buffer_type}"
        # aligned page must fit the socket FIFO
        aligned = getattr(src_tensor, "buffer_aligned_page_size", None)
        if callable(aligned):
            ap = src_tensor.buffer_aligned_page_size()
            assert self._page_size * 4 >= ap, f"socket fifo {self._page_size * 4} < aligned_page {ap}"

    def prepare(self, src_tensor, dst_mesh, *, tag=None):
        """Build+cache the socket pair and receiver buffer for this hop without issuing ops."""
        self._check_payload(src_tensor)
        send_sock, recv_sock = self._pair(src_tensor.device(), dst_mesh, tag)
        out_buf = self.allocate_recv_buffer(src_tensor, dst_mesh, tag=tag)
        return send_sock, recv_sock, out_buf

    def send_only(self, src_tensor, send_sock):
        """Issue ONLY the sender-side op (records into the sender mesh's trace)."""
        (self._send_op or ttnn.experimental.send_direct_async)(src_tensor, send_sock)

    def recv_only(self, out_buf, recv_sock):
        """Issue ONLY the receiver-side op (records into the receiver mesh's trace)."""
        (self._recv_op or ttnn.experimental.recv_direct_async)(out_buf, recv_sock)

    def send(self, src_tensor, dst_mesh, *, out_buf=None, tag=None):
        """Eager d2d transfer: send_direct_async + recv_direct_async. No synchronize_device."""
        self._check_payload(src_tensor)
        send_sock, recv_sock = self._pair(src_tensor.device(), dst_mesh, tag)
        if out_buf is None:
            out_buf = self.allocate_recv_buffer(src_tensor, dst_mesh, tag=tag)
        self.send_only(src_tensor, send_sock)
        self.recv_only(out_buf, recv_sock)
        return out_buf  # NO synchronize_device here (trace-compatible)

    def close(self):
        # Drop cached MeshSocket / buffer refs BEFORE submesh teardown.
        self._pairs.clear()
        self._recv_bufs.clear()
