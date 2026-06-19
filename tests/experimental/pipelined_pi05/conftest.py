# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest

import ttnn

from .pi05_helpers import REFERENCE_ROOT, require_reference


@pytest.fixture(scope="session")
def reference_root() -> str:
    require_reference()
    return REFERENCE_ROOT


@pytest.fixture(autouse=True)
def _tracy_signpost(request):
    if os.environ.get("PI05_TRACY_SIGNPOST"):
        try:
            from tracy import signpost

            signpost(header="T_" + request.node.name.replace("[", "_").replace("]", ""))
        except Exception:
            pass
    yield


def _open_parent_mesh():
    # MESH_DEVICE=P150 is the per-op arch: the pipeline carves the parent into four 1x1
    # submeshes and each stage runs on a single P150, matching the @run_on_devices(P150)
    # guards. The 4-device (P150x4) requirement is a hardware-count check below -- enforcing
    # it via MESH_DEVICE=P150x4 would resolve every op's arch to P150x4 and fall them all
    # back to torch.
    assert os.environ.get("MESH_DEVICE") == "P150", "pipelined denoise tests require MESH_DEVICE=P150"
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=134_217_728)
    n = parent.get_num_devices()
    if n != 4:
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise AssertionError(f"pipelined denoise requires a 4-device P150x4 system; found {n} device(s)")
    return parent


def _close_parent_mesh(parent):
    # Release ALL tracked traces + socket transports BEFORE closing the mesh (release_trace /
    # transport.close need a live device). TracedRun.release_all() drains the module-trace cache
    # (+ clears warm-up bookkeeping, else replay corrupts across back-to-back traced tests);
    # Pipeline.release_all() drains the pipeline loop/forward traces and the hop+wrap sockets.
    # Both are catch-alls even if a test skipped its own drv.close().
    for _release_all in ("tt_symbiote.core.run_config:TracedRun", "tt_symbiote.core.d2d_pipeline:Pipeline"):
        mod, cls = _release_all.split(":")
        try:
            import importlib

            getattr(importlib.import_module(mod), cls).release_all()
        except Exception:
            pass
    for submesh in parent.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="function")
def denoise_parent_mesh():
    parent = _open_parent_mesh()
    try:
        yield parent
    finally:
        _close_parent_mesh(parent)


@pytest.fixture(scope="module")
def dev():
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=134_217_728)
    try:
        yield device
    finally:
        try:
            from tt_symbiote.core.run_config import TracedRun

            TracedRun.release_all()
        except Exception:
            pass
        ttnn.close_device(device)
