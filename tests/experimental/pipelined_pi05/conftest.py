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
    assert os.environ.get("MESH_DEVICE") == "P150", "pipelined denoise tests require MESH_DEVICE=P150"
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=134_217_728)


def _close_parent_mesh(parent):
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
