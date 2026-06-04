# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware-bridge fixtures for the dots.ocr capability tests.

Normally ``device_params`` / ``mesh_device`` come from the ttnn pytest plugin
that ships with a tt-metal install. In this source-build environment there is
no such installed plugin -- those fixtures live only in tt-metal's root
``conftest.py``, which cannot be imported here because its top-level
``from tests.scripts.common import ...`` collides with this repo's own
``tests`` package.

This conftest therefore re-implements ``device_params`` and ``mesh_device``
by opening the mesh through ttnn's own public API, mirroring the open/close +
fabric set/reset logic of ``$TT_METAL_HOME/conftest.py::mesh_device`` exactly.
It is scoped to ``tests/models/dots_ocr/`` so it does not change fixture
behavior anywhere else in the suite.

Mesh shape comes from the ``MESH_DEVICE`` env var (e.g. ``T3K`` -> (1, 8)).
"""

import pytest

import ttnn


def _get_updated_device_params(device_params):
    """Inlined from ``tests/scripts/common.py::get_updated_device_params``.

    Pops the dispatch-core knobs and folds them into a ``DispatchCoreConfig``.
    The Blackhole-specific ROW/COL dispatch reconciliation is preserved.
    """
    new_device_params = dict(device_params)
    dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
    dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
    fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

    if ttnn.device.is_blackhole():
        fabric_config = new_device_params.get("fabric_config", None)
        if not (fabric_config and fabric_tensix_config):
            if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                dispatch_core_axis = ttnn.DispatchCoreAxis.COL

    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis, fabric_tensix_config)
    new_device_params["dispatch_core_config"] = dispatch_core_config
    return new_device_params


def _set_fabric(fabric_config, reliability_mode=None, fabric_tensix_config=None):
    if not fabric_config:
        return
    if reliability_mode is None:
        reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
    if fabric_tensix_config is None:
        fabric_tensix_config = ttnn.FabricTensixConfig.DISABLED
    ttnn.set_fabric_config(
        fabric_config,
        reliability_mode,
        None,
        fabric_tensix_config,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )


def _reset_fabric(fabric_config):
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _resolve_mesh_shape(param):
    """Dumb shape resolver: take what the test asked for and open it.

    The test owns the mesh-shape decision (e.g. dots.ocr passes ``(8, 1)`` for
    its data-parallel layout). This conftest carries no model-specific
    parallelism intelligence -- it just turns the requested shape into a
    ``ttnn.MeshShape`` and opens it.

      * ``(rows, cols)`` tuple/list -> that grid
      * ``int n``                   -> ``(1, n)``
      * ``None`` (test didn't ask)  -> ``(1, all available devices)``
    """
    if isinstance(param, (tuple, list)):
        assert len(param) == 2, "Device mesh grid shape should have exactly two elements."
        grid_dims = (int(param[0]), int(param[1]))
    elif isinstance(param, int):
        grid_dims = (1, int(param))
    else:
        grid_dims = (1, ttnn.get_num_devices())
    num_requested = grid_dims[0] * grid_dims[1]
    if not ttnn.using_distributed_env() and num_requested > ttnn.get_num_devices():
        pytest.skip("Requested more devices than available. Test not applicable for machine")
    return ttnn.MeshShape(*grid_dims)


@pytest.fixture(scope="function")
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """Open a ttnn mesh device, mirroring tt-metal's conftest mesh_device fixture."""
    param = getattr(request, "param", None)
    mesh_shape = _resolve_mesh_shape(param)

    device_params = dict(device_params)
    updated_device_params = _get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)

    _set_fabric(fabric_config, reliability_mode, fabric_tensix_config)
    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    yield mesh

    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    _reset_fabric(fabric_config)
    del mesh
