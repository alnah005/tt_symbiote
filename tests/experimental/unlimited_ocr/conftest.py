# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Local device fixtures for the unlimited_ocr experimental tests.

The canonical ``device`` / ``mesh_device`` / ``device_params`` fixtures live in
tt-metal's ROOT ``conftest.py`` (``$TT_METAL_HOME/conftest.py``) -- they are NOT a
pip-installed pytest plugin, so they are unavailable when pytest is invoked from
the tt_symbiote repo. This conftest provides API-compatible equivalents (same
names, same ``device_params`` indirection, same function scope) so every tier test
can simply request ``device`` and be parametrized with
``@pytest.mark.parametrize("device_params", [{...}], indirect=True)`` -- mirroring
``tests/shared/test_conv.py`` and the tt-metal fixtures.

FUNCTION scope is deliberate: a fresh device per test avoids the program-cache / L1
accumulation that hangs the suite when many distinct program configs (e.g. the 5
SAM/CLIP conv variants) run on one long-lived device. ``ttnn`` is imported lazily
inside the fixture bodies so plain collection stays safe without hardware.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-dp", action="store_true", default=False,
        help="run the opt-in data-parallel (4,1)-mesh tests (marked @pytest.mark.dp)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "dp: opt-in data-parallel (4,1)-mesh test; run with --run-dp in its own "
        "pytest invocation (a (4,1) then (1,1) mesh in one process corrupts mesh state)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-dp"):
        return
    skip_dp = pytest.mark.skip(reason="data-parallel test: pass --run-dp to run")
    for item in items:
        if "dp" in item.keywords:
            item.add_marker(skip_dp)


@pytest.fixture(scope="function")
def device_params(request):
    """Indirect-parametrizable device kwargs (e.g. {"l1_small_size": 32768})."""
    return getattr(request, "param", {})


@pytest.fixture(scope="function")
def device(device_params):
    """Single Blackhole device as a (1,1) mesh (arch P150), honoring device_params.

    Returns a 1x1 MeshDevice -- the exact device type every unlimited_ocr module was
    validated against -- while presenting the ``device`` name/lifecycle the tests use.
    """
    import ttnn

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **device_params)
    try:
        yield dev
    finally:
        ttnn.close_mesh_device(dev)


@pytest.fixture(scope="function")
def mesh_device(device_params):
    """Multi-device mesh honoring device_params. Shape via device_params["mesh_shape"]
    (default (1,1)); the 4x Blackhole host supports up to (2,2) = P150x4."""
    import ttnn

    params = dict(device_params)
    shape = params.pop("mesh_shape", (1, 1))
    dev = ttnn.open_mesh_device(ttnn.MeshShape(*shape), **params)
    try:
        yield dev
    finally:
        ttnn.close_mesh_device(dev)
