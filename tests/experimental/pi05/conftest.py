# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures/helpers for pi0.5 capability tests.

The ``device`` fixture comes from the ttnn pytest plugin (single Blackhole /
P150). This conftest adds the pi0_5 PyTorch reference tree to ``sys.path`` so
``models.experimental.pi0_5.reference.*`` (the PCC golden) is importable, and
provides random-weight builders so component (Tier 1/2/3) PCC tests can run on
hardware WITHOUT the gated ``pi05_base`` checkpoint. Pretrained-weight tests
skip automatically when the checkpoint is absent.
"""

from __future__ import annotations

import os

import pytest

import ttnn

# Importing the helpers inserts the pi0_5 reference tree into sys.path as a
# side effect (so the PCC golden is importable during collection).
from .pi05_helpers import REFERENCE_ROOT, require_reference


@pytest.fixture(scope="session")
def reference_root() -> str:
    require_reference()
    return REFERENCE_ROOT


@pytest.fixture(autouse=True)
def _tracy_signpost(request):
    """Emit a tracy signpost per test so a tracy run over the test file segments
    the device profile per test. Gated on PI05_TRACY_SIGNPOST=1 so normal/CI runs
    are unaffected (signpost is a no-op marker otherwise)."""
    if os.environ.get("PI05_TRACY_SIGNPOST"):
        try:
            from tracy import signpost

            signpost(header="T_" + request.node.name.replace("[", "_").replace("]", ""))
        except Exception:
            pass
    yield


# Per-test trace release is handled centrally by the ROOT conftest's autouse
# ``release_ttnn_traces`` fixture (applies to all model trees). The dev fixture below also
# releases before close_device as a device-close safeguard.


@pytest.fixture(scope="module")
def dev():
    """Open a single Blackhole (P150) device directly.

    The ttnn pytest-plugin ``device``/``mesh_device`` fixtures are not loaded in
    this repo's pytest setup, so single-device pi0.5 tests open device 0
    directly (same pattern as ``tests/capabilities/dots_ocr``). ``trace_region``
    is sized for the traced-execution stage; ``l1_small_size`` matches the
    reference pi0_5 tests.
    """
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=134_217_728)
    try:
        yield device
    finally:
        # Release any framework-captured traces before closing: ttnn.close_device
        # TT_FATALs ("!trace_id_.has_value()") if a captured trace is still live.
        try:
            from tt_symbiote.core.run_config import TracedRun

            TracedRun.release_all()
        except Exception:
            pass
        ttnn.close_device(device)
