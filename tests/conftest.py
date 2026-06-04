# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Root pytest conftest for the tt_symbiote test suite.

Fixtures here apply to EVERY test tree because tests/ is the common ancestor of:
  - tests/auto/         (software-only; ttnn deferred)
  - tests/shared/       (shared helpers + shared capability tests)
  - tests/models/       (RICH per-model dirs, e2e-traced correct)
  - tests/experimental/ (MINIMAL per-model dirs, partial TTNN)

pcc_threshold and the non-blocking tt_metal_commit_check fixture MUST live here
(NOT in tests/shared/conftest.py): tests/shared is a *sibling* of tests/models /
tests/experimental, so fixtures defined there would be invisible to the per-model
trees. Imports only stdlib + pytest so collection is safe without ttnn / hardware /
$TT_METAL_HOME.
"""

import functools
import json
import logging
import os
import subprocess
import warnings
from pathlib import Path

import pytest

_LOGGER = logging.getLogger(__name__)
_WARNED_MESSAGES: set[str] = set()


@pytest.fixture
def pcc_threshold():
    """Default PCC threshold for model accuracy tests."""
    return 0.99


def _find_test_config(start: Path):
    for parent in [start, *start.parents]:
        cfg = parent / "test_config.json"
        if cfg.is_file():
            return cfg
    return None


@functools.lru_cache(maxsize=1)
def _current_tt_metal_commit():
    home = os.environ.get("TT_METAL_HOME")
    if not home:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", home, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _warn_once(msg: str):
    if msg not in _WARNED_MESSAGES:
        _WARNED_MESSAGES.add(msg)
        warnings.warn(msg)
        _LOGGER.warning(msg)


@pytest.fixture(autouse=True)
def tt_metal_commit_check(request):
    """NON-BLOCKING: warn if recorded tt_metal_commit != current checkout.

    No-ops when no test_config.json is found above the test (so tests/auto and
    tests/shared are unaffected). Never skips or fails. Each distinct message is
    emitted at most once per process; the git lookup is memoized (one subprocess).
    """
    cfg_path = _find_test_config(Path(str(request.node.fspath)).parent)
    if cfg_path is None:
        yield
        return
    try:
        recorded = json.loads(cfg_path.read_text()).get("tt_metal_commit", "")
    except Exception:
        yield
        return
    if not recorded:
        yield
        return
    current = _current_tt_metal_commit()
    if current is None:
        _warn_once("tt_metal_commit_check: $TT_METAL_HOME unset or git rev-parse "
                   "failed; running anyway.")
        yield
        return
    if recorded != current:
        _warn_once(f"tt_metal_commit mismatch for {cfg_path.parent.name}: recorded "
                   f"{recorded} vs current {current}; running anyway.")
    yield
