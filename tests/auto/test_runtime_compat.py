# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Software-only tests for the runtime ttnn/model compatibility gate.

Covers :mod:`tt_symbiote.utils.runtime_compat` (soft warn / strict raise /
de-dup / commit resolution tiers) and the single-source-of-truth invariant
between ``RUNTIME_PINS`` and the per-model ``test_config.json``. No hardware,
no real ttnn (``tests/auto/conftest.py`` installs sys.modules stubs).
"""

import json
import sys
import warnings
from pathlib import Path

import pytest

from tt_symbiote.models import _runtime_pins
from tt_symbiote.utils import runtime_compat

_DOTS = "DotsOCRForCausalLM"
_DOTS_COMMIT = _runtime_pins.RUNTIME_PINS[_DOTS]["tt_metal_commit"]
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_warned():
    runtime_compat.reset_warned()
    yield
    runtime_compat.reset_warned()


def test_unknown_class_is_noop(recwarn, monkeypatch):
    monkeypatch.delenv("TT_SYMBIOTE_STRICT_TTNN", raising=False)
    runtime_compat.check_ttnn_compat("NotARegisteredModel")
    assert len(recwarn) == 0


def test_matching_commit_no_warning(recwarn, monkeypatch):
    monkeypatch.delenv("TT_SYMBIOTE_STRICT_TTNN", raising=False)
    monkeypatch.setattr(runtime_compat, "installed_ttnn_commit", lambda: _DOTS_COMMIT)
    runtime_compat.check_ttnn_compat(_DOTS)
    assert len(recwarn) == 0


def test_mismatched_commit_warns(monkeypatch):
    monkeypatch.delenv("TT_SYMBIOTE_STRICT_TTNN", raising=False)
    monkeypatch.setattr(runtime_compat, "installed_ttnn_commit", lambda: "0" * 40)
    with pytest.warns(UserWarning, match="Output may be incorrect"):
        runtime_compat.check_ttnn_compat(_DOTS)


def test_unknown_installed_commit_warns(monkeypatch):
    monkeypatch.delenv("TT_SYMBIOTE_STRICT_TTNN", raising=False)
    monkeypatch.setattr(runtime_compat, "installed_ttnn_commit", lambda: None)
    with pytest.warns(UserWarning, match="could not determine"):
        runtime_compat.check_ttnn_compat(_DOTS)


def test_dedup_emits_once(monkeypatch):
    monkeypatch.delenv("TT_SYMBIOTE_STRICT_TTNN", raising=False)
    monkeypatch.setattr(runtime_compat, "installed_ttnn_commit", lambda: "0" * 40)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        runtime_compat.check_ttnn_compat(_DOTS)
        runtime_compat.check_ttnn_compat(_DOTS)  # second call (e.g. set_device) is silent
    assert sum(issubclass(w.category, UserWarning) for w in rec) == 1


def test_strict_mode_raises(monkeypatch):
    monkeypatch.setenv("TT_SYMBIOTE_STRICT_TTNN", "1")
    monkeypatch.setattr(runtime_compat, "installed_ttnn_commit", lambda: "0" * 40)
    with pytest.raises(RuntimeError, match="Output may be incorrect"):
        runtime_compat.check_ttnn_compat(_DOTS)


def test_installed_commit_tier1_reads_dunder(monkeypatch):
    """If ttnn is imported and exposes __tt_metal_commit__, use it directly."""
    monkeypatch.setattr(sys.modules["ttnn"], "__tt_metal_commit__", "cafef00d" * 5, raising=False)
    assert runtime_compat.installed_ttnn_commit() == "cafef00d" * 5


def test_installed_commit_tier2_version_table(monkeypatch):
    """With no dunder, fall back to the version -> commit table."""
    # Ensure tier 1 misses: the stub's __getattr__ returns a non-str sentinel.
    if hasattr(sys.modules["ttnn"], "__tt_metal_commit__"):
        monkeypatch.delattr(sys.modules["ttnn"], "__tt_metal_commit__", raising=False)
    from importlib.metadata import version

    installed = version("ttnn")  # real dist version in the dev venv
    monkeypatch.setattr(runtime_compat, "TTNN_VERSION_COMMITS", {installed: "feedface" * 5})
    assert runtime_compat.installed_ttnn_commit() == "feedface" * 5


def test_registry_matches_dots_ocr_test_config():
    """Single source of truth: test_config.json must echo the registry commit."""
    cfg = json.loads((_REPO_ROOT / "tests/experimental/dots_ocr/test_config.json").read_text())
    assert cfg["tt_metal_commit"] == _DOTS_COMMIT


# --- scalable-schema invariants (100+ models) ------------------------------- #


def test_every_pin_has_its_own_commit():
    for hf_class, pin in _runtime_pins.RUNTIME_PINS.items():
        assert pin.get("tt_metal_commit"), f"{hf_class} missing tt_metal_commit"


def test_model_extras_reference_known_capabilities():
    for hf_class, pin in _runtime_pins.RUNTIME_PINS.items():
        for group in pin.get("extras", []):
            assert group in _runtime_pins.CAPABILITY_EXTRAS, f"{hf_class}: unknown extra {group!r}"


def test_serving_tier_is_valid_when_present():
    for hf_class, pin in _runtime_pins.RUNTIME_PINS.items():
        if "serving_tier" in pin:
            assert (
                pin["serving_tier"] in _runtime_pins.SERVING_TIERS
            ), f"{hf_class}: unknown serving_tier {pin['serving_tier']!r}"
    # The default must itself be a valid tier.
    assert _runtime_pins.serving_tier_for("__missing__") in _runtime_pins.SERVING_TIERS


def test_all_extra_is_union_of_capabilities():
    expected: set = set()
    for pkgs in _runtime_pins.CAPABILITY_EXTRAS.values():
        expected.update(pkgs)
    assert set(_runtime_pins.all_extra_packages()) == expected


def test_pyproject_extras_in_sync_with_registry():
    """The generated pyproject blocks must match _runtime_pins.py (no drift)."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "scripts/sync_ttnn_extras.py", "--check"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
