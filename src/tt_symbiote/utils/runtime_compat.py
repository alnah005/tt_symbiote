# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runtime ttnn <-> model compatibility gate.

:func:`check_ttnn_compat` compares the tt-metal commit a recipe was verified
against (``tt_symbiote.models._runtime_pins.RUNTIME_PINS``) with the commit the
*installed* ttnn was built from, and emits an advisory on mismatch (or when the
installed commit cannot be determined):

- default: a :class:`UserWarning` (soft); the runtime sibling of the test-time
  ``tt_metal_commit_check`` fixture in ``tests/conftest.py``.
- ``TT_SYMBIOTE_STRICT_TTNN=1``: raises :class:`RuntimeError` instead (CI / strict).

The gate is best-effort and defensive: every internal error is swallowed so it
can never break model loading. The *only* exception it raises is the deliberate
strict-mode :class:`RuntimeError`.

Hook sites:
  - ``tt_symbiote.models.auto.auto_factory.from_pretrained`` (model class known
    immediately after HF load).
  - ``tt_symbiote.utils.device_management.set_device`` (covers hand-constructed
    models that bypass ``from_pretrained``; de-duplicated against the first call).
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional

from tt_symbiote.models._runtime_pins import RUNTIME_PINS, TTNN_VERSION_COMMITS

__all__ = ["check_ttnn_compat", "installed_ttnn_commit", "reset_warned"]

# (hf_class_name, want_commit, have_commit) tuples already emitted, so the two
# hook sites (from_pretrained + set_device) don't double-warn for one model.
_WARNED: set = set()


def reset_warned() -> None:
    """Clear the de-dup ledger (test helper; not used in production paths)."""
    _WARNED.clear()


def installed_ttnn_commit() -> Optional[str]:
    """Best-effort tt-metal commit of the installed ttnn, or ``None`` if unknown.

    Tier 1 (future-proof): if ``ttnn`` is *already* imported and exposes
    ``__tt_metal_commit__``, return it. We never import ttnn here merely to probe
    (keeps the gate cheap and usable under the ``tests/auto`` import stubs).

    Tier 2 (today): map ``importlib.metadata.version("ttnn")`` through
    ``TTNN_VERSION_COMMITS``. Returns ``None`` for versions not in the table.
    """
    mod = sys.modules.get("ttnn")
    if mod is not None:
        commit = getattr(mod, "__tt_metal_commit__", None)
        if isinstance(commit, str) and commit:
            return commit
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("ttnn")
        except PackageNotFoundError:
            return None
    except Exception:
        return None
    return TTNN_VERSION_COMMITS.get(installed)


def _install_hint() -> str:
    return (
        " The release ships a single ttnn runtime; to reproduce the verified result, "
        "install/build the ttnn matching the recipe's tt_metal_commit "
        "(see docs/development/ttnn_pinning.md)."
    )


def _compute_advisory(hf_class_name: str) -> Optional[str]:
    """Return the advisory string for ``hf_class_name``, or ``None``.

    Records the (class, want, have) key in ``_WARNED`` when it returns a message
    so callers emit each distinct advisory at most once. Never raises.
    """
    try:
        pin = RUNTIME_PINS.get(hf_class_name)
        if not pin:
            return None
        want = pin.get("tt_metal_commit") or ""
        if not want:
            return None
        have = installed_ttnn_commit()
        if have == want:
            return None
        key = (hf_class_name, want, have)
        if key in _WARNED:
            return None
        _WARNED.add(key)
        hint = _install_hint()
        if have is None:
            return (
                f"{hf_class_name}: could not determine the tt-metal commit of the installed "
                f"ttnn; this recipe was verified against {want[:12]}. Output may be incorrect "
                f"if they differ.{hint}"
            )
        return (
            f"{hf_class_name}: installed ttnn was built from tt-metal {have[:12]} but this "
            f"recipe was verified against {want[:12]}. Output may be incorrect.{hint}"
        )
    except Exception:
        return None


def check_ttnn_compat(hf_class_name: str) -> None:
    """Warn (or, under ``TT_SYMBIOTE_STRICT_TTNN=1``, raise) on ttnn/model drift."""
    msg = _compute_advisory(hf_class_name)
    if msg is None:
        return
    if os.environ.get("TT_SYMBIOTE_STRICT_TTNN") == "1":
        raise RuntimeError(msg)
    warnings.warn(msg, stacklevel=2)
