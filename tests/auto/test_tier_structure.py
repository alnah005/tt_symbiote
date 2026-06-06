# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Software-only pytest wrapper around ``scripts/check_tier_structure.py``.

Stdlib-only (no ttnn / tt_symbiote import) so it runs under ``tests/auto/``'s
software-only collection. The full audit logic (tier-dir layout, 5-key config
schema, naming/alias rules, repo invariants, CLAUDE.md semantic checks, and the
grandfather warn-not-fail policy) lives in the script; this wrapper invokes it in
``--format json --skip-lint`` mode and asserts the STRUCTURAL contract is clean.

Structural checks (must be error-free): ``tier_structure``, ``config``,
``naming``, ``repo_invariants``, and ``claude_md_checks``. The first four encode
the relocated old-lint checks and the tier migration. ``claude_md_checks`` is now
also guarded: the forward-discipline ERROR categories (``MISSING_RUN_ON_DEVICES``,
``TORCH_IN_FORWARD``, ``MISSING_LICENSE_HEADER``) were promoted from WARN to ERROR
in ``scripts/check_tier_structure.py``, so this wrapper fails on them too
(``EMPTY_TIER_DIR`` is already covered via ``tier_structure``). The CANONICAL DONE
gate remains the DIRECT full-script run (0 errors / 0 warnings); this wrapper is
the secondary software guard. ``linting`` is intentionally NOT in the structural
set here (the wrapper runs ``--skip-lint`` and yamllint may be absent in CI). All
warnings are emitted via ``warnings.warn``.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "check_tier_structure.py"

# the structural check groups whose error-freeness this gate enforces.
# claude_md_checks is included now that MISSING_RUN_ON_DEVICES / TORCH_IN_FORWARD /
# MISSING_LICENSE_HEADER are ERRORs (EMPTY_TIER_DIR is covered via tier_structure).
_STRUCTURAL = ("tier_structure", "config", "naming", "repo_invariants", "claude_md_checks")


def _run():
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--format", "json", "--skip-lint"],
        capture_output=True,
        text=True,
    )
    return json.loads(r.stdout), r


def test_tier_structure_clean():
    report, r = _run()
    assert report["schema_version"] == 1
    for grp in report["checks"].values():
        for w in grp.get("warnings", []):
            warnings.warn(str(w))
    # surface (non-structural) semantic errors as warnings -- pre-existing debt.
    for name, grp in report["checks"].items():
        if name in _STRUCTURAL:
            continue
        for f in grp.get("failures", []):
            warnings.warn("PRE-EXISTING (out of structure-gate scope): " + str(f))
    struct_fails = [f for name in _STRUCTURAL for f in report["checks"][name].get("failures", [])]
    assert not struct_fails, (
        "structural (tier/config/naming/repo) violations:\n" + "\n".join(str(f) for f in struct_fails) + "\n" + r.stderr
    )


def test_tier_structure_selftest():
    r = subprocess.run([sys.executable, str(_SCRIPT), "--selftest"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
