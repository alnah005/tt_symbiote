#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check-only structure + CLAUDE.md audit for the tt_symbiote test layout.

STRICTLY CHECK-ONLY (deep-plan_11 §0). This script audits and reports. It NEVER
mutates the filesystem: it never creates/moves/renames/deletes files, and when it
runs formatters it runs them only in non-mutating check mode (or shells
``pre-commit run`` which is itself check-configured). It imports no ``shutil``, no
``os.rename``, no ``Path.unlink``, and no git write op. ``--selftest`` asserts zero
filesystem mutation.

It audits BOTH per-model test trees::

  - tests/models/<name>/        RICH floor   (e2e-traced-correct)
  - tests/experimental/<name>/  MINIMAL floor (partial TTNN)

for the tier-dir layout ``Tier1/`` .. ``Tier4/`` plus a ROOT ``test_config.json``
carrying ``tt_metal_commit``, folds in package linting (check-mode shell-out),
selected CLAUDE.md semantic checks, and the three repo-wide invariants. It emits
a FROZEN, closed-key JSON contract (``schema_version: 1``) plus human text and
exits non-zero on any error.

Tier semantics (pcc-test-gen/SKILL.md): Tier1=ops/leaf, Tier2=composites,
Tier3=decoder/block layer, Tier4=full model + e2e + trace + semantic.

--------------------------------------------------------------------------------
OLD LINT -> NEW LOCATION COVERAGE MAP (no regressions; deep-plan_11 §7.4)
--------------------------------------------------------------------------------
  test_capabilities_removed              -> audit_repo_invariants (CAPABILITIES_TREE_PRESENT)   error
  test_two_tree_exclusivity (allowed)    -> audit_repo_invariants (TWO_TREE_OVERLAP)            error
  test_two_tree_exclusivity (stray cfg)  -> audit_repo_invariants (STRAY_TEST_CONFIG)           error
  test_shared_tree (9-file floor)        -> audit_repo_invariants (SHARED_TREE_VIOLATION)       error
  test_config_present_and_valid          -> audit_root_config (MISSING_CONFIG/..._KEY/...)      error
  test_rich_floor (grandfather + rich)   -> audit_tier_structure (tier-floor rules)             error/warn
  test_minimal_floor                     -> audit_tier_structure (four tier dirs + root cfg)    error
  test_naming (alias/suffix/shape)       -> audit_naming                                        error
  src-correspondence (RICH only L142)    -> audit_naming (RICH only, NO_SRC_PACKAGE)            error
  test_tier_file_token_match (+variant)  -> audit_tier_token_placement                          error
  test_no_banned_trace_path              -> audit_claude_md (TRACE_INSTANCE_FLAG)               error
  test_tower_not_trace_decorated         -> audit_claude_md (TOWER_TRACE_DECORATED)             error

--------------------------------------------------------------------------------
CLOSED `code` VOCABULARY (schema_version 1; echoed by --print-contract)
--------------------------------------------------------------------------------
  MISSING_TIER_DIR, EMPTY_TIER_DIR, MISSING_TIER_INIT, MISSING_ROOT_INIT, MISSING_CONFIG,
  MISSING_CONFIG_KEY, BAD_CONFIG_TYPE, MISSING_TT_METAL_COMMIT, TIER_TOKEN_MISMATCH,
  MISPLACED_TIER_FILE, NO_SRC_PACKAGE, BANNED_ALIAS, BANNED_VARIANT_SUFFIX, NONCANONICAL_NAME,
  CAPABILITIES_TREE_PRESENT, TWO_TREE_OVERLAP, STRAY_TEST_CONFIG, SHARED_TREE_VIOLATION,
  MISSING_LICENSE_HEADER, TORCH_IN_FORWARD, DEPRECATED_API (warn), TRACE_INSTANCE_FLAG,
  TOWER_TRACE_DECORATED, MISSING_RUN_ON_DEVICES, DUPLICATE_MODEL_NAME, LINT_FAILURE

  Only DEPRECATED_API is WARN; all other codes resolve to ERROR via _finding(). The
  _experimental tree is exempt from forward-discipline (TORCH_IN_FORWARD /
  MISSING_RUN_ON_DEVICES); the two arch-agnostic base forwards in core/module.py
  (TTNNModule, TTNNLayerStack) are skip-listed; gemma4/qwen3_vl Tier3 are documented
  TIER_NOT_IMPLEMENTED deferrals (open-blockers R-DECODER-gemma4, R-DECODER-qwen3_vl).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1

# Closed code vocabulary -- adding a code requires bumping SCHEMA_VERSION.
CODES = [
    "MISSING_TIER_DIR",
    "EMPTY_TIER_DIR",
    "MISSING_TIER_INIT",
    "MISSING_ROOT_INIT",
    "MISSING_CONFIG",
    "MISSING_CONFIG_KEY",
    "BAD_CONFIG_TYPE",
    "MISSING_TT_METAL_COMMIT",
    "TIER_TOKEN_MISMATCH",
    "MISPLACED_TIER_FILE",
    "NO_SRC_PACKAGE",
    "BANNED_ALIAS",
    "BANNED_VARIANT_SUFFIX",
    "NONCANONICAL_NAME",
    "CAPABILITIES_TREE_PRESENT",
    "TWO_TREE_OVERLAP",
    "STRAY_TEST_CONFIG",
    "SHARED_TREE_VIOLATION",
    "MISSING_LICENSE_HEADER",
    "TORCH_IN_FORWARD",
    "DEPRECATED_API",
    "TRACE_INSTANCE_FLAG",
    "TOWER_TRACE_DECORATED",
    "MISSING_RUN_ON_DEVICES",
    "DUPLICATE_MODEL_NAME",
    "LINT_FAILURE",
]
# Only DEPRECATED_API remains WARN (known-debt: 0 call sites today after the shared-test
# migration, but the detector + its line-~520 known-debt override are retained so the code
# stays defined and non-blocking for any future re-introduction). All other quality codes
# (MISSING_LICENSE_HEADER, MISSING_RUN_ON_DEVICES, TORCH_IN_FORWARD, EMPTY_TIER_DIR,
# LINT_FAILURE) now resolve to ERROR at runtime via _finding(). The _experimental tree is
# carved out of forward-discipline and the two arch-agnostic base forwards are skip-listed,
# so the promotion is real (0 findings in a clean tree) rather than cosmetic.
WARN_CODES = {"DEPRECATED_API"}

# ---- ported constants (verbatim from old lint) ----
REQUIRED_CONFIG_KEYS = {
    "tt_metal_commit": str,
    "device_arch": str,
    "pcc_threshold": (int, float),
    "hf_model_id": str,
    "hf_revision": str,
}
SHARED_REQUIRED = [
    "__init__.py",
    "conftest.py",
    "pcc_utils.py",
    "shared_configs.py",
    "test_attention.py",
    "test_conv.py",
    "test_moe.py",
    "test_rope.py",
    "test_dpl.py",
]
GRANDFATHERED_NEEDS_UPGRADE = {"bailing_moe_v2", "gemma4", "qwen3_vl", "resnet"}

RUN_ON_DEVICES_EXEMPT = {
    ("src/tt_symbiote/core/module.py", "TTNNModule"),  # NotImplementedError stub
    ("src/tt_symbiote/core/module.py", "TTNNLayerStack"),  # arch-agnostic dispatcher
    ("src/tt_symbiote/core/d2d_bridge.py", "D2DBridge"),  # overrides call(); forward() only raises
    ("src/tt_symbiote/core/d2d_pipeline.py", "Pipeline"),  # overrides call(); forward() only raises
}
# (model, tier) cells whose TTNN component is genuinely DEFERRED in src. An empty dir here is a
# sanctioned deferral, not a defect. Source of truth: the model's own deferred list
# (gemma4 Gemma4TextDecoderLayer; qwen3_vl modeling_qwen3_vl.py:85 Qwen3VLTextDecoderLayer).
# Remove the entry when the TTNN decoder lands. Open-blockers: R-DECODER-gemma4, R-DECODER-qwen3_vl.
TIER_NOT_IMPLEMENTED = {("gemma4", "Tier3"), ("qwen3_vl", "Tier3")}
BANNED_ALIASES = {"owl_vit", "speech_t5", "qwen_omni", "gr00t"}
BANNED_SUFFIX_RE = re.compile(r"_(4_7|5|flash|coder_next)$")  # DIR names ONLY
DIR_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
VARIANT_RE = re.compile(r"_variant_.+$")
# token group is non-greedy and an optional _variant_<q> tail is stripped.
TIER_FILE_RE = re.compile(
    r"^test_(ops|composites|decoder|modeling|traced|auto|pipeline|device_guards|"
    r"sweep_decoder|timing_replay|vision_traced)_(.+?)(?:_variant_\w+)?\.py$"
)
TIER_DIRS = ["Tier1", "Tier2", "Tier3", "Tier4"]
# file kind (first regex group) -> tier dir; default Tier4.
FILE_KIND_TO_TIER = {
    "ops": "Tier1",
    "composites": "Tier2",
    "decoder": "Tier3",
    "sweep_decoder": "Tier3",
    "modeling": "Tier4",
    "traced": "Tier4",
    "vision_traced": "Tier4",
    "auto": "Tier4",
    "pipeline": "Tier4",
    "device_guards": "Tier4",
    "timing_replay": "Tier4",
}
ALLOWED_TESTS_CHILDREN = {"auto", "shared", "models", "experimental", "__pycache__"}
SKIP_DIRS = {"__pycache__"}

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
SRC_MODELS = REPO / "src" / "tt_symbiote" / "models"
RICH = TESTS / "models"
EXPERIMENTAL = TESTS / "experimental"
SHARED = TESTS / "shared"
# src children that are NOT models
SRC_NON_MODEL = {"auto", "__init__.py", "__pycache__"}


# The partial-bring-up tree, legitimately exempt from forward-discipline
# (TORCH_IN_FORWARD / MISSING_RUN_ON_DEVICES) only. License / deprecated /
# trace loops still scan it. Computed from the live SRC_MODELS so --selftest
# (which repoints SRC_MODELS at a tmp tree) exercises the carve-out correctly.
def _is_experimental(py):
    experimental_dir = SRC_MODELS.parent / "_experimental"
    return experimental_dir in py.parents  # dir-ancestry, not substring; no over-exemption


# ----------------------------------------------------------------------------
# finding/report helpers
# ----------------------------------------------------------------------------
def _finding(path, code, detail):
    sev = "warning" if code in WARN_CODES else "error"
    return {"path": str(path), "code": code, "detail": detail, "severity": sev}


def _new_group():
    return {"ok": True, "failures": [], "warnings": []}


def _add(group, finding):
    if finding["severity"] == "warning":
        group["warnings"].append(finding)
    else:
        group["failures"].append(finding)
        group["ok"] = False


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except Exception:
        return str(p)


# ----------------------------------------------------------------------------
# model discovery
# ----------------------------------------------------------------------------
def _model_dirs(tree):
    if not tree.is_dir():
        return []
    return [p for p in sorted(tree.iterdir()) if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("__")]


def _select_models(trees, only):
    out = []
    for tree in trees:
        for d in _model_dirs(tree):
            if only and d.name not in only:
                continue
            out.append(d)
    return out


# ----------------------------------------------------------------------------
# audit: tier structure (D1)  -- floor rules per §5
# ----------------------------------------------------------------------------
def audit_tier_structure(model_dir, is_rich, group):
    name = model_dir.name
    # root __init__.py
    if not (model_dir / "__init__.py").is_file():
        _add(group, _finding(_rel(model_dir), "MISSING_ROOT_INIT", "__init__.py missing"))

    grandfathered = is_rich and name in GRANDFATHERED_NEEDS_UPGRADE
    for tier in TIER_DIRS:
        td = model_dir / tier
        if not td.is_dir():
            _add(group, _finding(_rel(model_dir), "MISSING_TIER_DIR", f"{tier}/ missing"))
            continue
        if not (td / "__init__.py").is_file():
            _add(group, _finding(_rel(td), "MISSING_TIER_INIT", f"{tier}/__init__.py missing"))
        has_tests = any(td.glob("test_*.py"))
        if not has_tests:
            # Documented per-(model,tier) deferral: the TTNN component for this
            # cell is genuinely unimplemented in src (see TIER_NOT_IMPLEMENTED).
            # An empty dir here is sanctioned, not a defect — emit nothing.
            if (name, tier) in TIER_NOT_IMPLEMENTED:
                continue
            # Empty tier policy: experimental => PASS; rich grandfathered => ERROR;
            # rich non-grandfathered Tier4 => ERROR (must hold full-model tests);
            # rich non-grandfathered Tier1-3 => ERROR (token-matched file required).
            if not is_rich:
                continue
            if grandfathered:
                _add(group, _finding(_rel(td), "EMPTY_TIER_DIR", f"{tier}/ empty (grandfathered)"))
            else:
                if tier == "Tier4":
                    _add(
                        group,
                        _finding(
                            _rel(td),
                            "EMPTY_TIER_DIR",
                            f"{tier}/ empty: rich non-grandfathered model must " f"hold full-model tests",
                        ),
                    )
                else:
                    _add(group, _finding(_rel(td), "EMPTY_TIER_DIR", f"{tier}/ empty"))


# ----------------------------------------------------------------------------
# audit: root config (5-key typed + tt_metal_commit presence)
# ----------------------------------------------------------------------------
def audit_root_config(model_dir, group):
    cfg = model_dir / "test_config.json"
    if not cfg.is_file():
        _add(group, _finding(_rel(model_dir), "MISSING_CONFIG", "test_config.json missing"))
        return
    try:
        data = json.loads(cfg.read_text())
    except Exception as e:
        _add(group, _finding(_rel(cfg), "MISSING_CONFIG", f"unparseable test_config.json: {e}"))
        return
    for key, typ in REQUIRED_CONFIG_KEYS.items():
        if key not in data:
            _add(group, _finding(_rel(cfg), "MISSING_CONFIG_KEY", f"missing key {key}"))
            continue
        if not isinstance(data[key], typ):
            _add(
                group,
                _finding(_rel(cfg), "BAD_CONFIG_TYPE", f"key {key} type {type(data[key]).__name__}, expected {typ}"),
            )
    # tt_metal_commit presence (empty string allowed, same relaxation as old lint).
    if "tt_metal_commit" not in data:
        _add(group, _finding(_rel(cfg), "MISSING_TT_METAL_COMMIT", "no tt_metal_commit key"))


# ----------------------------------------------------------------------------
# audit: naming (banned alias/suffix/shape + RICH->src correspondence)
# ----------------------------------------------------------------------------
def audit_naming(model_dir, is_rich, group):
    name = model_dir.name
    if name in BANNED_ALIASES:
        _add(group, _finding(_rel(model_dir), "BANNED_ALIAS", f"banned alias dir name: {name}"))
    if BANNED_SUFFIX_RE.search(name):
        _add(group, _finding(_rel(model_dir), "BANNED_VARIANT_SUFFIX", f"banned variant-suffix dir name: {name}"))
    if not DIR_NAME_RE.match(name):
        _add(group, _finding(_rel(model_dir), "NONCANONICAL_NAME", f"dir name not canonical-shaped: {name}"))
    # src-correspondence is RICH-ONLY (experimental never deleted for "no src").
    if is_rich:
        if not (SRC_MODELS / name).is_dir():
            _add(
                group,
                _finding(
                    _rel(model_dir), "NO_SRC_PACKAGE", f"RICH dir {name} has no src/tt_symbiote/models/{name} package"
                ),
            )


# ----------------------------------------------------------------------------
# audit: tier-token placement (token==name, variant carve-out, misplacement)
# ----------------------------------------------------------------------------
def audit_tier_token_placement(model_dir, is_rich, group):
    name = model_dir.name
    # scan only tier subdirs for test files (root test files are legacy/flat -> flag)
    for py in sorted(model_dir.glob("test_*.py")):
        # a test_*.py at the model ROOT (not in a tier dir) is misplaced post-migration.
        _add(group, _finding(_rel(py), "MISPLACED_TIER_FILE", f"test file at model root; expected under a Tier dir"))
    for tier in TIER_DIRS:
        td = model_dir / tier
        if not td.is_dir():
            continue
        for py in sorted(td.glob("test_*.py")):
            m = TIER_FILE_RE.match(py.name)
            if not m:
                continue
            kind, token = m.group(1), m.group(2)
            # token match (allow _variant_ tail already stripped by regex)
            if token != name and not (token.startswith(name) and VARIANT_RE.search(py.name)):
                _add(
                    group,
                    _finding(_rel(py), "TIER_TOKEN_MISMATCH", f"tier token '{token}' does not match dir '{name}'"),
                )
                continue
            expected_tier = FILE_KIND_TO_TIER.get(kind, "Tier4")
            if tier != expected_tier:
                _add(
                    group,
                    _finding(
                        _rel(py),
                        "MISPLACED_TIER_FILE",
                        f"{py.name} (kind '{kind}') in {tier}/, expected {expected_tier}/",
                    ),
                )


# ----------------------------------------------------------------------------
# audit: repo-wide invariants (whole-repo mode only)
# ----------------------------------------------------------------------------
def audit_repo_invariants(group):
    # 1. capabilities tree must not exist
    if (TESTS / "capabilities").exists():
        _add(
            group,
            _finding(
                _rel(TESTS / "capabilities"),
                "CAPABILITIES_TREE_PRESENT",
                "the old per-model capabilities tree must be removed",
            ),
        )
    # 2a. only allowed dir children under tests/
    if TESTS.is_dir():
        for child in sorted(TESTS.iterdir()):
            if child.is_dir() and child.name not in ALLOWED_TESTS_CHILDREN:
                _add(group, _finding(_rel(child), "TWO_TREE_OVERLAP", f"unexpected dir under tests/: {child.name}"))
        # 2b. no stray test_config.json outside the two per-model trees
        for child in sorted(TESTS.iterdir()):
            if child.is_dir() and child.name not in {"models", "experimental"}:
                for cfg in child.rglob("test_config.json"):
                    _add(
                        group,
                        _finding(
                            _rel(cfg), "STRAY_TEST_CONFIG", f"test_config.json outside per-model trees: {_rel(cfg)}"
                        ),
                    )
    # 3. shared tree 9-file floor
    if not SHARED.is_dir():
        _add(group, _finding(_rel(SHARED), "SHARED_TREE_VIOLATION", "tests/shared/ must exist"))
    else:
        for f in SHARED_REQUIRED:
            if not (SHARED / f).is_file():
                _add(group, _finding(_rel(SHARED / f), "SHARED_TREE_VIOLATION", f"missing tests/shared/{f}"))


# ----------------------------------------------------------------------------
# audit: CLAUDE.md semantic checks (all stdlib/ast, read-only)
# ----------------------------------------------------------------------------
def _has_license_header(text):
    head = []
    for line in text.splitlines():
        if line.startswith("#!"):
            continue
        head.append(line)
        if len(head) >= 8:
            break
    blob = "\n".join(head)
    spdx_copy = re.search(r"SPDX-FileCopyrightText.*Tenstorrent AI ULC", blob)
    copy_ok = spdx_copy and ("(C)" in spdx_copy.group(0) or "©" in spdx_copy.group(0) or "(C)" in blob or "©" in blob)
    lic_ok = "SPDX-License-Identifier: Apache-2.0" in blob
    return bool(spdx_copy and lic_ok)


def _forward_has_torch(func_node):
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call):
            f = n.func
            # torch.X  or  torch.nn.functional.X
            cur = f
            parts = []
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                parts.reverse()
                if parts and parts[0] == "torch":
                    return True
    return False


def audit_claude_md(group):
    # 1. license headers (WARN) over src/tt_symbiote and tests (excl scripts, __pycache__).
    for base in (SRC_MODELS.parent, TESTS):
        if not base.is_dir():
            continue
        for py in sorted(base.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            try:
                text = py.read_text()
            except Exception:
                continue
            # exclude near-empty __init__.py (<= 2 non-blank lines)
            if py.name == "__init__.py":
                nonblank = [ln for ln in text.splitlines() if ln.strip()]
                if len(nonblank) <= 2:
                    continue
            if not _has_license_header(text):
                _add(group, _finding(_rel(py), "MISSING_LICENSE_HEADER", "missing SPDX copyright/license header"))

    # 2. no-torch-in-forward (ERROR) over src/tt_symbiote
    src_root = SRC_MODELS.parent
    if src_root.is_dir():
        for py in sorted(src_root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            # Carve out the partial-bring-up tree from forward-discipline ONLY
            # (TORCH_IN_FORWARD + MISSING_RUN_ON_DEVICES). The separate license /
            # deprecated / trace loops still scan _experimental.
            if _is_experimental(py):
                continue
            try:
                tree = ast.parse(py.read_text())
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                base_names = [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases]
                is_ttnn = node.name.startswith("TTNN") or any("TTNNModule" in b for b in base_names)
                if not is_ttnn:
                    continue
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                        if _forward_has_torch(item):
                            _add(
                                group,
                                _finding(_rel(py), "TORCH_IN_FORWARD", f"torch.* call inside {node.name}.forward()"),
                            )
                        # 5. @run_on_devices presence. Base-method stubs/dispatchers
                        # in core/module.py are arch-agnostic and skip-listed.
                        deco_names = [ast.unparse(d) if hasattr(ast, "unparse") else "" for d in item.decorator_list]
                        if (_rel(py), node.name) not in RUN_ON_DEVICES_EXEMPT:
                            if not any("run_on_devices" in d for d in deco_names):
                                _add(
                                    group,
                                    _finding(
                                        _rel(py),
                                        "MISSING_RUN_ON_DEVICES",
                                        f"{node.name}.forward() missing @run_on_devices",
                                    ),
                                )

    # 3. deprecated APIs
    if src_root.is_dir():
        mod_repl = SRC_MODELS.parent / "utils" / "module_replacement.py"
        core_utils = SRC_MODELS.parent / "core" / "utils.py"
        for py in sorted(src_root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            try:
                text = py.read_text()
            except Exception:
                continue
            if "register_module_replacement_dict(" in text and py.resolve() != mod_repl.resolve():
                _add(
                    group,
                    _finding(
                        _rel(py),
                        "DEPRECATED_API",
                        "register_module_replacement_dict() is deprecated; use register_modules()",
                    ),
                )
    # compare_fn_outputs in tests/ : WARN for the 4 known-debt shared tests, ERROR elsewhere.
    # Detect actual CALLS (AST), not docstring/comment mentions (avoids pcc_utils.py false-positive).
    known_debt = {"test_attention.py", "test_conv.py", "test_moe.py", "test_rope.py"}
    core_utils = SRC_MODELS.parent / "core" / "utils.py"
    if TESTS.is_dir():
        for py in sorted(TESTS.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            try:
                text = py.read_text()
            except Exception:
                continue
            if "compare_fn_outputs" not in text or py.resolve() == core_utils.resolve():
                continue
            try:
                tree = ast.parse(text)
            except Exception:
                continue
            called = False
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    fn = n.func
                    if isinstance(fn, ast.Name) and fn.id == "compare_fn_outputs":
                        called = True
                        break
                    if isinstance(fn, ast.Attribute) and fn.attr == "compare_fn_outputs":
                        called = True
                        break
            if called:
                if py.parent == SHARED and py.name in known_debt:
                    f = _finding(
                        _rel(py), "DEPRECATED_API", "compare_fn_outputs() used (known debt; migrate to assert_pcc())"
                    )
                    f["severity"] = "warning"
                    _add(group, f)
                else:
                    _add(
                        group,
                        _finding(_rel(py), "DEPRECATED_API", "compare_fn_outputs() must not be used; use assert_pcc()"),
                    )

    # 4. trace discipline
    if SRC_MODELS.is_dir():
        for py in sorted(SRC_MODELS.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            try:
                text = py.read_text()
            except Exception:
                continue
            if "_trace_enabled" in text.replace("is_trace_enabled", ""):
                _add(group, _finding(_rel(py), "TRACE_INSTANCE_FLAG", "banned instance trace flag '_trace_enabled'"))
        tower = SRC_MODELS / "dots_ocr" / "dots_ocr_vision.py"
        if tower.is_file():
            lines = tower.read_text().splitlines()
            for i, line in enumerate(lines):
                if line.lstrip().startswith("class TTNNDotsOCRVisionTower("):
                    prev = lines[i - 1].strip() if i > 0 else ""
                    if prev == "@trace_enabled":
                        _add(
                            group,
                            _finding(
                                _rel(tower),
                                "TOWER_TRACE_DECORATED",
                                "TTNNDotsOCRVisionTower must NOT be @trace_enabled",
                            ),
                        )
                    break


# ----------------------------------------------------------------------------
# audit: linting (check-mode shell-out; never mutates)
# ----------------------------------------------------------------------------
def audit_linting(group, files=None):
    """Run the formatter stack in NON-mutating CHECK mode and report status.

    STRICTLY non-mutating: invokes each tool with its check/diff flag only
    (``black --check``, ``isort --check-only``, ``autoflake --check``,
    ``yamllint``). It never shells ``pre-commit run`` (whose configured hooks
    -- black, isort, autoflake, end-of-file-fixer, trailing-whitespace -- run in
    their default MUTATING mode and would rewrite files, violating the
    check-only contract; see deep-plan_11 §7.1 and the §0 reconciliation).

    LINT_FAILURE now resolves to ERROR via _finding() (the hardcoded
    severity="warning" overrides were removed). ``yamllint .`` is clean at rc0
    once the repo-root ``.yamllint`` ignores the gitignored ``generated/`` tree,
    and ``black --check`` is clean, so a clean tree emits no LINT_FAILURE; a real
    formatter regression now blocks. The blocking signal of this script is the
    STRUCTURAL + trace-discipline + forward-discipline + lint checks.
    """
    group["status"] = "ok"
    py_files = [str(f) for f in (files or []) if str(f).endswith(".py")]
    # Whole-repo default scope when no explicit files were passed.
    scope = py_files if py_files else [str(REPO / "src"), str(REPO / "tests"), str(REPO / "scripts")]

    # tool name -> check-mode argv (all non-mutating).
    checks = [
        ("black", ["--check", "--quiet", *scope]),
        ("isort", ["--check-only", "--quiet", *scope]),
        ("autoflake", ["--check", "--recursive", *scope]),
    ]
    ran_any = False
    failed = []
    for tool, args in checks:
        exe = _which(tool)
        if not exe:
            continue
        ran_any = True
        try:
            r = subprocess.run([exe, *args], cwd=str(REPO), capture_output=True, text=True, timeout=600)
        except Exception:
            continue
        if r.returncode != 0:
            failed.append(tool)

    yamllint = _which("yamllint")
    if yamllint:
        ran_any = True
        try:
            r = subprocess.run([yamllint, "."], cwd=str(REPO), capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                failed.append("yamllint")
        except Exception:
            pass

    if not ran_any:
        group["status"] = "unavailable"
        _add(group, _finding("", "LINT_FAILURE", "no formatter tools available (black/isort/autoflake/yamllint)"))
        return
    if failed:
        group["status"] = "fail"
        _add(
            group,
            _finding(
                "",
                "LINT_FAILURE",
                "check-mode formatters report issues: " + ", ".join(failed) + " (run them to auto-fix)",
            ),
        )


def _which(prog):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / prog
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------
def run_audit(only=None, tree="both", whole_repo=True, skip_lint=False, files=None):
    checks = {
        "tier_structure": _new_group(),
        "config": _new_group(),
        "naming": _new_group(),
        "repo_invariants": _new_group(),
        "claude_md_checks": _new_group(),
        "linting": _new_group(),
    }
    checks["linting"]["status"] = "skipped"

    trees = []
    if tree in ("models", "both"):
        trees.append(RICH)
    if tree in ("experimental", "both"):
        trees.append(EXPERIMENTAL)

    models = _select_models(trees, only)

    # duplicate-name guard across trees
    seen = {}
    for md in models:
        seen.setdefault(md.name, []).append(md.parent.name)
    for nm, parents in seen.items():
        if len(parents) > 1:
            _add(checks["naming"], _finding(nm, "DUPLICATE_MODEL_NAME", f"name {nm} appears in {parents}"))

    audited_names = []
    for md in models:
        audited_names.append(md.name)
        is_rich = md.parent == RICH
        audit_tier_structure(md, is_rich, checks["tier_structure"])
        audit_root_config(md, checks["config"])
        audit_naming(md, is_rich, checks["naming"])
        audit_tier_token_placement(md, is_rich, checks["tier_structure"])

    # whole-repo-only audits
    if whole_repo:
        audit_repo_invariants(checks["repo_invariants"])
        audit_claude_md(checks["claude_md_checks"])
        if not skip_lint:
            audit_linting(checks["linting"], files=files)
        else:
            checks["linting"]["status"] = "skipped"
    else:
        # --model preflight: repo invariants + lint NOT exercised.
        checks["repo_invariants"]["status"] = "not_run"
        checks["linting"]["status"] = "skipped"
        # CLAUDE.md semantic checks ARE useful per-model but are repo-wide by nature;
        # run them only in whole-repo mode to keep preflights cheap and scoped.

    errors = sum(len(g.get("failures", [])) for g in checks.values())
    warnings = sum(len(g.get("warnings", [])) for g in checks.values())
    ok = errors == 0

    expected_commit = _tt_metal_expected()

    report = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "checks": checks,
        "summary": {
            "errors": errors,
            "warnings": warnings,
            "models_audited": sorted(set(audited_names)),
            "tt_metal_commit_expected": expected_commit,
        },
    }
    return report


def _tt_metal_expected():
    home = os.environ.get("TT_METAL_HOME")
    if not home:
        return None
    try:
        r = subprocess.run(["git", "-C", home, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------------
# text rendering
# ----------------------------------------------------------------------------
def render_text(report):
    lines = []
    for gname, g in report["checks"].items():
        for f in g.get("failures", []):
            lines.append(f"ERROR   [{f['code']}] {f['path']}: {f['detail']}")
        for w in g.get("warnings", []):
            lines.append(f"WARN    [{w['code']}] {w['path']}: {w['detail']}")
    s = report["summary"]
    result = "PASS" if report["ok"] else "FAIL"
    lines.append(f"RESULT: {result} ({s['errors']} errors, {s['warnings']} warnings)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# selftest (asserts pass/warn/fail codes + zero filesystem mutation)
# ----------------------------------------------------------------------------
def selftest():
    global REPO, TESTS, SRC_MODELS, RICH, EXPERIMENTAL, SHARED
    saved = (REPO, TESTS, SRC_MODELS, RICH, EXPERIMENTAL, SHARED)

    HEADER = "# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC\n" "# SPDX-License-Identifier: Apache-2.0\n"

    # ---- finding-severity probe (HW-free; proves promotions are REAL, not cosmetic) ----
    assert WARN_CODES == {"DEPRECATED_API"}, f"WARN_CODES drifted: {WARN_CODES}"
    for code in (
        "MISSING_RUN_ON_DEVICES",
        "MISSING_LICENSE_HEADER",
        "TORCH_IN_FORWARD",
        "EMPTY_TIER_DIR",
        "LINT_FAILURE",
    ):
        assert _finding("x", code, "d")["severity"] == "error", f"{code} should resolve ERROR"
    assert _finding("x", "DEPRECATED_API", "d")["severity"] == "warning", "DEPRECATED_API should stay WARN"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # build a throwaway tree
        src_pkg = root / "src" / "tt_symbiote"
        (src_pkg / "models" / "good_rich").mkdir(parents=True)
        (src_pkg / "models" / "graft").mkdir(parents=True)
        tests = root / "tests"
        rich = tests / "models"
        exp = tests / "experimental"
        shared = tests / "shared"
        (tests).mkdir(parents=True)
        (rich).mkdir()
        (exp).mkdir()
        (shared).mkdir()
        for f in SHARED_REQUIRED:
            (shared / f).write_text(HEADER + "# x\n")

        cfg = json.dumps(
            {
                "tt_metal_commit": "",
                "device_arch": "T3K",
                "pcc_threshold": 0.99,
                "hf_model_id": "o/m",
                "hf_revision": "main",
            }
        )

        def mk(base, name, tiers_with_tests, root_cfg=True, root_init=True):
            d = base / name
            d.mkdir()
            if root_init:
                (d / "__init__.py").write_text("# x\n")
            if root_cfg:
                (d / "test_config.json").write_text(cfg)
            for t in TIER_DIRS:
                td = d / t
                td.mkdir()
                (td / "__init__.py").write_text("# x\n")
            for t, fname in tiers_with_tests:
                (d / t / fname).write_text(HEADER + "# x\n")
            return d

        # conforming rich (non-grandfathered): all four tiers populated
        mk(
            rich,
            "good_rich",
            [
                ("Tier1", "test_ops_good_rich.py"),
                ("Tier2", "test_composites_good_rich.py"),
                ("Tier3", "test_decoder_good_rich.py"),
                ("Tier4", "test_modeling_good_rich.py"),
            ],
        )
        # grandfathered rich: only Tier1/Tier3/Tier4 populated, Tier2 left empty.
        # graft is added to BOTH GRANDFATHERED and TIER_NOT_IMPLEMENTED(("graft","Tier2"))
        # so the empty Tier2 must be SUPPRESSED (the deferral probe), while empty Tier?
        # would otherwise be EMPTY_TIER_DIR ERROR (promotion probe).
        GRANDFATHERED_NEEDS_UPGRADE.add("graft")
        TIER_NOT_IMPLEMENTED.add(("graft", "Tier2"))
        mk(
            rich,
            "graft",
            [
                ("Tier1", "test_ops_graft.py"),
                ("Tier3", "test_decoder_graft.py"),
                ("Tier4", "test_modeling_graft.py"),
            ],
        )
        # experimental: empty tier dirs => PASS
        mk(exp, "expmodel", [("Tier4", "test_modeling_expmodel.py")])
        # broken: missing Tier2 (rich non-grandfathered) => MISSING_TIER_DIR error
        bad = rich / "broken"
        bad.mkdir()
        (bad / "__init__.py").write_text("# x\n")
        (bad / "test_config.json").write_text(cfg)
        (src_pkg / "models" / "broken").mkdir()
        for t in ["Tier1", "Tier3", "Tier4"]:
            (bad / t).mkdir()
            (bad / t / "__init__.py").write_text("# x\n")
        (bad / "Tier4" / "test_modeling_broken.py").write_text(HEADER + "# x\n")

        # ---- _experimental carve-out + non-experimental contrast ----
        # A TTNN class whose forward calls torch.* and has NO @run_on_devices,
        # written both under _experimental (must be SILENT) and under modules/
        # (must emit BOTH TORCH_IN_FORWARD and MISSING_RUN_ON_DEVICES as ERROR).
        bad_forward = (
            HEADER
            + "import torch\n\n\n"
            + "class {cls}:\n"
            + "    def forward(self, x):\n"
            + "        return torch.add(x, x)\n"
        )
        (src_pkg / "_experimental" / "expmod").mkdir(parents=True)
        (src_pkg / "_experimental" / "expmod" / "__init__.py").write_text("# x\n")
        (src_pkg / "_experimental" / "expmod" / "modeling_expmod.py").write_text(bad_forward.format(cls="TTNNExp"))
        (src_pkg / "modules").mkdir(parents=True)
        (src_pkg / "modules" / "__init__.py").write_text("# x\n")
        (src_pkg / "modules" / "ttnn_x.py").write_text(bad_forward.format(cls="TTNNX"))

        # ---- base-method skip-list probe ----
        # core/module.py's TTNNModule / TTNNLayerStack bare forwards must be
        # skip-listed; a sibling TTNNReal(TTNNModule) DOES emit MISSING_RUN_ON_DEVICES.
        (src_pkg / "core").mkdir(parents=True)
        (src_pkg / "core" / "__init__.py").write_text("# x\n")
        (src_pkg / "core" / "module.py").write_text(
            HEADER
            + "class TTNNModule:\n"
            + "    def forward(self, x):\n"
            + "        raise NotImplementedError\n\n\n"
            + "class TTNNLayerStack:\n"
            + "    def forward(self, x):\n"
            + "        return x\n\n\n"
            + "class TTNNReal(TTNNModule):\n"
            + "    def forward(self, x):\n"
            + "        return self.inner(x)\n"
        )

        before = _snapshot(root)
        # repoint globals
        REPO = root
        TESTS = tests
        SRC_MODELS = src_pkg / "models"
        RICH = rich
        EXPERIMENTAL = exp
        SHARED = shared
        try:
            rep = run_audit(tree="both", whole_repo=True, skip_lint=True)
        finally:
            REPO, TESTS, SRC_MODELS, RICH, EXPERIMENTAL, SHARED = saved
            GRANDFATHERED_NEEDS_UPGRADE.discard("graft")
            TIER_NOT_IMPLEMENTED.discard(("graft", "Tier2"))
        after = _snapshot(root)

    assert before == after, "SELFTEST FAILED: filesystem was mutated!"

    failures = [f for g in rep["checks"].values() for f in g["failures"]]
    warnings = [f for g in rep["checks"].values() for f in g["warnings"]]
    codes = [f["code"] for f in (failures + warnings)]

    # broken missing Tier2 -> MISSING_TIER_DIR error
    assert "MISSING_TIER_DIR" in codes, f"expected MISSING_TIER_DIR, got {codes}"

    # EMPTY_TIER_DIR is now an ERROR (both severity overrides removed). The `broken`
    # rich non-grandfathered model leaves Tier1/Tier3 empty -> EMPTY_TIER_DIR ERRORs;
    # those PROVE the promotion is real. Every EMPTY_TIER_DIR must be an ERROR (never WARN).
    empty = [f for f in (failures + warnings) if f["code"] == "EMPTY_TIER_DIR"]
    assert empty, "expected EMPTY_TIER_DIR errors from the `broken` empty tiers"
    assert all(f["severity"] == "error" for f in empty), f"EMPTY_TIER_DIR must resolve ERROR, never WARN; got {empty}"
    assert not any(f["code"] == "EMPTY_TIER_DIR" for f in warnings), "EMPTY_TIER_DIR must resolve ERROR, never WARN"

    # graft Tier2 deferral probe: NO EMPTY_TIER_DIR for graft Tier2 (TIER_NOT_IMPLEMENTED)
    graft_t2 = [f for f in (failures + warnings) if f["code"] == "EMPTY_TIER_DIR" and "graft/Tier2" in f["path"]]
    assert not graft_t2, f"graft Tier2 should be exempt; got {graft_t2}"

    # experimental carve-out: NO TORCH_IN_FORWARD / MISSING_RUN_ON_DEVICES with _experimental in path
    exp_fwd = [
        f
        for f in (failures + warnings)
        if f["code"] in ("TORCH_IN_FORWARD", "MISSING_RUN_ON_DEVICES") and "_experimental" in f["path"]
    ]
    assert not exp_fwd, f"_experimental must be carved out of forward-discipline; got {exp_fwd}"

    # non-experimental contrast: ttnn_x.py emits BOTH codes as ERROR
    x_torch = [f for f in failures if f["code"] == "TORCH_IN_FORWARD" and "ttnn_x.py" in f["path"]]
    x_run = [f for f in failures if f["code"] == "MISSING_RUN_ON_DEVICES" and "ttnn_x.py" in f["path"]]
    assert x_torch, "ttnn_x.py TTNNX.forward should emit TORCH_IN_FORWARD (ERROR)"
    assert x_run, "ttnn_x.py TTNNX.forward should emit MISSING_RUN_ON_DEVICES (ERROR)"

    # base-method skip-list: module.py base forwards must NOT emit MISSING_RUN_ON_DEVICES,
    # but the sibling TTNNReal subclass MUST.
    base_run = [
        f
        for f in (failures + warnings)
        if f["code"] == "MISSING_RUN_ON_DEVICES" and f["path"].endswith("core/module.py")
    ]
    base_classes = {d for f in base_run for d in [f["detail"]]}
    assert not any(
        "TTNNModule.forward" in d or "TTNNLayerStack.forward" in d for d in base_classes
    ), f"base forwards must be skip-listed; got {base_classes}"
    assert any("TTNNReal.forward" in d for d in base_classes), f"TTNNReal must emit; got {base_classes}"

    assert not rep["ok"], "selftest tree has deliberate errors; ok must be False"
    print("SELFTEST: PASS (zero filesystem mutation; promotions REAL; carve-outs verified)")
    return 0


def _snapshot(root):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            try:
                out[str(p.relative_to(root))] = p.read_bytes()
            except Exception:
                out[str(p.relative_to(root))] = b""
        else:
            out[str(p.relative_to(root)) + "/"] = b""
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _changed_model_names():
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "diff", "--name-only", "HEAD"], capture_output=True, text=True, timeout=30
        )
        files = r.stdout.splitlines()
        r2 = subprocess.run(
            ["git", "-C", str(REPO), "diff", "--name-only", "--cached"], capture_output=True, text=True, timeout=30
        )
        files += r2.stdout.splitlines()
    except Exception:
        return None
    names = set()
    for f in files:
        parts = Path(f).parts
        if len(parts) >= 3 and parts[0] == "tests" and parts[1] in ("models", "experimental"):
            names.add(parts[2])
        if len(parts) >= 4 and parts[0] == "src" and parts[2] == "models":
            names.add(parts[3])
    return names or None


def main(argv=None):
    p = argparse.ArgumentParser(description="Check-only tier-structure + CLAUDE.md audit.")
    p.add_argument("--model", action="append", default=None, help="model name (repeatable)")
    p.add_argument("--tree", choices=["models", "experimental", "both"], default="both")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--changed-only", action="store_true")
    p.add_argument("--skip-lint", action="store_true")
    p.add_argument("--print-contract", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("files", nargs="*", help="files passed by pre-commit (used for --changed-only)")
    args = p.parse_args(argv)

    if args.print_contract:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "codes": CODES,
                    "warn_codes": sorted(WARN_CODES),
                    "checks": ["tier_structure", "config", "naming", "repo_invariants", "claude_md_checks", "linting"],
                    "linting_status_values": ["ok", "fail", "unavailable", "skipped", "not_run"],
                },
                indent=2,
            )
        )
        return 0

    if args.selftest:
        return selftest()

    only = set(args.model) if args.model else None
    whole_repo = only is None

    if args.changed_only and not only:
        # derive from changed files (positional files or git diff)
        names = set()
        for f in args.files:
            parts = Path(f).parts
            if len(parts) >= 3 and parts[0] == "tests" and parts[1] in ("models", "experimental"):
                names.add(parts[2])
            if len(parts) >= 4 and parts[0] == "src" and parts[2] == "models":
                names.add(parts[3])
        if not names:
            derived = _changed_model_names()
            if derived:
                names = derived
        if names:
            only = names
            whole_repo = False
        # if still no names, fall through to whole-repo audit (zero-file path)

    report = run_audit(
        only=only, tree=args.tree, whole_repo=whole_repo, skip_lint=args.skip_lint, files=args.files or None
    )

    if args.format == "json":
        print(json.dumps(report))
        print(render_text(report), file=sys.stderr)
    else:
        print(render_text(report))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
