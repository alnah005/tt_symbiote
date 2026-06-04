# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Machine-checkable structure lint for the tt_symbiote test layout.

Stdlib-only (os/json/re/warnings/pathlib + pytest). Imports NOTHING from
tt_symbiote or ttnn, so it runs in any software-only environment. Enforces the
two-tree per-model taxonomy:

  - tests/models/<name>/        RICH floor   (e2e-traced-correct)
  - tests/experimental/<name>/  MINIMAL floor (partial TTNN)
  - tests/shared/               shared helpers + shared capability tests
  - the old per-model capabilities tree is REMOVED

Every per-model dir carries a schema-valid test_config.json. The 4 historically
thin "grandfathered" model dirs are allowed to skip the rich floor (warn, not
fail) until they can be upgraded offline.
"""

import json
import re
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"
SRC_MODELS = REPO / "src" / "tt_symbiote" / "models"
RICH = TESTS / "models"
EXPERIMENTAL = TESTS / "experimental"
SHARED = TESTS / "shared"

REQUIRED_CONFIG_KEYS = {
    "tt_metal_commit": str,
    "device_arch": str,
    "pcc_threshold": (int, float),
    "hf_model_id": str,
    "hf_revision": str,
}


def RICH_REQUIRED(n):
    return [
        "__init__.py", "test_config.json", "shapes.json", "op_map.json",
        f"test_ops_{n}.py", f"test_composites_{n}.py", f"test_decoder_{n}.py",
        f"test_modeling_{n}.py", f"test_traced_{n}.py",
    ]


MINIMAL_REQUIRED = ["__init__.py", "test_config.json"]
SHARED_REQUIRED = [
    "__init__.py", "conftest.py", "pcc_utils.py", "shared_configs.py",
    "test_attention.py", "test_conv.py", "test_moe.py", "test_rope.py", "test_dpl.py",
]
GRANDFATHERED_NEEDS_UPGRADE = {"bailing_moe_v2", "gemma4", "qwen3_vl", "resnet"}
BANNED_ALIASES = {"owl_vit", "speech_t5", "qwen_omni", "gr00t"}
BANNED_SUFFIX_RE = re.compile(r"_(4_7|5|flash|coder_next)$")
DIR_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
SKIP_DIRS = {"__pycache__"}
NON_MODEL_FILES = {"README.md", "__init__.py"}
TIER_FILE_RE = re.compile(r"^test_(ops|composites|decoder|modeling|traced)_(.+)\.py$")
VARIANT_RE = re.compile(r"_variant_.+$")


def _model_dirs(tree):
    return [] if not tree.is_dir() else [
        p for p in sorted(tree.iterdir())
        if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("__")
    ]


def test_capabilities_removed():
    assert not (TESTS / "capabilities").exists(), (
        "the old per-model capabilities tree must be removed"
    )


def test_two_tree_exclusivity():
    allowed = {"auto", "shared", "models", "experimental", "__pycache__"}
    for child in TESTS.iterdir():
        if child.is_dir():
            assert child.name in allowed, f"unexpected dir under tests/: {child.name}"
    # only the two per-model trees may host a test_config.json
    for child in TESTS.iterdir():
        if child.is_dir() and child.name not in {"models", "experimental"}:
            for cfg in child.rglob("test_config.json"):
                pytest.fail(f"test_config.json outside per-model trees: {cfg}")


def test_shared_tree():
    assert SHARED.is_dir(), "tests/shared/ must exist"
    for f in SHARED_REQUIRED:
        assert (SHARED / f).is_file(), f"missing tests/shared/{f}"


@pytest.mark.parametrize(
    "model_dir",
    _model_dirs(RICH) + _model_dirs(EXPERIMENTAL),
    ids=lambda p: p.name,
)
def test_config_present_and_valid(model_dir):
    cfg = model_dir / "test_config.json"
    assert cfg.is_file(), f"missing test_config.json in {model_dir}"
    data = json.loads(cfg.read_text())
    for key, typ in REQUIRED_CONFIG_KEYS.items():
        assert key in data, f"{cfg}: missing key {key}"
        assert isinstance(data[key], typ), (
            f"{cfg}: key {key} has wrong type {type(data[key])}, expected {typ}"
        )


@pytest.mark.parametrize("model_dir", _model_dirs(RICH), ids=lambda p: p.name)
def test_rich_floor(model_dir):
    name = model_dir.name
    if name in GRANDFATHERED_NEEDS_UPGRADE:
        for f in MINIMAL_REQUIRED:
            assert (model_dir / f).is_file(), f"missing {f} in grandfathered {name}"
        warnings.warn(f"{name} GRANDFATHERED_NEEDS_UPGRADE")
        return
    for f in RICH_REQUIRED(name):
        assert (model_dir / f).is_file(), f"RICH floor: missing {f} in {name}"


@pytest.mark.parametrize("model_dir", _model_dirs(EXPERIMENTAL), ids=lambda p: p.name)
def test_minimal_floor(model_dir):
    for f in MINIMAL_REQUIRED:
        assert (model_dir / f).is_file(), f"MINIMAL floor: missing {f} in {model_dir.name}"


@pytest.mark.parametrize(
    "model_dir",
    _model_dirs(RICH) + _model_dirs(EXPERIMENTAL),
    ids=lambda p: p.name,
)
def test_naming(model_dir):
    name = model_dir.name
    assert name not in BANNED_ALIASES, f"banned alias dir name: {name}"
    assert not BANNED_SUFFIX_RE.search(name), f"banned variant-suffix dir name: {name}"
    assert DIR_NAME_RE.match(name), f"dir name not canonical-shaped: {name}"
    # RICH dirs must correspond to a real src package (transformers-canonical).
    if model_dir.parent == RICH:
        assert (SRC_MODELS / name).is_dir(), (
            f"RICH dir {name} has no src/tt_symbiote/models/{name} package"
        )


@pytest.mark.parametrize(
    "model_dir",
    _model_dirs(RICH) + _model_dirs(EXPERIMENTAL),
    ids=lambda p: p.name,
)
def test_tier_file_token_match(model_dir):
    name = model_dir.name
    for py in model_dir.glob("test_*.py"):
        m = TIER_FILE_RE.match(py.name)
        if not m:
            continue
        token = m.group(2)
        if token == name:
            continue
        # experimental allows test_modeling_<name>_variant_<qualifier>.py
        if model_dir.parent == EXPERIMENTAL and token.startswith(name) and VARIANT_RE.search(token):
            continue
        pytest.fail(f"{py.name}: tier token '{token}' does not match dir '{name}'")


def test_no_banned_trace_path():
    if not SRC_MODELS.is_dir():
        return
    for py in SRC_MODELS.rglob("*.py"):
        text = py.read_text()
        assert "_trace_enabled" not in text.replace("is_trace_enabled", ""), (
            f"banned instance trace flag '_trace_enabled' in {py}"
        )


def test_tower_not_trace_decorated():
    f = SRC_MODELS / "dots_ocr" / "dots_ocr_vision.py"
    assert f.is_file(), "dots_ocr_vision.py missing"
    lines = f.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("class TTNNDotsOCRVisionTower("):
            prev = lines[i - 1].strip() if i > 0 else ""
            assert prev != "@trace_enabled", (
                "TTNNDotsOCRVisionTower must NOT be @trace_enabled (would change "
                "global is_trace_enabled / TracedRun dispatch)"
            )
            return
    pytest.fail("class TTNNDotsOCRVisionTower not found")
