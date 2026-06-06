#!/usr/bin/env bash
# scripts/bootstrap_venv.sh
#
# Stand up a working tt_symbiote env from a clean machine, using a tt_metal
# SOURCE BUILD at $TT_METAL_HOME for ttnn (NOT a PyPI wheel).
#
# OPTIONAL convenience: tt_symbiote auto-wires the source-built ttnn from
# $TT_METAL_HOME at import time, so `pip install -e .` (with $TT_METAL_HOME set)
# is already enough to `import tt_symbiote`. This script adds the extras: a
# persistent .pth (so a bare `import ttnn` works too), an sfpi pre-flight check,
# and the git pre-commit hook.
#
# What this does
# --------------
# 1. Requires $TT_METAL_HOME to be set and a git checkout. ttnn is imported
#    from that source build (not pip-installed). There is NO global tt-metal
#    commit pin — each model declares its own tt_metal commit in its
#    test_config.json / modeling_<model>.py, validated per test by the autouse
#    tt_metal_commit_check fixture. This script builds against whatever commit
#    your $TT_METAL_HOME checkout is on and just records it for the log.
#
# 2. Derives the required sfpi version from $TT_METAL_HOME/tt_metal/sfpi-version
#    (format: sfpi_version='X.Y.Z') unless SFPI_REQUIRED is set in the pin file.
#    Probes the system-wide sfpi RISC-V toolchain at /opt/tenstorrent/sfpi/ and
#    refuses to proceed if it doesn't match (ttnn JIT-compiles firmware kernels
#    with that compiler at first device-open).
#
# 3. Creates a fresh venv (default ${REPO_ROOT}/.venv), upgrades pip, makes the
#    source-built ttnn importable (via a .pth pointing at $TT_METAL_HOME and
#    $TT_METAL_HOME/ttnn), and pip-installs tt_symbiote in editable mode.
#    tt_symbiote's transitive deps (torch, transformers==5.9.0, accelerate, ...)
#    come along via pyproject.toml. NOTE: ttnn is NOT pip-installed.
#
# 4. Smoke-checks the result by importing every public surface and confirming
#    ttnn resolves under $TT_METAL_HOME.
#
# Usage
# -----
#   ./scripts/bootstrap_venv.sh                    # default: ./.venv
#   VENV=/tmp/foo ./scripts/bootstrap_venv.sh      # custom location
#   PYTHON=python3.11 ./scripts/bootstrap_venv.sh  # override interpreter
#
# tt-metal commit policy
# -----------------------
# There is NO global commit to update here. Per-model tt_metal commits live in
# each model's test_config.json ("tt_metal_commit") and modeling_<model>.py
# (TT_METAL_COMMIT); drift is surfaced per test by the autouse
# tt_metal_commit_check fixture. SFPI is auto-derived from
# $TT_METAL_HOME/tt_metal/sfpi-version unless SFPI_REQUIRED overrides it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$REPO_ROOT/.venv}"
PIN_FILE="$REPO_ROOT/scripts/ttnn-pin.txt"
SFPI_GCC="/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

if [[ ! -f "$PIN_FILE" ]]; then
    echo "error: $PIN_FILE is missing; can't determine which tt-metal commit to build against"
    exit 2
fi

# scripts/ttnn-pin.txt is a tiny shell-sourceable file.
# shellcheck source=/dev/null
source "$PIN_FILE"

echo "==> Checking \$TT_METAL_HOME"
if [[ -z "${TT_METAL_HOME:-}" ]]; then
    cat <<EOF
error: \$TT_METAL_HOME is not set. ttnn is provided by a tt_metal SOURCE BUILD,
not a PyPI wheel. Set \$TT_METAL_HOME to your tt-metal checkout (with ttnn built)
before running this script.
EOF
    exit 2
fi
if [[ ! -d "$TT_METAL_HOME/.git" ]]; then
    echo "error: \$TT_METAL_HOME ($TT_METAL_HOME) is not a git checkout"
    exit 2
fi

TT_METAL_HEAD="$(git -C "$TT_METAL_HOME" rev-parse HEAD)"
# No global commit pin: each model carries its own tt_metal_commit (validated per
# test by the autouse tt_metal_commit_check fixture). Just record the build commit.
echo "  building against \$TT_METAL_HOME HEAD $TT_METAL_HEAD"
echo "  (per-model tt_metal commits are checked at test time, not here)"

echo "==> Deriving required sfpi version"
SFPI_VERSION_FILE="$TT_METAL_HOME/tt_metal/sfpi-version"
if [[ -z "${SFPI_REQUIRED:-}" ]]; then
    if [[ ! -f "$SFPI_VERSION_FILE" ]]; then
        echo "error: cannot auto-derive sfpi: $SFPI_VERSION_FILE missing. Set SFPI_REQUIRED in $PIN_FILE."
        exit 3
    fi
    # format: sfpi_version='7.52.0'
    SFPI_REQUIRED="$(grep -oE "sfpi_version='[0-9.]+'" "$SFPI_VERSION_FILE" | head -1 | cut -d\' -f2 || true)"
    if [[ -z "$SFPI_REQUIRED" ]]; then
        echo "error: could not parse sfpi_version from $SFPI_VERSION_FILE"
        exit 3
    fi
    echo "  derived sfpi $SFPI_REQUIRED from $SFPI_VERSION_FILE"
else
    echo "  sfpi $SFPI_REQUIRED (from \$SFPI_REQUIRED override)"
fi

echo "==> Probing system sfpi"
if [[ ! -x "$SFPI_GCC" ]]; then
    cat <<EOF
error: sfpi RISC-V toolchain not found at $SFPI_GCC.
This is a Tenstorrent system prerequisite (installed by the official
tt-installer apt package), not a tt-metal dependency. Without it, ttnn
cannot JIT-compile its firmware kernels.
See https://github.com/tenstorrent/sfpi/releases for releases.
EOF
    exit 3
fi

SFPI_FOUND=$("$SFPI_GCC" --version 2>/dev/null \
             | head -1 \
             | grep -oE 'sfpi:[0-9.]+' \
             | cut -d: -f2 || true)

if [[ -z "$SFPI_FOUND" ]]; then
    echo "error: could not parse sfpi version from $SFPI_GCC --version"
    exit 3
fi

if [[ "$SFPI_FOUND" != "$SFPI_REQUIRED" ]]; then
    cat <<EOF
error: sfpi version mismatch.
  found at $SFPI_GCC:                $SFPI_FOUND
  required by tt_metal source build: $SFPI_REQUIRED
Either install the matching sfpi system-wide, or set SFPI_REQUIRED in
scripts/ttnn-pin.txt to override.
EOF
    exit 3
fi
echo "  sfpi $SFPI_FOUND OK (matches tt_metal source build)"

if [[ -e "$VENV" ]]; then
    echo "==> $VENV already exists. Refusing to overwrite — delete it or set VENV=."
    exit 1
fi

echo "==> Creating venv at $VENV (python: $PYTHON)"
"$PYTHON" -m venv "$VENV"

# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Making source-built ttnn importable (.pth -> \$TT_METAL_HOME)"
SITE_PKGS="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PTH_FILE="$SITE_PKGS/tt_metal_ttnn.pth"
{
    echo "$TT_METAL_HOME"
    echo "$TT_METAL_HOME/ttnn"
} > "$PTH_FILE"
echo "  wrote $PTH_FILE"

echo "==> Installing tt_symbiote (editable). NOTE: ttnn comes from the source build, not pip."
# Torch CPU wheel comes from PyTorch's index; everything else from PyPI.
pip install --extra-index-url "$TORCH_INDEX" -e "$REPO_ROOT"

# Wire the git pre-commit hook so the tier-structure + lint checks run locally on
# every commit. Git does not install hooks on clone (by design), so this is the
# post-clone step that enables them; CI (.github/workflows/lint.yml) enforces the
# same hooks server-side regardless. `pre-commit` ships in the `dev` extra.
if command -v pre-commit >/dev/null 2>&1; then
    echo "==> Installing git pre-commit hook (pre-commit install)"
    pre-commit install
else
    echo "==> Skipping pre-commit hook install (pre-commit not found; pip install '.[dev]' to enable)"
fi

echo
echo "==> Smoke check"
python - <<'PY'
import os
import sys
import torch
import transformers
import ttnn
import tt_symbiote

ok = "ok" if hasattr(tt_symbiote, "AutoModelForCausalLM") else "MISSING"
have_recipe = "BailingMoeV2ForCausalLM" in tt_symbiote.TT_MODEL_REGISTRY
ttnn_under_home = os.environ.get("TT_METAL_HOME", "") in (ttnn.__file__ or "")

print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"ttnn from   : {ttnn.__file__}  (under $TT_METAL_HOME: {ttnn_under_home})")
print(f"tt_symbiote : AutoModelForCausalLM={ok}, set_device={'ok' if hasattr(tt_symbiote, 'set_device') else 'MISSING'}")
print(f"recipe ready: BailingMoeV2ForCausalLM in TT_MODEL_REGISTRY = {have_recipe}")

if ok != "ok" or not have_recipe or not ttnn_under_home:
    sys.exit(1)
PY

echo
echo "==> Done. To use:"
echo "    source $VENV/bin/activate"
echo "    python /path/to/run_ling.py    # e.g. the canonical example"
