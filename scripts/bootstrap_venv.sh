#!/usr/bin/env bash
# scripts/bootstrap_venv.sh
#
# Stand up a working tt_symbiote env from a clean machine, with no
# tt-metal source checkout or tt-metal-built ttnn required.
#
# What this does
# --------------
# 1. Probes the system-wide sfpi RISC-V toolchain at /opt/tenstorrent/sfpi/.
#    ttnn JIT-compiles firmware kernels at first device-open time with that
#    compiler. Each PyPI ttnn release pins exactly one sfpi version (see the
#    `ttnn/tt_metal/sfpi-version` file inside the wheel); if the installed
#    sfpi doesn't match, ttnn rejects the host's compiler flags and the
#    open_mesh_device() call dies. This script reads
#    `scripts/ttnn-pin.txt` to learn which (ttnn, sfpi) pair is supported
#    and refuses to proceed if the system sfpi doesn't match.
#
# 2. Creates a fresh venv (default ${REPO_ROOT}/.venv), upgrades pip, and
#    pip-installs the pinned ttnn version plus tt_symbiote in editable
#    mode. tt_symbiote's transitive deps (torch, transformers==5.9.0,
#    accelerate, tokenizers, ...) come along via pyproject.toml.
#
# 3. Smoke-checks the result by importing every public surface
#    (AutoModelForCausalLM, set_device) and confirming the
#    BailingMoeV2ForCausalLM recipe is in TT_MODEL_REGISTRY.
#
# Usage
# -----
#   ./scripts/bootstrap_venv.sh                    # default: ./.venv
#   VENV=/tmp/foo ./scripts/bootstrap_venv.sh      # custom location
#   PYTHON=python3.11 ./scripts/bootstrap_venv.sh  # override interpreter
#
# Updating the pin
# ----------------
# When you upgrade the system sfpi (or want to chase a newer ttnn), find
# the (ttnn, sfpi) pair that matches and update scripts/ttnn-pin.txt.
# `pip download --no-deps ttnn==<v>` and inspect `tt_metal/sfpi-version`
# inside the wheel — that file is the source of truth.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$REPO_ROOT/.venv}"
PIN_FILE="$REPO_ROOT/scripts/ttnn-pin.txt"
SFPI_GCC="/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

if [[ ! -f "$PIN_FILE" ]]; then
    echo "error: $PIN_FILE is missing; can't determine which ttnn version to install"
    exit 2
fi

# scripts/ttnn-pin.txt is a tiny shell-sourceable file with two vars.
# shellcheck source=/dev/null
source "$PIN_FILE"

if [[ -z "${TTNN_VERSION:-}" || -z "${SFPI_REQUIRED:-}" ]]; then
    echo "error: $PIN_FILE must set TTNN_VERSION and SFPI_REQUIRED"
    exit 2
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
  found at $SFPI_GCC: $SFPI_FOUND
  required by ttnn==$TTNN_VERSION:    $SFPI_REQUIRED
Either install the matching sfpi system-wide, or update
scripts/ttnn-pin.txt to a ttnn version whose sfpi matches what you have.
EOF
    exit 3
fi
echo "  sfpi $SFPI_FOUND OK (matches ttnn==$TTNN_VERSION)"

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

echo "==> Installing ttnn==$TTNN_VERSION and tt_symbiote (editable)"
# Torch CPU wheel comes from PyTorch's index; everything else from PyPI.
pip install --extra-index-url "$TORCH_INDEX" \
    "ttnn==$TTNN_VERSION" \
    -e "$REPO_ROOT"

echo
echo "==> Smoke check"
python - <<'PY'
import sys
import torch
import transformers
import ttnn
import tt_symbiote

ok = "ok" if hasattr(tt_symbiote, "AutoModelForCausalLM") else "MISSING"
have_recipe = "BailingMoeV2ForCausalLM" in tt_symbiote.TT_MODEL_REGISTRY

print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"ttnn from   : {ttnn.__file__}")
print(f"tt_symbiote : AutoModelForCausalLM={ok}, set_device={'ok' if hasattr(tt_symbiote, 'set_device') else 'MISSING'}")
print(f"recipe ready: BailingMoeV2ForCausalLM in TT_MODEL_REGISTRY = {have_recipe}")

if ok != "ok" or not have_recipe:
    sys.exit(1)
PY

echo
echo "==> Done. To use:"
echo "    source $VENV/bin/activate"
echo "    python /path/to/run_ling.py    # e.g. the canonical example"
