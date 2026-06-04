# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for pi0.5 capability tests (importable; see conftest.py).

Adds the pi0_5 PyTorch reference tree to ``sys.path`` so the PCC golden
(``models.experimental.pi0_5.reference.*``) is importable, and exposes
skip helpers for the reference and the gated checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REFERENCE_ROOT = os.environ.get("PI05_REFERENCE_ROOT", "/home/ttuser/salnahari/pi0_5_ref")
if REFERENCE_ROOT not in sys.path:
    sys.path.insert(0, REFERENCE_ROOT)

CHECKPOINT_PATH = os.environ.get("PI05_CHECKPOINT_DIR", "/storage/sdawle/pi05_weights/pi05_base")
SEED = 42


def reference_available() -> bool:
    try:
        import models.experimental.pi0_5.reference.torch_gemma  # noqa: F401

        return True
    except Exception:
        return False


def require_reference():
    if not reference_available():
        pytest.skip(f"pi0_5 PyTorch reference not importable from {REFERENCE_ROOT}.")


def require_checkpoint() -> str:
    p = Path(CHECKPOINT_PATH)
    if not p.exists():
        pytest.skip(f"pi05_base checkpoint not found at {CHECKPOINT_PATH} (gated, ~14.5GB).")
    return str(p)


# --------------------------------------------------------------------------- #
# PCC adapters over the shared utility.
#
# The pi0.5 modules set ``_bypass_tensor_wrapping=True`` and return RAW
# ``ttnn.Tensor`` outputs (not ``TorchTTNNTensor``). The shared
# ``tests.shared.pcc_utils`` only extracts torch / TorchTTNNTensor, so it would
# see zero tensors for a raw ttnn output. These thin adapters coerce pi0.5
# outputs to torch, then delegate to the shared PCC math -- so the single
# source of truth for PCC stays in tests/shared, while ``compute_pcc`` keeps the
# float return the pi0.5 tests were written against.
# --------------------------------------------------------------------------- #
def _to_torch_out(x):
    """Coerce a pi0.5 module output (raw ttnn.Tensor / TorchTTNNTensor / torch /
    nested list/tuple) to a torch.Tensor (or matching nested structure)."""
    import torch

    if isinstance(x, (list, tuple)):
        return type(x)(_to_torch_out(i) for i in x)
    try:
        import ttnn

        if isinstance(x, ttnn.Tensor):
            return ttnn.to_torch(x)
    except ImportError:
        pass
    from tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(x, TorchTTNNTensor):
        return x.to_torch
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def compute_pcc(actual, expected) -> float:
    """Single-float PCC for pi0.5 (raw-ttnn-aware), via tests.shared.pcc_utils."""
    from tests.shared.pcc_utils import compute_pcc as _shared_compute_pcc

    results = _shared_compute_pcc(_to_torch_out(actual), _to_torch_out(expected))
    return results[0][0]


def assert_pcc(actual, expected, threshold: float = 0.99, msg: str = "") -> None:
    """assert_pcc for pi0.5 (raw-ttnn-aware), via tests.shared.pcc_utils."""
    from tests.shared.pcc_utils import assert_pcc as _shared_assert_pcc

    _shared_assert_pcc(_to_torch_out(actual), _to_torch_out(expected), threshold=threshold, msg=msg)
