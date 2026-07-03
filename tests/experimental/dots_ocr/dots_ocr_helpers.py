# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the dots.ocr tiered capability tests.

These mirror the proven driving logic in the tt-metal reference test
(``models/experimental/tt_symbiote/tests/test_dots_ocr.py``) but target the
downstream tt_symbiote public surface:

  * ``set_device(obj, device)`` is the STRICT two-argument bind (it subsumes
    ``preprocess_weights`` + ``move_weights_to_device``); the tt-metal
    reference's ``register_forward_hook=``/``dump_visualization=`` kwargs do
    NOT exist here.
  * PCC is asserted with ``tests.shared.pcc_utils`` (the repo single source of
    truth). dots.ocr modules return raw ``ttnn.Tensor`` distributed on the
    mesh, so we convert to torch with an explicit ``ConcatMeshToTensor`` first
    (a bare ``ttnn.to_torch`` errors on a multi-device tensor).

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"
"""

from __future__ import annotations

import os

import torch

import ttnn

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"

# Mesh shapes keyed by the MESH_DEVICE env var. The DP map is used when
# DOTS_OCR_PARALLELISM=DP (the dots.ocr data-parallel decode path); T3K -> (8, 1).
MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
    "P150x4": (4, 1),
}


def _is_dp() -> bool:
    return os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP"


def resolve_mesh_device_shape():
    """Resolve the (rows, cols) mesh shape for the indirect ``mesh_device`` fixture."""
    mesh_device = os.environ.get("MESH_DEVICE")
    if _is_dp():
        return DOTS_OCR_DP_MESH_DEVICE_MAP.get(
            mesh_device, MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))
        )
    return MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))


def mesh_num_devices() -> int:
    sh = resolve_mesh_device_shape()
    if isinstance(sh, int):
        return max(1, int(sh))
    if isinstance(sh, (tuple, list)):
        if len(sh) >= 2:
            return int(sh[0]) * int(sh[1])
        if len(sh) == 1:
            return int(sh[0])
    return 1


def dots_ocr_device_params() -> dict:
    dp = {"trace_region_size": 300000000, "num_command_queues": 1}
    if mesh_num_devices() > 1:
        dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        dp["fabric_config"] = ttnn.FabricConfig.DISABLED
    return dp


def pipeline_batch_size() -> int:
    """DP requires ``batch_size == num_devices`` (one stream per chip)."""
    if not _is_dp():
        return 1
    n = mesh_num_devices()
    return n if n > 1 else 1


def stack_input_ids_for_dp(input_ids: torch.Tensor) -> torch.Tensor:
    """Turn ``[1, S]`` into ``[B, S]`` by repeating the prompt on each DP stream."""
    bs = pipeline_batch_size()
    if bs <= 1 or input_ids.shape[0] == bs:
        return input_ids
    if input_ids.shape[0] != 1:
        raise ValueError(f"DP batch stacking expects base shape [1, S], got {tuple(input_ids.shape)}")
    return input_ids.expand(bs, -1).contiguous()


def resolve_model_path() -> str:
    """env var > HF cache snapshot > bare model id."""
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def canonical_to_torch(output, mesh_device) -> torch.Tensor:
    """Convert a (possibly multi-device) raw ``ttnn.Tensor`` to a single torch tensor.

    On a multi-device mesh, DP-with-batch=1 and TP-after-all-reduce both produce
    identical data on every device, so we stack per-device slices along batch
    (``ConcatMeshToTensor(dim=0)``) and take the first as the canonical output.
    """
    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    if isinstance(output, ttnn.Tensor):
        if num_devices > 1:
            return ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
        return ttnn.to_torch(output)
    return output


# PCC adapters delegating to the repo single source of truth. Inputs are already
# converted to torch via ``canonical_to_torch``, so the shared math applies directly.
def compute_pcc(actual, expected):
    from tests.shared.pcc_utils import compute_pcc as _shared

    return _shared(actual, expected)


def assert_pcc(actual, expected, threshold: float = 0.99, msg: str = "") -> None:
    from tests.shared.pcc_utils import assert_pcc as _shared

    _shared(actual, expected, threshold=threshold, msg=msg)


def assert_l1_resident(tensor, name: str) -> None:
    assert isinstance(tensor, ttnn.Tensor), f"{name} should be a TTNN tensor"
    assert tensor.memory_config().buffer_type == ttnn.BufferType.L1, f"{name} should reside in L1"
