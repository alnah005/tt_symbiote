# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier1 (leaf-op) PCC tests for diffusion_gemma.

Validates the individual TTNN ops the model relies on against their torch
reference, at the real architectural shapes (see ../shapes.json):
  * ttnn.linear            (gate/up, down, router projections)
  * ttnn.rms_norm          (scaled hidden norm; weightless; per-head)
  * ttnn.gelu              (gelu_pytorch_tanh activation)

Device: P150x4 (4-chip Blackhole mesh) via the ttnn-plugin ``mesh_device``
fixture. ``pcc_threshold`` comes from the ROOT tests/conftest.py.
"""

import json
from pathlib import Path

import pytest
import torch
import ttnn

from tests.shared.pcc_utils import assert_pcc

_SHAPES = json.loads((Path(__file__).parent.parent / "shapes.json").read_text())
_CFG = _SHAPES["model_config"]
_EPS = _CFG["rms_norm_eps"]

# 1x4 Blackhole mesh; inputs replicated across devices.
_DEVICE_PARAMS = [{"mesh_shape": (1, 4)}]


def _from_torch(t, mesh_device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_torch(t, mesh_device):
    # Inputs are replicated, so every shard is identical -> take the first.
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        : t.shape[0] if hasattr(t, "shape") else None
    ]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq", [32, 128])
def test_linear_gate_up(mesh_device, pcc_threshold, seq):
    """ttnn.linear: hidden(2816) -> intermediate(2112)."""
    torch.manual_seed(0)
    x = torch.randn(1, seq, _CFG["hidden_size"], dtype=torch.bfloat16)
    w = torch.randn(_CFG["intermediate_size"], _CFG["hidden_size"], dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x.float(), w.float())

    tt_x = _from_torch(x, mesh_device)
    tt_w = _from_torch(w.transpose(0, 1).contiguous(), mesh_device)
    tt_out = ttnn.linear(tt_x, tt_w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = _to_torch(tt_out, mesh_device)[:1]

    assert_pcc(got, ref, threshold=pcc_threshold, msg="linear gate/up")


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq", [32, 128])
def test_rms_norm_scaled(mesh_device, pcc_threshold, seq):
    """ttnn.rms_norm with a learnable scale, matching DiffusionGemmaRMSNorm."""
    torch.manual_seed(1)
    dim = _CFG["hidden_size"]
    x = torch.randn(1, seq, dim, dtype=torch.bfloat16)
    weight = torch.randn(dim, dtype=torch.bfloat16)

    xf = x.float()
    normed = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + _EPS, -0.5)
    ref = (normed * weight.float()).to(torch.bfloat16).float()

    tt_x = _from_torch(x, mesh_device)
    tt_w = _from_torch(weight.unsqueeze(0), mesh_device)
    tt_out = ttnn.rms_norm(tt_x, weight=tt_w, epsilon=_EPS)
    got = _to_torch(tt_out, mesh_device)[:1]

    assert_pcc(got, ref, threshold=pcc_threshold, msg="rms_norm scaled")


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq", [32, 128])
def test_rms_norm_weightless(mesh_device, pcc_threshold, seq):
    """ttnn.rms_norm without scale (router pre-norm / per-head v_norm)."""
    torch.manual_seed(2)
    dim = _CFG["hidden_size"]
    x = torch.randn(1, seq, dim, dtype=torch.bfloat16)
    xf = x.float()
    ref = (xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + _EPS, -0.5)).to(torch.bfloat16).float()

    tt_x = _from_torch(x, mesh_device)
    tt_out = ttnn.rms_norm(tt_x, epsilon=_EPS)
    got = _to_torch(tt_out, mesh_device)[:1]

    assert_pcc(got, ref, threshold=pcc_threshold, msg="rms_norm weightless")


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq", [32, 128])
def test_gelu_tanh(mesh_device, pcc_threshold, seq):
    """ttnn.gelu (tanh approx) vs torch gelu_pytorch_tanh."""
    torch.manual_seed(3)
    x = torch.randn(1, seq, _CFG["intermediate_size"], dtype=torch.bfloat16)
    ref = torch.nn.functional.gelu(x.float(), approximate="tanh")

    tt_x = _from_torch(x, mesh_device)
    tt_out = ttnn.gelu(tt_x)
    got = _to_torch(tt_out, mesh_device)[:1]

    assert_pcc(got, ref, threshold=pcc_threshold, msg="gelu tanh")
