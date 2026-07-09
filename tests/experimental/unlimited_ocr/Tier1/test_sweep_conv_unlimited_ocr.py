# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""FAST conv dtype/fidelity micro-sweep for the SAM conv chain (no 3.3B load).

Audits whether the 5 SAM convs (patch_embed / neck_conv1 / neck_conv2 / net_2 /
net_3) are PCC-safe under bfloat8_b weights (free DRAM save; conv is ~0.05% of
device time so the perf delta is negligible). Uses the REAL cached input/output
activations (fixtures/conv_*.pt) so PCC is faithful to the full model.

Grid per conv: weights_dtype {bfloat16, bfloat8_b} x math_fidelity {LoFi (the
shared default), HiFi2, HiFi4}. Builds the ttnn conv directly via Conv2dConfiguration
so weights_dtype/fidelity are configurable (the shared TTNNConv2dNHWC.forward hard-
codes bf16/LoFi). Per-config dealloc.

Run:
    pytest tests/experimental/unlimited_ocr/Tier1/test_sweep_conv_unlimited_ocr.py \
        -p no:randomly -q -s
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import ttnn

from tt_symbiote.modules.tt_cnn.builder import Conv2dConfiguration, TtConv2d
from tests.shared.pcc_utils import compute_pcc

FIX = Path(__file__).parent.parent / "fixtures"
OUT = Path(__file__).parent.parent / "sweep_results" / "conv_sweep.json"

pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)

_CONVS = ["patch_embed", "neck_conv1", "neck_conv2", "net_2", "net_3"]

# H,W the conv actually sees inside the SAM encoder at 1024x1024 (from
# _configure_convs_on_device in modeling_unlimited_ocr.py).
_SPATIAL = {
    "patch_embed": (1024, 1024),
    "neck_conv1": (64, 64),
    "neck_conv2": (64, 64),
    "net_2": (64, 64),
    "net_3": (32, 32),
}

_FID = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}


def _rebuild_conv(fx) -> nn.Conv2d:
    conv = nn.Conv2d(
        fx["in_channels"], fx["out_channels"], kernel_size=fx["kernel_size"],
        stride=fx["stride"], padding=fx["padding"], bias=fx["bias"],
    ).eval()
    conv.load_state_dict(fx["state_dict"])
    return conv


def _run_conv(device, conv, x_nchw, h, w, wdtype, fid):
    """Build+run one conv via Conv2dConfiguration with given weight dtype/fidelity.
    Returns the NHWC torch output (flattened form reshaped by caller)."""
    b, c_in = x_nchw.shape[0], x_nchw.shape[1]
    weight = ttnn.from_torch(conv.weight, dtype=ttnn.float32)
    bias = None
    if conv.bias is not None:
        bias = ttnn.from_torch(conv.bias.reshape(1, 1, 1, -1), dtype=ttnn.float32)
    cfg = Conv2dConfiguration(
        input_height=h, input_width=w,
        in_channels=c_in, out_channels=conv.out_channels, batch_size=b,
        kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
        groups=conv.groups, dilation=conv.dilation, weight=weight, bias=bias,
        weights_dtype=wdtype, math_fidelity=fid,
    )
    x_nhwc = ttnn.from_torch(
        x_nchw.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    layer = TtConv2d(cfg, device)
    out, hw = layer(x_nhwc, return_output_dim=True)
    got = ttnn.to_torch(out).reshape(b, hw[0], hw[1], conv.out_channels)
    ttnn.deallocate(out)
    return got


def test_conv_dtype_fidelity_sweep(device):
    results = {}
    for name in _CONVS:
        fpath = FIX / f"conv_{name}.pt"
        if not fpath.exists():
            pytest.skip(f"missing fixture {fpath.name}")
        fx = torch.load(fpath, weights_only=False)
        conv = _rebuild_conv(fx)
        x_nchw = fx["input_nchw"]
        h, w = _SPATIAL[name]
        ref_nhwc = fx["output_nchw"].permute(0, 2, 3, 1).contiguous()
        results[name] = {}
        for dname, wdtype in _DTYPE.items():
            for fname, fid in _FID.items():
                got = _run_conv(device, conv, x_nchw, h, w, wdtype, fid)
                pcc, maxdiff = compute_pcc(got.reshape(ref_nhwc.shape), ref_nhwc)[0]
                key = f"{dname}/{fname}"
                results[name][key] = {"pcc": float(pcc), "maxdiff": float(maxdiff)}
                print(f"[conv-sweep {name:12s}] {key:16s} PCC={pcc:.6f} maxdiff={maxdiff:.4g}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    # Report-only sweep: never fails (the apply decision is made from the numbers).
    assert results
