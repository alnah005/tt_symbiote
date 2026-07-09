# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""FAST per-conv micro-tests for the SAM conv chain (no 3.3B model load).

These use REAL cached activations (from fixtures/capture_reference_activations.py:
the actual input each conv sees inside the SAM encoder + its fp32 output) so the
PCC is faithful to the full model -- but each test just rebuilds ONE nn.Conv2d
from its saved weights, runs it on TTNN, and asserts PCC. Iteration = open device
+ 1 conv (~seconds), instead of loading the 3.3B model + running the whole vision
tower every time.

Regenerate fixtures once (CPU, no device):
    $TT_METAL_HOME/python_env/bin/python \
        tests/experimental/unlimited_ocr/fixtures/capture_reference_activations.py

Fast loop while tuning the conv path (e.g. reshape_output / block config):
    pytest tests/experimental/unlimited_ocr/Tier1/test_micro_conv_unlimited_ocr.py \
        -k net_2 -p no:randomly -q -s
"""

import json
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import ttnn

from tt_symbiote.utils.device_management import set_device
from tests.shared.pcc_utils import assert_pcc, compute_pcc

FIX = Path(__file__).parent.parent / "fixtures"
# Strict bring-up threshold so the micro-tests are SENSITIVE to precision changes
# (real convs on real activations comfortably clear this; a drop flags a regression).
PCC = 0.999

pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)

_CONVS = ["patch_embed", "neck_conv1", "neck_conv2", "net_2", "net_3"]


def _rebuild_conv(fx) -> nn.Conv2d:
    conv = nn.Conv2d(
        fx["in_channels"], fx["out_channels"], kernel_size=fx["kernel_size"],
        stride=fx["stride"], padding=fx["padding"], bias=fx["bias"],
    ).eval()
    conv.load_state_dict(fx["state_dict"])
    return conv


def _fallback_warnings(records) -> list:
    """Any captured warning whose text mentions a torch fallback (the P150
    TTNNReshape host round-trip we are eliminating). Returns the offending
    messages so the assertion can name them."""
    return [str(w.message) for w in records if "fallback" in str(w.message).lower()]


@pytest.mark.parametrize("name", _CONVS)
def test_micro_conv(device, name):
    """Rebuild one SAM conv from cached weights, run on TTNN with the REAL cached
    input activation, assert PCC vs the cached fp32 output. Fast: no model load.

    Exercises the ON-DEVICE 2D path used by the SAM encoder: ``reshape_output=False``
    with H,W pre-recorded in ``model_config[<module>]["input_shapes"]`` so the conv
    returns the native flattened ``[1,1,H*W,C]`` on device WITHOUT the T3K-only
    ``TTNNReshape`` (which torch-falls-back on P150). We (a) assert PCC>=0.999 vs the
    cached fp32 ref (host-reshaping the flattened output only for the compare) and
    (b) assert NO torch fallback warning fired anywhere in the conv forward."""
    from tt_symbiote.modules.ttnn_conv import TTNNConv2dNHWC

    fpath = FIX / f"conv_{name}.pt"
    if not fpath.exists():
        pytest.skip(f"missing fixture {fpath.name}; run capture_reference_activations.py")
    fx = torch.load(fpath, weights_only=False)

    conv = _rebuild_conv(fx)
    x_nchw = fx["input_nchw"]
    b, c_in, h, w = x_nchw.shape
    ref_nhwc = fx["output_nchw"].permute(0, 2, 3, 1).contiguous()  # align to TTNN NHWC out

    tt = TTNNConv2dNHWC.from_torch(conv)
    set_device(tt, device)
    # Pre-record H,W so the conv can stay flattened even when fed a flattened
    # [1,1,H*W,C] input (as it will be when chained inside the SAM encoder), and
    # force the on-device 2D path (reshape_output=False -> no TTNNReshape).
    tt.set_model_config({tt.module_name: {"input_shapes": [[b, h, w, c_in]], "reshape_output": False}})
    x_nhwc = ttnn.from_torch(
        x_nchw.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tt.forward(x_nhwc)  # reshape_output picked up from model_config (False)
    fb = _fallback_warnings(caught)
    # Flattened native conv output [1,1,H*W,C]; host-reshape only for the PCC compare.
    got = ttnn.to_torch(out).reshape(ref_nhwc.shape)
    pcc, maxdiff = compute_pcc(got, ref_nhwc)[0]
    print(f"[micro conv {name}] PCC={pcc:.6f} maxdiff={maxdiff:.4g} "
          f"out_shape={tuple(out.shape)} in{tuple(x_nchw.shape)} "
          f"k{fx['kernel_size']} s{fx['stride']} p{fx['padding']} fallback={bool(fb)}")
    assert not fb, f"torch fallback fired in on-device 2D conv path for {name}: {fb}"
    assert_pcc(got, ref_nhwc, threshold=PCC, msg=f"micro conv {name}")
