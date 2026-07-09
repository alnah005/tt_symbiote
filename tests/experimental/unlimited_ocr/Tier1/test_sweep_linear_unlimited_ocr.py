# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""OP-SWEEP: weight-dtype x math-fidelity x fp32-dest-acc for the unlimited_ocr
bottleneck LINEARS (compute-bound vision prefill + LM projections).

WHAT / WHY
    The unlimited_ocr prefill is compute-bound on the SAM + CLIP vision towers'
    matmuls (~10.7s device compute). Every linear currently runs
    ``TTNNUnlimitedOcrLinear`` == bf16 weights + HiFi4 + fp32_dest_acc_en=True
    (see modeling_unlimited_ocr._lm_compute_kernel_config). This sweep measures,
    per representative bottleneck linear, whether a cheaper weight dtype
    (bfloat8_b / bfloat4_b) and/or lower math fidelity (HiFi2 / LoFi) and/or
    fp32_dest_acc=False CUTS device compute while HOLDING per-op PCC.

    NOTE: this sweep only MEASURES (op-sweep stage). It does NOT change the
    model's committed configs (that is the later config-optimize stage).

PCC (fast, no 3.3B reload)
    Uses cached per-op reference fixtures dumped by
    ``fixtures/capture_reference_activations.py`` (real input activation + real
    fp32 output + weights of each linear, captured inside the real model forward).
    Each grid point rebuilds ONE nn.Linear, runs it on TTNN with the config, and
    computes PCC vs the cached fp32 output. Written to
    ``sweep_results/linear_pcc.csv``.

DEVICE TIME (tracy-only, Req 4)
    Per-grid-point device time comes SOLELY from the tracy
    ``ops_perf_results_*.csv`` DEVICE KERNEL/FW DURATION columns. A tracy
    signpost ``SWEEP_START|<op>|<dtype>|<fid>|<fp32>`` delimits each config's
    MEASURED matmul iterations so the parser
    (``sweep_results/parse_tracy_devtime.py``) can map each MatmulDeviceOperation
    row back to its grid point. Run under tracy via:

        export TT_METAL_HOME=/home/ttuser/salnahari/tt-metal
        export TT_SYMBIOTE_SIGNPOST_MODE=1
        python -m tracy -p -r -v --op-support-count 20000 \
          -m 'pytest tests/experimental/unlimited_ocr/Tier1/test_sweep_linear_unlimited_ocr.py -x -s'
        tt-perf-report --ignore-signposts */ops_perf_results_*.csv > /dev/null  # optional summary

DERIVED GRID (Req 7 -- derived at sweep time, NOT invented)
    tt-metal commit: a0b506c780979538b6d2fc1e57fdbfdfdabc7e31
    Grepping $TT_METAL_HOME for the dtypes / fidelities / fp32 settings actually
    used on linear/matmul weights in the vision towers + DiT + CNN + transformers
    references yielded (counts in parens):
      weight dtype  : ttnn.bfloat16, ttnn.bfloat8_b (429), ttnn.bfloat4_b (used)
      math fidelity : MathFidelity.LoFi, HiFi2 (26), HiFi4 (31)
      fp32_dest_acc : True (45), False (9)
    -> cartesian grid = 3 x 3 x 2 = 18 points/op (quick mode).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from tt_symbiote.core.module import DeviceArch, run_on_devices
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.utils.device_management import set_device
from tests.shared.pcc_utils import compute_pcc

# --------------------------------------------------------------------------- #
# Paths + derived grid (recorded to sweep_results for provenance).
# --------------------------------------------------------------------------- #
TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"
FIX = Path(__file__).parent.parent / "fixtures"
SWEEP_DIR = Path(__file__).parent.parent / "sweep_results"
PCC_CSV = SWEEP_DIR / "linear_pcc.csv"

# Representative bottleneck linears (fixture stem -> human label / tower).
OPS = ["clip_qkv", "clip_fc1", "sam_qkv", "sam_lin1", "projector", "lm_gate", "lm_qproj"]

WEIGHT_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
FP32_ACC = [False, True]

# Full cartesian grid (18 points/op). Baseline (current committed config) is
# bfloat16 / HiFi4 / fp32_dest_acc=True.
GRID = [
    (wd, fid, acc)
    for wd in WEIGHT_DTYPES
    for fid in FIDELITIES
    for acc in FP32_ACC
]

_MEASURED_ITERS = 3  # measured matmul iterations per config (tracy takes the min)

pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)


# --------------------------------------------------------------------------- #
# Parameterized sweep linear: weight dtype baked at preprocess; compute kernel
# config (fidelity + fp32_dest_acc) applied per forward. Pure-ttnn forward.
# --------------------------------------------------------------------------- #
class TTNNLinearSweep(TTNNLinear):
    """TTNNLinear whose weight dtype and compute-kernel-config are sweepable.

    ``_sweep_weight_dtype`` is baked into the device weight at preprocess time;
    ``_sweep_ck`` (a WormholeComputeKernelConfig) is read per forward call, so a
    single built instance can be re-timed across all fidelity/fp32 points that
    share its weight dtype. Bias is always bf16 (tiny; bfloat4_b bias is
    unsupported) -- matches how ttnn adds bias in the bf16 output.
    """

    _sweep_weight_dtype = ttnn.bfloat16
    _sweep_ck = None

    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=self._sweep_weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

    @run_on_devices(DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)
    def forward(self, input_tensor):
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(
                input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._sweep_ck,
        )
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


def _signpost(text: str) -> None:
    """Emit a tracy signpost (no-op if unavailable / not profiling)."""
    try:
        from tracy import signpost

        signpost(text)
    except Exception:  # noqa: BLE001
        pass


def _load_fixture(op: str):
    fpath = FIX / f"linear_{op}.pt"
    if not fpath.exists():
        pytest.skip(f"missing fixture {fpath.name}; run capture_reference_activations.py")
    return torch.load(fpath, weights_only=False)


def _rebuild_linear(fx) -> nn.Linear:
    lin = nn.Linear(fx["in_features"], fx["out_features"], bias=fx["bias"]).eval()
    lin.load_state_dict(fx["state_dict"])
    return lin


# Module-scoped cache for the (CPU-side) fixtures ONLY. Device tensors are NEVER
# cached across grid points -- each grid point builds its own device weight/input/
# output and deallocates them before the next point (leak fix: the prior version
# cached up to 21 on-device weight sets in _LIN_CACHE and never freed per-config
# tensors, OOM-ing after the first block of configs).
_FX_CACHE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _init_csv():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(PCC_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["op", "weight_dtype", "math_fidelity", "fp32_dest_acc", "pcc", "max_diff", "status"]
        )
    # Provenance: record the derived grid + commit alongside the results.
    with open(SWEEP_DIR / "derived_grid.json", "w") as f:
        json.dump(
            {
                "tt_metal_commit": TT_METAL_COMMIT,
                "device_arch": "P150 (Blackhole, single (1,1) mesh)",
                "ops": OPS,
                "weight_dtypes": list(WEIGHT_DTYPES),
                "math_fidelities": list(FIDELITIES),
                "fp32_dest_acc": FP32_ACC,
                "grid_points_per_op": len(GRID),
                "baseline_current_config": {
                    "weight_dtype": "bfloat16",
                    "math_fidelity": "HiFi4",
                    "fp32_dest_acc": True,
                },
                "packer_l1_acc": True,
                "measured_iters_per_config": _MEASURED_ITERS,
            },
            f,
            indent=2,
        )
    yield


def _build_linear(op, wd_name, device):
    """Build a FRESH device-resident TTNNLinearSweep for one grid point.

    Nothing here is cached on device: the caller is responsible for
    ``tt.deallocate_weights()`` after the grid point so the device weight/bias
    free before the next config (leak fix)."""
    if op not in _FX_CACHE:
        _FX_CACHE[op] = _load_fixture(op)
    fx = _FX_CACHE[op]
    lin = _rebuild_linear(fx)
    tt = TTNNLinearSweep.from_torch(lin)
    tt._sweep_weight_dtype = WEIGHT_DTYPES[wd_name]
    set_device(tt, device)
    return tt, fx


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("weight_dtype,math_fidelity,fp32_acc", GRID)
def test_sweep_linear(device, op, weight_dtype, math_fidelity, fp32_acc):
    """One grid point: build a FRESH linear (weight_dtype), run it on the cached
    real activation with (math_fidelity, fp32_dest_acc), record PCC, and emit
    tracy signposts around MEASURED matmul iterations for device-time extraction.

    Leak fix: EVERY device tensor created for this config (weight, bias, input
    activation, output) is deallocated in the ``finally`` block before the next
    grid point runs, and the linear reference is dropped. No device tensor is
    cached across grid points."""
    tag = f"{op}|{weight_dtype}|{math_fidelity}|{fp32_acc}"

    status = "ok"
    pcc = float("nan")
    max_diff = float("nan")
    tt = None
    x = None
    out = None
    try:
        tt, fx = _build_linear(op, weight_dtype, device)
        tt._sweep_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=FIDELITIES[math_fidelity],
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=True,
        )
        x_torch = fx["input"]
        ref = fx["output"]

        x = ttnn.from_torch(
            x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # Warm-up (compiles the program) -- excluded from the measured signpost region.
        out = tt.forward(x)
        got = ttnn.to_torch(out).reshape(ref.shape)
        pcc, max_diff = compute_pcc(got, ref)[0]
        ttnn.deallocate(out)
        out = None
        ttnn.synchronize_device(device)

        # Measured region: tracy maps MatmulDeviceOperation rows between these
        # signposts to this grid point (min DEVICE KERNEL/FW DURATION is used).
        _signpost(f"SWEEP_START|{tag}")
        for _ in range(_MEASURED_ITERS):
            out = tt.forward(x)
            ttnn.deallocate(out)
            out = None
        ttnn.synchronize_device(device)
        _signpost(f"SWEEP_END|{tag}")
    except Exception as exc:  # noqa: BLE001 -- OOM / unsupported combo: record + move on
        status = f"error:{type(exc).__name__}"
        print(f"[sweep {tag}] {status}: {exc}")
    finally:
        # Deallocate EVERY device tensor built for this grid point before the next
        # config runs (the leak fix). Guard each dealloc so a mid-config failure
        # still frees whatever did get allocated.
        if out is not None:
            try:
                ttnn.deallocate(out)
            except Exception:  # noqa: BLE001
                pass
        if x is not None:
            try:
                ttnn.deallocate(x)
            except Exception:  # noqa: BLE001
                pass
        if tt is not None:
            try:
                tt.deallocate_weights()  # frees tt_weight + tt_bias on device
            except Exception:  # noqa: BLE001
                pass
        del tt, x, out
        try:
            ttnn.synchronize_device(device)
        except Exception:  # noqa: BLE001
            pass

    with open(PCC_CSV, "a", newline="") as f:
        csv.writer(f).writerow(
            [op, weight_dtype, math_fidelity, fp32_acc,
             f"{pcc:.6f}", f"{max_diff:.6g}", status]
        )
    print(f"[sweep {tag}] PCC={pcc:.6f} maxdiff={max_diff:.4g} status={status}")
