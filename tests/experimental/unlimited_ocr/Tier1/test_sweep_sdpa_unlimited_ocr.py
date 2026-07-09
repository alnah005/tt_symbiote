# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""OP-SWEEP: flash-SDPA (``ttnn.transformer.scaled_dot_product_attention``) for the
four unlimited_ocr attention call sites (CLIP, SAM window, SAM global, LM prefill).

WHAT / WHY
    SDPA has NO weight dtype -> the sweep varies the Q/K/V INPUT dtype +
    math_fidelity + fp32_dest_acc + (program-config) q/k chunk size + exp_approx.
    Two known findings to CONFIRM (Blackhole P150, tt-metal
    a0b506c780979538b6d2fc1e57fdbfdfdabc7e31):
      * fp32_dest_acc_en=True DEGRADES the flash-SDPA kernel (vision) -> expect a
        PCC DROP when fp32=True (the fp32-dest kernel bug).
      * exp_approx_mode=False + matched q/k chunk sizes helps the SAM global SDPA.

    MEASURE-ONLY: does not change the model's committed configs.

PCC (fast, no 3.3B reload)
    Cached per-site fixtures (real q,k,v[,attn_mask] + torch-fp32 SDPA output)
    from ``fixtures/capture_reference_activations.py``. Each grid point rebuilds
    the SDPA op on device with the config, computes PCC vs the cached fp32 output.
    Written to ``sweep_results/sdpa_pcc.csv``.

DEVICE TIME (tracy-only, Req 4)
    Per-grid-point device time comes SOLELY from the tracy ops_perf_results_*.csv
    DEVICE KERNEL DURATION. Signposts ``SWEEP_START|<tag>`` / ``SWEEP_END|<tag>``
    delimit each config's MEASURED SDPA iterations for
    ``sweep_results/parse_tracy_sdpa.py``.

DERIVED GRID (Req 7 -- derived at sweep time)
    Grepping $TT_METAL_HOME (models/tt_transformers/tt/attention.py,
    models/tt_dit/, models/demos/, ttnn/) for SDPA compute/program config usage:
      input dtype   : ttnn.bfloat16, ttnn.bfloat8_b
      math fidelity : MathFidelity.LoFi, HiFi2, HiFi4
      fp32_dest_acc : True, False
      program cfg   : SDPAProgramConfig(q_chunk_size,k_chunk_size in {128,256,512},
                      exp_approx_mode in {True,False})
    Part A (compute-kernel config) sweeps dtype x fidelity x fp32 (program_config
    left at ttnn default). Part B (program config) sweeps chunk x exp_approx at the
    baseline bf16/HiFi4/fp32=False (chunk+exp are numerically orthogonal to
    weightless-SDPA dtype/fidelity; separating them keeps the grid bounded).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import ttnn

from tests.shared.pcc_utils import compute_pcc

TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"
FIX = Path(__file__).parent.parent / "fixtures"
SWEEP_DIR = Path(__file__).parent.parent / "sweep_results"
PCC_CSV = SWEEP_DIR / "sdpa_pcc.csv"

# site -> (default program-config chunk for Part A left as ttnn default; Part-B
# explicit chunk candidates -- multiples of 32, <= seq, equal q=k).
SITES = {
    "sdpa_clip":       {"prog_chunks": [128, 256]},          # seq 257
    "sdpa_sam_window": {"prog_chunks": [64, 128]},           # seq 196
    "sdpa_sam_global": {"prog_chunks": [128, 256, 512]},     # seq 4096
    "sdpa_lm_prefill": {"prog_chunks": [32]},                # seq 8
}

INPUT_DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
FP32_ACC = [False, True]

_MEASURED_ITERS = 3

pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)

_FX_CACHE: dict = {}


def _build_grid():
    """(op, input_dtype, math_fidelity, fp32_acc, exp_approx, chunk) tuples.

    exp_approx / chunk == "default" means NO explicit SDPAProgramConfig (ttnn
    default). Part A: dtype x fid x fp32 at default program-config. Part B:
    explicit chunk x exp_approx at bf16/HiFi4/fp32=False.
    """
    grid = []
    for op in SITES:
        # Part A -- compute kernel config, ttnn-default program config.
        for dt in INPUT_DTYPES:
            for fid in FIDELITIES:
                for acc in FP32_ACC:
                    grid.append((op, dt, fid, acc, "default", "default"))
        # Part B -- program config (chunk x exp_approx) at bf16/HiFi4/fp32=False.
        for c in SITES[op]["prog_chunks"]:
            for exp in (True, False):
                grid.append((op, "bfloat16", "HiFi4", False, exp, c))
    return grid


GRID = _build_grid()


def _signpost(text: str) -> None:
    try:
        from tracy import signpost

        signpost(text)
    except Exception:  # noqa: BLE001
        pass


def _load_fixture(op: str):
    fpath = FIX / f"{op}.pt"
    if not fpath.exists():
        pytest.skip(f"missing fixture {fpath.name}; run capture_reference_activations.py")
    return torch.load(fpath, weights_only=False)


@pytest.fixture(scope="module", autouse=True)
def _init_csv():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(PCC_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["op", "input_dtype", "math_fidelity", "fp32_dest_acc",
             "exp_approx", "chunk_size", "pcc", "max_diff", "status"]
        )
    with open(SWEEP_DIR / "derived_grid_sdpa.json", "w") as f:
        json.dump(
            {
                "tt_metal_commit": TT_METAL_COMMIT,
                "device_arch": "P150 (Blackhole, single (1,1) mesh)",
                "sites": {k: v for k, v in SITES.items()},
                "input_dtypes": list(INPUT_DTYPES),
                "math_fidelities": list(FIDELITIES),
                "fp32_dest_acc": FP32_ACC,
                "exp_approx_mode": [True, False],
                "chunk_sizes_partB": "per-site prog_chunks (mult of 32, <= seq, q==k)",
                "partA": "dtype x fidelity x fp32 @ ttnn-default program_config",
                "partB": "chunk x exp_approx @ bf16/HiFi4/fp32=False",
                "measured_iters_per_config": _MEASURED_ITERS,
                "grid_points_total": len(GRID),
            },
            f,
            indent=2,
        )
    yield


@pytest.mark.parametrize("op,input_dtype,math_fidelity,fp32_acc,exp_approx,chunk", GRID)
def test_sweep_sdpa(device, op, input_dtype, math_fidelity, fp32_acc, exp_approx, chunk):
    tag = f"{op}|{input_dtype}|{math_fidelity}|{fp32_acc}|{exp_approx}|{chunk}"

    status = "ok"
    pcc = float("nan")
    max_diff = float("nan")
    q = k = v = mask = out = None
    try:
        if op not in _FX_CACHE:
            _FX_CACHE[op] = _load_fixture(op)
        fx = _FX_CACHE[op]
        ref = fx["output"]
        dt = INPUT_DTYPES[input_dtype]

        q = ttnn.from_torch(fx["q"], dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        k = ttnn.from_torch(fx["k"], dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        v = ttnn.from_torch(fx["v"], dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        if fx["attn_mask"] is not None:
            mask = ttnn.from_torch(
                fx["attn_mask"].to(torch.float32), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=device,
            )

        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=FIDELITIES[math_fidelity],
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=True,
        )
        prog = None
        if chunk != "default":
            grid = device.compute_with_storage_grid_size()
            prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=int(chunk),
                k_chunk_size=int(chunk),
                exp_approx_mode=bool(exp_approx),
            )

        def _run():
            return ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=bool(fx["is_causal"]),
                scale=float(fx["scale"]), compute_kernel_config=ck,
                program_config=prog,
            )

        out = _run()
        got = ttnn.to_torch(out).reshape(ref.shape)
        pcc, max_diff = compute_pcc(got, ref)[0]
        ttnn.deallocate(out)
        out = None
        ttnn.synchronize_device(device)

        _signpost(f"SWEEP_START|{tag}")
        for _ in range(_MEASURED_ITERS):
            out = _run()
            ttnn.deallocate(out)
            out = None
        ttnn.synchronize_device(device)
        _signpost(f"SWEEP_END|{tag}")
    except Exception as exc:  # noqa: BLE001
        status = f"error:{type(exc).__name__}"
        print(f"[sdpa {tag}] {status}: {exc}")
    finally:
        for t in (out, q, k, v, mask):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:  # noqa: BLE001
                    pass
        del q, k, v, mask, out
        try:
            ttnn.synchronize_device(device)
        except Exception:  # noqa: BLE001
            pass

    with open(PCC_CSV, "a", newline="") as f:
        csv.writer(f).writerow(
            [op, input_dtype, math_fidelity, fp32_acc, exp_approx, chunk,
             f"{pcc:.6f}", f"{max_diff:.6g}", status]
        )
    print(f"[sdpa {tag}] PCC={pcc:.6f} maxdiff={max_diff:.4g} status={status}")
