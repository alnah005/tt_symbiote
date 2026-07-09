# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""OP-SWEEP: the remaining unlimited_ocr MATMULS not covered by the bottleneck-7
linear sweep -- lm_head, the SAM rel-pos bmms, and the MoE router gate matmul.

WHAT / WHY
    Same grid family as the linears (weight dtype x math_fidelity x fp32_dest_acc)
    plus a BOUNDED output-memory-config second pass (DRAM vs L1) on the big
    matmuls (lm_head, clip_fc1). Questions to answer:
      * lm_head (1280->129280, kept bf16 in the model): is bf8 safe (argmax /
        logit PCC) or does it flip the token?
      * SAM rel-pos matmuls (rq_h @ RhT, rq_w @ RwT; static-weight bmm): best dtype.
      * MoE router gate (x @ gate.T; currently fp32, accuracy-critical routing):
        is fp32 necessary vs bf16/bf8?

    MEASURE-ONLY: does not change committed configs.

PCC: cached fixtures (real I/O + torch-fp32 reference), per-config device dealloc.
DEVICE TIME: tracy-only (signposts -> sweep_results/parse_tracy_matmul_extra.py).

PROGRAM CONFIG (honest scope note)
    Explicit MatmulProgramConfig / MatmulMultiCoreReuse* configs were NOT
    exhaustively swept per shape (not tractable to auto-generate a valid program
    config per arbitrary matmul shape). This sweep uses the ttnn AUTO program
    config for every matmul and sweeps output MEMORY CONFIG (DRAM vs L1) on the
    big matmuls as the program-config-adjacent axis, per the bounded-scope
    directive. DRAM-sharded output was not swept (recorded as not-attempted).

DERIVED GRID (Req 7): grepped $TT_METAL_HOME as for the linear sweep:
      weight dtype  : ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b (+ float32 for
                      the MoE gate, whose committed weight is fp32)
      math fidelity : MathFidelity.LoFi, HiFi2, HiFi4
      fp32_dest_acc : True, False
      output memcfg : ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG (big matmuls)
    tt-metal commit: a0b506c780979538b6d2fc1e57fdbfdfdabc7e31
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_linear_weight

from tt_symbiote.core.module import DeviceArch, run_on_devices
from tests.shared.pcc_utils import compute_pcc
from tests.experimental.unlimited_ocr.Tier1.test_sweep_linear_unlimited_ocr import (
    TTNNLinearSweep,
    _rebuild_linear,
)

TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"
FIX = Path(__file__).parent.parent / "fixtures"
SWEEP_DIR = Path(__file__).parent.parent / "sweep_results"
PCC_CSV = SWEEP_DIR / "matmul_extra_pcc.csv"

WEIGHT_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
MEMCFGS = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}
FP32_ACC = [False, True]
_MEASURED_ITERS = 3

# Linear-like matmul ops (fixture stem -> dtype set). lm_head/moe_gate rebuild an
# nn.Linear from the fixture and reuse TTNNLinearSweep.
LINEAR_OPS = {
    "lm_head": ["bfloat16", "bfloat8_b", "bfloat4_b"],
    "moe_gate": ["float32", "bfloat16", "bfloat8_b"],
}
# Batched-matmul (bmm) rel-pos ops.
RELPOS_OPS = {
    "relpos_h_global": ["bfloat16", "bfloat8_b", "bfloat4_b"],
    "relpos_w_global": ["bfloat16", "bfloat8_b", "bfloat4_b"],
    "relpos_h_window": ["bfloat16", "bfloat8_b", "bfloat4_b"],
    "relpos_w_window": ["bfloat16", "bfloat8_b", "bfloat4_b"],
}
# Output-memory-config second pass (big matmuls). clip_fc1 dtype/fid PCC already in
# linear_pcc.csv; here we add the memcfg axis at a representative winner config.
MEMCFG_OPS = {
    "lm_head": {"kind": "linear", "dtypes": ["bfloat16", "bfloat8_b"]},
    "clip_fc1": {"kind": "linear", "dtypes": ["bfloat16", "bfloat8_b"]},
}

pytestmark = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768}], indirect=True
)

_FX_CACHE: dict = {}


class TTNNLinearSweepMem(TTNNLinearSweep):
    """TTNNLinearSweep with a sweepable OUTPUT memory config."""

    _sweep_out_memcfg = ttnn.DRAM_MEMORY_CONFIG

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
            memory_config=self._sweep_out_memcfg,
            compute_kernel_config=self._sweep_ck,
        )
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


def _signpost(text: str) -> None:
    try:
        from tracy import signpost

        signpost(text)
    except Exception:  # noqa: BLE001
        pass


def _fx(op):
    if op not in _FX_CACHE:
        # lm_head / clip_fc1 fixtures were dumped with the "linear_" prefix by the
        # capture script's linear-WANTED hook; relpos/moe_gate use the bare stem.
        p = FIX / f"{op}.pt"
        if not p.exists():
            p = FIX / f"linear_{op}.pt"
        if not p.exists():
            pytest.skip(f"missing fixture {op}.pt / linear_{op}.pt")
        _FX_CACHE[op] = torch.load(p, weights_only=False)
    return _FX_CACHE[op]


def _build_grid():
    grid = []
    # Linear-like: dtype x fid x fp32 at DRAM.
    for op, dts in LINEAR_OPS.items():
        for dt in dts:
            for fid in FIDELITIES:
                for acc in FP32_ACC:
                    grid.append(("linear", op, dt, fid, acc, "DRAM"))
    # Rel-pos bmm: dtype x fid x fp32 at DRAM.
    for op, dts in RELPOS_OPS.items():
        for dt in dts:
            for fid in FIDELITIES:
                for acc in FP32_ACC:
                    grid.append(("relpos", op, dt, fid, acc, "DRAM"))
    # Memcfg second pass: {DRAM,L1} x dtype at HiFi2/fp32=False (L1 only; DRAM
    # already covered above for lm_head, and for clip_fc1 in linear_pcc.csv).
    for op, spec in MEMCFG_OPS.items():
        for dt in spec["dtypes"]:
            grid.append((spec["kind"], op, dt, "HiFi2", False, "L1"))
    return grid


GRID = _build_grid()


@pytest.fixture(scope="module", autouse=True)
def _init_csv():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(PCC_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["op", "kind", "weight_dtype", "math_fidelity", "fp32_dest_acc",
             "out_memcfg", "program_config", "pcc", "max_diff", "status"]
        )
    with open(SWEEP_DIR / "derived_grid_matmul_extra.json", "w") as f:
        json.dump(
            {
                "tt_metal_commit": TT_METAL_COMMIT,
                "device_arch": "P150 (Blackhole, single (1,1) mesh)",
                "linear_ops": LINEAR_OPS,
                "relpos_ops": RELPOS_OPS,
                "memcfg_ops": MEMCFG_OPS,
                "math_fidelities": list(FIDELITIES),
                "fp32_dest_acc": FP32_ACC,
                "out_memcfgs": list(MEMCFGS),
                "program_config": "auto (explicit MatmulProgramConfig NOT swept)",
                "measured_iters_per_config": _MEASURED_ITERS,
                "grid_points_total": len(GRID),
            },
            f,
            indent=2,
        )
    yield


def _run_linear(op, dt, fid, acc, memcfg_name, device):
    fx = _fx(op)
    if "state_dict" in fx:
        lin = _rebuild_linear(fx)
        x_torch, ref = fx["input"], fx["output"]
    else:  # moe_gate: build a bias-free Linear(hidden, n_experts) from weight.
        w = fx["weight"]  # [n_experts, hidden]
        lin = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False).eval()
        with torch.no_grad():
            lin.weight.copy_(w)
        x_torch, ref = fx["input"], fx["output"]
    cls = TTNNLinearSweepMem if memcfg_name != "DRAM" else TTNNLinearSweep
    from tt_symbiote.utils.device_management import set_device

    tt = cls.from_torch(lin)
    tt._sweep_weight_dtype = WEIGHT_DTYPES[dt]
    if cls is TTNNLinearSweepMem:
        tt._sweep_out_memcfg = MEMCFGS[memcfg_name]
    set_device(tt, device)
    tt._sweep_ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=FIDELITIES[fid], math_approx_mode=False,
        fp32_dest_acc_en=acc, packer_l1_acc=True,
    )
    # input dtype: fp32 for the moe_gate float32 case (fp32 router), else bf16.
    in_dt = ttnn.float32 if dt == "float32" else ttnn.bfloat16
    x = ttnn.from_torch(x_torch, dtype=in_dt, layout=ttnn.TILE_LAYOUT, device=device)
    return tt, x, ref


def _run_relpos(op, dt, fid, acc, device):
    fx = _fx(op)
    ref = fx["output"]
    inp = ttnn.from_torch(fx["input"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(fx["weight"], dtype=WEIGHT_DTYPES[dt], layout=ttnn.TILE_LAYOUT, device=device)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=FIDELITIES[fid], math_approx_mode=False,
        fp32_dest_acc_en=acc, packer_l1_acc=True,
    )
    return inp, w, ck, ref


@pytest.mark.parametrize("kind,op,weight_dtype,math_fidelity,fp32_acc,out_memcfg", GRID)
def test_sweep_matmul_extra(device, kind, op, weight_dtype, math_fidelity, fp32_acc, out_memcfg):
    tag = f"{op}|{weight_dtype}|{math_fidelity}|{fp32_acc}|{out_memcfg}"
    status = "ok"
    pcc = float("nan")
    max_diff = float("nan")
    tt = x = w = inp = out = None
    try:
        if kind == "linear":
            tt, x, ref = _run_linear(op, weight_dtype, math_fidelity, fp32_acc, out_memcfg, device)

            def _run():
                return tt.forward(x)
        else:
            inp, w, ck, ref = _run_relpos(op, weight_dtype, math_fidelity, fp32_acc, device)

            def _run():
                return ttnn.matmul(inp, w, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   compute_kernel_config=ck)

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
        print(f"[matmul_extra {tag}] {status}: {exc}")
    finally:
        for t in (out, x, w, inp):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:  # noqa: BLE001
                    pass
        if tt is not None:
            try:
                tt.deallocate_weights()
            except Exception:  # noqa: BLE001
                pass
        del tt, x, w, inp, out
        try:
            ttnn.synchronize_device(device)
        except Exception:  # noqa: BLE001
            pass

    with open(PCC_CSV, "a", newline="") as f:
        csv.writer(f).writerow(
            [op, kind, weight_dtype, math_fidelity, fp32_acc, out_memcfg, "auto",
             f"{pcc:.6f}", f"{max_diff:.6g}", status]
        )
    print(f"[matmul_extra {tag}] PCC={pcc:.6f} maxdiff={max_diff:.4g} status={status}")
