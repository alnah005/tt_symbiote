# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Op-sweep over Qwen2DecoderLayer linears at rednote-hilab/dots.ocr dims.

Target: ``tests/capabilities/dots_ocr/test_decoder_dots_ocr.py::test_text_qwen2_decoder_layer_prefill``
Grid:   wide (full Cartesian product of weight_dtype x math_fidelity x
        fp32_dest_acc_en x packer_l1_acc x activation_dtype x memory_config).

For each grid point a fresh ``TTNNLinearSweep`` subclass is synthesized that
overrides ``preprocess_weights_impl()`` for weight dtype and ``forward()``
to inject the chosen ``WormholeComputeKernelConfig`` + output memory_config
into ``ttnn.linear``. The sweep subclass forward stays pure TTNN (no torch.*).

Every grid point's result (PCC, max_abs_diff, device-time per forward, status)
is appended to ``sweep_results/decoder_sweep.csv``. Tests do not fail on PCC
shortfall -- the CSV is the source of truth and a post-run analysis ranks the
top configs by time among those that cleared the PCC threshold.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import copy
import csv
import itertools
import json
import os
import pathlib
import time
import warnings

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from tt_symbiote.core.module import DeviceArch, run_on_devices
from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.modules.ttnn_activation import TTNNSilu
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.modules.ttnn_normalization import TTNNRMSNorm
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules

from tests.capabilities.pcc_utils import compute_pcc

_HERE = pathlib.Path(__file__).parent
_SHAPES = json.loads((_HERE / "shapes.json").read_text())
_CSV_PATH = _HERE / "sweep_results" / "decoder_sweep.csv"
_CSV_HEADER = [
    "cfg_id",
    "weight_dtype",
    "math_fidelity",
    "fp32_dest_acc_en",
    "packer_l1_acc",
    "activation_dtype",
    "memory_config",
    "pcc",
    "max_abs_diff",
    "forward_ms",
    "status",
    "error",
]

_PCC_THRESHOLD = 0.999
_SEQ_LEN = 128
_N_ITER = 3  # forward passes timed per config (after one warm-up)

_DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}
_FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
_MEM_CFGS = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def _build_grid():
    """Sweep grid. Wide = full 144-config Cartesian product.

    Set ``DOTS_OCR_SWEEP_MODE=quick`` to restrict to the 18-config core grid
    (3 weight dtypes x 3 fidelities x 2 fp32_acc; packer_l1=True,
    activation=bfloat16, memory=DRAM fixed) so the sweep runs in bounded time
    on hardware. Default is the wide grid.
    """
    quick = os.environ.get("DOTS_OCR_SWEEP_MODE", "wide").lower() == "quick"
    grid = list(
        itertools.product(
            ["bfloat16", "bfloat8_b", "bfloat4_b"],
            ["LoFi", "HiFi2", "HiFi4"],
            [True, False],
            [True] if quick else [True, False],
            ["bfloat16"] if quick else ["bfloat16", "bfloat8_b"],
            ["DRAM"] if quick else ["DRAM", "L1"],
        )
    )
    out = []
    for weight_dtype, fidelity, fp32_acc, packer_l1, act_dtype, mem_cfg in grid:
        cfg_id = (
            f"w{weight_dtype}-mf{fidelity}-fp32{int(fp32_acc)}"
            f"-pkr{int(packer_l1)}-act{act_dtype}-{mem_cfg}"
        )
        out.append(
            (
                cfg_id,
                {
                    "weight_dtype": weight_dtype,
                    "math_fidelity": fidelity,
                    "fp32_dest_acc_en": fp32_acc,
                    "packer_l1_acc": packer_l1,
                    "activation_dtype": act_dtype,
                    "memory_config": mem_cfg,
                },
            )
        )
    return out


_GRID = _build_grid()


# ---------------------------------------------------------------------------
# TTNNLinearSweep subclass factory
# ---------------------------------------------------------------------------


def _make_sweep_class(cfg):
    """Build a TTNNLinear subclass with the sweep config baked into its forward."""
    weight_dtype = _DTYPES[cfg["weight_dtype"]]
    activation_dtype = _DTYPES[cfg["activation_dtype"]]
    math_fidelity = _FIDELITIES[cfg["math_fidelity"]]
    fp32_acc = cfg["fp32_dest_acc_en"]
    packer_l1 = cfg["packer_l1_acc"]
    output_mem_cfg = _MEM_CFGS[cfg["memory_config"]]

    class TTNNLinearSweep(TTNNLinear):
        """Pure-TTNN linear with sweep parameters baked into ttnn.linear."""

        def preprocess_weights_impl(self):
            self.tt_weight_host = preprocess_linear_weight(
                self.weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
            )
            self.tt_bias_host = None
            if self.bias is not None:
                self.tt_bias_host = preprocess_linear_bias(
                    self.bias, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT
                )

        @run_on_devices(DeviceArch.T3K)
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
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_acc,
                packer_l1_acc=packer_l1,
            )
            tt_output = ttnn.linear(
                input_tensor,
                self.tt_weight,
                bias=self.tt_bias,
                compute_kernel_config=compute_kernel_config,
                dtype=activation_dtype,
                memory_config=output_mem_cfg,
            )
            tt_output = ttnn.reshape(
                tt_output, input_tensor_shape[:-1] + [self.out_features]
            )
            return tt_output

    return TTNNLinearSweep


# ---------------------------------------------------------------------------
# Shared torch reference (built once per module)
# ---------------------------------------------------------------------------


def _qwen2_config():
    t = _SHAPES["config"]
    return Qwen2Config(
        vocab_size=t["vocab_size"],
        hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"],
        num_hidden_layers=1,
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"],
        max_position_embeddings=t["max_position_embeddings"],
        rms_norm_eps=t["rms_norm_eps"],
        rope_parameters={"rope_theta": t["rope_theta"], "rope_type": "default"},
        attention_dropout=0.0,
        attention_bias=True,
        hidden_act=t["hidden_act"],
        use_cache=False,
        tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def torch_reference():
    torch.manual_seed(42)
    cfg = _qwen2_config()
    layer = modeling_qwen2.Qwen2DecoderLayer(cfg, layer_idx=0).to(torch.bfloat16)
    rotary = modeling_qwen2.Qwen2RotaryEmbedding(cfg).to(torch.bfloat16)
    layer.eval()
    rotary.eval()
    torch.set_grad_enabled(False)
    hidden = torch.randn(1, _SEQ_LEN, cfg.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(hidden, position_ids)
    torch_out = layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
    return {
        "layer": layer,
        "hidden": hidden,
        "cos": cos,
        "sin": sin,
        "torch_out": torch_out,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _ensure_csv():
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _CSV_PATH.exists():
        with _CSV_PATH.open("w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def _append_csv_row(cfg_id, cfg, pcc, max_diff, forward_ms, status, error):
    _ensure_csv()
    with _CSV_PATH.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                cfg_id,
                cfg["weight_dtype"],
                cfg["math_fidelity"],
                int(cfg["fp32_dest_acc_en"]),
                int(cfg["packer_l1_acc"]),
                cfg["activation_dtype"],
                cfg["memory_config"],
                f"{pcc:.6f}" if isinstance(pcc, float) and pcc == pcc else "nan",
                f"{max_diff:.6f}" if isinstance(max_diff, float) and max_diff == max_diff else "nan",
                f"{forward_ms:.3f}" if isinstance(forward_ms, float) and forward_ms == forward_ms else "nan",
                status,
                error.replace("\n", " ")[:200] if error else "",
            ]
        )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_id,cfg", _GRID, ids=[c[0] for c in _GRID])
def test_sweep_qwen2_decoder_layer(mesh_device, torch_reference, cfg_id, cfg):
    """Run one grid point and append its metrics to decoder_sweep.csv."""
    pcc = float("nan")
    max_diff = float("nan")
    forward_ms = float("nan")
    status = "ok"
    err = ""

    try:
        layer = copy.deepcopy(torch_reference["layer"])
        SweepCls = _make_sweep_class(cfg)
        swap_map = {
            nn.Linear: SweepCls,
            nn.SiLU: TTNNSilu,
            modeling_qwen2.Qwen2RMSNorm: TTNNRMSNorm,
        }
        modules = register_modules(layer, swap_map, model_config=None)
        set_device(layer, mesh_device)
        for _, mod in modules.items():
            mod.preprocess_weights()
            mod.move_weights_to_device()

        hidden = TorchTTNNTensor(torch_reference["hidden"].clone())
        pos_emb = (torch_reference["cos"], torch_reference["sin"])

        # Warm-up
        _ = layer(hidden, position_embeddings=pos_emb, attention_mask=None)

        # Timed forwards
        t0 = time.perf_counter()
        for _ in range(_N_ITER):
            ttnn_out = layer(hidden, position_embeddings=pos_emb, attention_mask=None)
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(mesh_device)
        forward_ms = ((time.perf_counter() - t0) / _N_ITER) * 1000.0

        results = compute_pcc(ttnn_out, torch_reference["torch_out"])
        pcc, max_diff = results[0]
        if pcc < _PCC_THRESHOLD:
            status = "pcc_below_threshold"
        else:
            status = "ok"
    except Exception as exc:  # noqa: BLE001 -- sweeps must capture every failure mode
        msg = str(exc)
        err = msg
        if "out of memory" in msg.lower() or "L1" in msg and "alloc" in msg.lower():
            status = "oom"
        elif "not implemented" in msg.lower() or "unsupported" in msg.lower():
            status = "unsupported"
        else:
            status = "error"
        warnings.warn(f"sweep[{cfg_id}] {status}: {msg[:120]}", stacklevel=1)
    finally:
        _append_csv_row(cfg_id, cfg, pcc, max_diff, forward_ms, status, err)


# ---------------------------------------------------------------------------
# Post-sweep analysis (separate "test" so it always runs last)
# ---------------------------------------------------------------------------


def test_zzz_sweep_summary():
    """Aggregate decoder_sweep.csv and write decoder_best.json with the top configs.

    This runs after all sweep grid points (alphabetical test ordering) so the
    CSV is fully populated. It does not contact hardware.
    """
    if not _CSV_PATH.exists():
        pytest.skip("decoder_sweep.csv not produced (sweep did not run)")

    rows = []
    with _CSV_PATH.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["pcc"] = float(row["pcc"])
            except ValueError:
                row["pcc"] = float("nan")
            try:
                row["forward_ms"] = float(row["forward_ms"])
            except ValueError:
                row["forward_ms"] = float("nan")
            rows.append(row)

    passing = [
        r for r in rows
        if r["status"] == "ok" and r["pcc"] == r["pcc"] and r["pcc"] >= _PCC_THRESHOLD
    ]
    passing.sort(key=lambda r: r["forward_ms"])

    summary = {
        "csv_path": str(_CSV_PATH),
        "pcc_threshold": _PCC_THRESHOLD,
        "n_configs": len(rows),
        "n_passing": len(passing),
        "top5": [
            {
                "cfg_id": r["cfg_id"],
                "weight_dtype": r["weight_dtype"],
                "math_fidelity": r["math_fidelity"],
                "fp32_dest_acc_en": r["fp32_dest_acc_en"],
                "packer_l1_acc": r["packer_l1_acc"],
                "activation_dtype": r["activation_dtype"],
                "memory_config": r["memory_config"],
                "pcc": r["pcc"],
                "forward_ms": r["forward_ms"],
            }
            for r in passing[:5]
        ],
    }

    out_path = _HERE / "sweep_results" / "decoder_best.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSweep summary written to {out_path}")
    print(f"  configs run:   {summary['n_configs']}")
    print(f"  configs pass:  {summary['n_passing']} (PCC >= {_PCC_THRESHOLD})")
    if summary["top5"]:
        print("  fastest passing configs:")
        for entry in summary["top5"]:
            print(
                f"    {entry['cfg_id']:60s}  "
                f"pcc={entry['pcc']:.5f}  ms={entry['forward_ms']:.2f}"
            )
