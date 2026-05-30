---
name: op-sweep
description: Sweep TTNN op configurations (weight dtype, math fidelity, memory layout) across a parameter grid for a specific op or module. Generates sweep test files with TTNNLinear subclasses that override preprocess_weights_impl() and forward(), runs them, collects PCC and timing data.
---

# Op Parameter Sweep

Systematically explore the configuration space for TTNN ops to find optimal settings.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/capabilities/<model_name>/test_modeling_<model_name>.py`

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.
  - `tests/models/` does NOT exist. Never create files there.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.
  - Import: `from tt_symbiote.core.module import run_on_devices, DeviceArch`
  - Default: `@run_on_devices(DeviceArch.T3K)`

**License headers**: Every generated `.py` file must start with:
  ```python
  # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
  # SPDX-License-Identifier: Apache-2.0
  ```

**PCC assertions**: Use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.
  NEVER rely on `compare_fn_outputs()`.

**Deprecated API**: Never use `register_module_replacement_dict()`.

**Config system**: The typed config system does NOT exist yet. Weight dtype is controlled
  by selecting different TTNNLinear subclasses that override `preprocess_weights_impl()`.
  Math fidelity is passed via `compute_kernel_config` in `forward()`.

## Prerequisites

- pcc-test-gen skill should have been run first to produce `shapes.json` and `op_map.json`.
  If not, ask the user for shapes manually.

## Step 1 -- Collect Inputs (ASK the user)

1. **Model name**: Which model's ops to sweep (determines shapes and test paths)

2. **Which op(s) to sweep**: Choose from:
   - `linear` (most common -- all Q/K/V/O/gate/up/down projections)
   - `attention` (SDPA parameters)
   - `rms_norm` / `layer_norm`
   - `moe` (expert routing)
   - Or a specific module path

3. **Sweep mode**: Quick (~18 configs) or full (~100+ configs)
   Quick defaults:
   - Weight dtype: bfloat16, bfloat8_b, bfloat4_b (3)
   - Math fidelity: HiFi4, HiFi2, LoFi (3)
   - fp32_dest_acc_en: True, False (2)
   - Total: 18 combinations

4. **PCC threshold**: Minimum acceptable (default: 0.999)

## How Config Application Works in tt_symbiote

**IMPORTANT**: The codebase does NOT have a unified config system. Config is applied via:

1. **Class selection**: Different TTNNLinear subclasses hardcode different weight dtypes in
   `preprocess_weights_impl()`:
   - `TTNNLinear` -> bfloat16
   - `TTNNLinearLLama` -> bfloat8_b

2. **Compute kernel config**: Passed directly to `ttnn.linear()` or `ttnn.matmul()` in `forward()`.

3. **Memory config**: `ttnn.DRAM_MEMORY_CONFIG` or `ttnn.L1_MEMORY_CONFIG`.

For sweeping, we create a **TTNNLinearSweep subclass** that parameterizes both:

## Step 2 -- Generate Sweep Test File

Create `tests/capabilities/<model_name>/test_sweep_<op_name>.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Parameter sweep for <op_name> in <model_name>.

HOW THIS WORKS:
Weight dtype is controlled by overriding preprocess_weights_impl() in a
TTNNLinear subclass. Math fidelity is passed via ttnn.linear()'s
compute_kernel_config parameter. This matches the codebase's actual
config application mechanism.
"""

import csv
import json
import time
import pytest
import torch
from pathlib import Path

import ttnn
from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.integrations.ttnn_linear import TTNNLinear
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
from tt_symbiote.utils.device_management import set_device
from tests.capabilities.pcc_utils import assert_pcc, compute_pcc

SHAPES = json.loads((Path(__file__).parent / "shapes.json").read_text())

WEIGHT_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}
MATH_FIDELITIES = {
    "HiFi4": ttnn.MathFidelity.HiFi4,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "LoFi": ttnn.MathFidelity.LoFi,
}
FP32_ACC = [True, False]

RESULTS_FILE = Path(__file__).parent / "sweep_results" / "<op_name>_sweep.csv"


class TTNNLinearSweep(TTNNLinear):
    """TTNNLinear subclass for runtime dtype/fidelity sweeping.

    Overrides preprocess_weights_impl() for weight dtype and
    forward() to pass compute_kernel_config -- matching the codebase's
    actual config mechanism (e.g., TTNNLinearLLama overrides preprocess_weights_impl).
    """

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self._sweep_weight_dtype = ttnn.bfloat16
        self._sweep_math_fidelity = ttnn.MathFidelity.HiFi4
        self._sweep_fp32_acc = True

    @classmethod
    def from_torch_with_config(cls, torch_linear, weight_dtype, math_fidelity, fp32_acc):
        """Create from a PyTorch linear with specific sweep configuration."""
        instance = cls.from_torch(torch_linear)
        instance._sweep_weight_dtype = weight_dtype
        instance._sweep_math_fidelity = math_fidelity
        instance._sweep_fp32_acc = fp32_acc
        return instance

    def preprocess_weights_impl(self):
        """Override to use the sweep-configured weight dtype."""
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=self._sweep_weight_dtype, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.bias, dtype=self._sweep_weight_dtype, layout=ttnn.TILE_LAYOUT
            )


@pytest.mark.parametrize("weight_dtype_name", list(WEIGHT_DTYPES.keys()))
@pytest.mark.parametrize("math_fidelity_name", list(MATH_FIDELITIES.keys()))
@pytest.mark.parametrize("fp32_acc", FP32_ACC)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_sweep_linear(device, weight_dtype_name, math_fidelity_name, fp32_acc):
    """Sweep linear with different configs."""
    torch.set_grad_enabled(False)
    hidden_size = SHAPES["config"]["hidden_size"]
    num_heads = SHAPES["config"]["num_attention_heads"]
    head_dim = SHAPES["config"].get("head_dim", hidden_size // num_heads)

    torch_linear = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    torch_linear = torch_linear.to(torch.bfloat16).eval()

    weight_dtype = WEIGHT_DTYPES[weight_dtype_name]
    math_fidelity = MATH_FIDELITIES[math_fidelity_name]

    try:
        ttnn_linear = TTNNLinearSweep.from_torch_with_config(
            torch_linear, weight_dtype, math_fidelity, fp32_acc
        )
        set_device(ttnn_linear, device)

        input_tensor = torch.randn(1, 128, hidden_size, dtype=torch.bfloat16)
        torch_output = torch_linear(input_tensor)
        ttnn_input = TorchTTNNTensor(input_tensor)

        # Warm up
        _ = ttnn_linear(ttnn_input)

        # Timed run
        start = time.perf_counter()
        ttnn_output = ttnn_linear(ttnn_input)
        elapsed = time.perf_counter() - start

        pcc_results = compute_pcc(ttnn_output, torch_output)
        pcc_val = pcc_results[0][0] if pcc_results else float('nan')
        status = "PASS" if pcc_val >= 0.999 else "FAIL"
    except RuntimeError as e:
        if "OOM" in str(e) or "out of memory" in str(e).lower():
            pytest.skip(f"OOM with config: {weight_dtype_name}/{math_fidelity_name}/fp32={fp32_acc}")
        raise

    # Write result to CSV
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["weight_dtype", "math_fidelity", "fp32_acc", "pcc", "time_ms", "status"])
        writer.writerow([
            weight_dtype_name, math_fidelity_name, fp32_acc,
            f"{pcc_val:.6f}", f"{elapsed*1000:.3f}", status
        ])

    assert pcc_val >= 0.999, (
        f"PCC {pcc_val:.6f} below threshold for "
        f"{weight_dtype_name}/{math_fidelity_name}/fp32={fp32_acc}"
    )
```

For SDPA attention sweeps, generate a separate file that parametrizes `q_chunk_size`, `k_chunk_size`, `exp_approx_mode` via `ttnn.SDPAProgramConfig`.

## Step 3 -- Run the Sweep

```bash
mkdir -p tests/capabilities/<model_name>/sweep_results
rm -f tests/capabilities/<model_name>/sweep_results/<op_name>_sweep.csv
pytest tests/capabilities/<model_name>/test_sweep_<op_name>.py -v --tb=no -q 2>&1 | tee sweep_output.txt
```

## Step 4 -- Analyze and Present Results

Parse the CSV, filter passing configs (PCC >= 0.999), sort by time. Present top 5 to user.

Save best config as `tests/capabilities/<model_name>/sweep_results/<op_name>_best.json`:
```json
{
    "op": "<op_name>",
    "model": "<model_name>",
    "device_arch": "T3K",
    "best_config": {
        "weight_dtype": "bfloat8_b",
        "math_fidelity": "HiFi2",
        "fp32_dest_acc_en": false
    },
    "recommended_class": "TTNNLinearLLama",
    "pcc": 0.999234,
    "time_ms": 1.234
}
```

## Error Handling

| Problem | Resolution |
|---------|------------|
| PCC < threshold for all configs | Widen threshold temporarily; report to user |
| bfloat4_b crashes | Known incompatibility with some ops; skip |
| Device OOM | Reduce batch size or seq_len; try DRAM-only configs |
| Timing variance too high | Increase number of runs; check for background processes |
