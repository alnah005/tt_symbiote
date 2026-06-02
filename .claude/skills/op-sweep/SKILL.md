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

**Pure TTNN forward**: ALL `TTNNModule.forward()` methods must use pure `ttnn.*` ops only.
  No `torch.*` calls in the compute path. Weight preprocessing may use PyTorch.
  The sweep subclass overrides `preprocess_weights_impl()` for dtype (torch allowed here)
  and inherits `forward()` which must be pure ttnn.
  Reference: `$TT_METAL_HOME/models/tt_transformers/tt/mlp.py` -- note how math fidelity
  and dtype are applied within pure ttnn `ttnn.linear()` calls.

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

## Step 0 -- Mandatory Exploration Preamble

**This step is NON-NEGOTIABLE. Complete it IN FULL before proceeding to Step 1.**

### 0a. Read ALL Tech Reports

Read every tech report in `$TT_METAL_HOME/tech_reports/`:

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
for f in $(find "$TT_METAL_HOME/tech_reports" -name "*.md" | sort); do
  echo "=== Reading: $f ==="
  cat "$f"
done
```

Key reports for op-sweep (read FIRST):
1. `data_formats/data_formats.md` -- bfloat16 vs bfloat8_b vs bfloat4_b tradeoffs, PCC impact
2. `GEMM_FLOPS/GEMM_FLOPS.md` -- Understanding matmul performance ceiling
3. `memory/allocator.md` -- L1 vs DRAM memory config impact on performance
4. `tensor_sharding/tensor_sharding.md` -- Sharding strategy impact on op performance
5. `ttnn/TTNN-model-bringup.md` -- Optimization stage guidance
6. `YoloV4-TTNN/yolov4.md` -- Data type optimization examples (section 2.3)
7. `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` -- Advanced optimization patterns

Read ALL remaining reports after these priority ones.

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
ls "$TT_METAL_HOME/models/tt_dit/" 2>/dev/null
ls "$TT_METAL_HOME/models/tt_cnn/tt/" 2>/dev/null
ls "$TT_METAL_HOME/models/demos/" 2>/dev/null
```

**Key extraction for sweeps**: Study `model_config.py` in tt_transformers (~196K bytes,
~4,236 lines) to understand how math fidelity, dtype, and memory configs are selected.
Also study `mlp.py` to see how `compute_kernel_config` and `activation_dtype`
are applied within pure ttnn `ttnn.linear()` calls.

## Plan-Verify-Execute Loop

This skill follows a mandatory loop structure. If the loop fails 5 times, report failure to the caller.

### PLAN Phase
1. Collect inputs (model name, ops to sweep, sweep mode, PCC threshold)
2. Read shapes.json and op_map.json to determine parameter space
3. Draft sweep test file contents with the TTNNLinearSweep subclass
4. Determine the sweep parameter grid

### VERIFY Phase (no hardware, no user approval needed)
1. Verify shapes.json exists and is valid: `python -c "import json; json.load(open('tests/capabilities/<model_name>/shapes.json'))"`
2. Verify all imports resolve: `python -c "from tt_symbiote.modules.ttnn_linear import TTNNLinear; from ttnn.model_preprocessing import preprocess_linear_weight"`
3. Verify the sweep subclass forward() uses only `ttnn.*` ops (pure TTNN constraint)
4. Verify `@run_on_devices` is present on any overridden `forward()` methods
5. Verify license headers are present
6. If ANY verification fails, return to PLAN with failure details and re-plan

### EXECUTE Phase (only after VERIFY passes)
Write sweep test file, run the sweep, collect and analyze results.

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

**Reference from $TT_METAL_HOME**: In `tt_transformers/tt/mlp.py`, the forward() uses
`ttnn.linear(x, self.w1, ..., compute_kernel_config=li_ff1_3_compute_kernel_cfg, ...)` --
the compute kernel config is passed inline to the ttnn op, not set on the module.

For sweeping, we create a **TTNNLinearSweep subclass** that parameterizes both.

## Step 2 -- Generate Sweep Test File

Create `tests/capabilities/<model_name>/test_sweep_<op_name>.py` with:
- A `TTNNLinearSweep` subclass that overrides `preprocess_weights_impl()` for weight dtype
- Parametrized test function sweeping weight_dtype x math_fidelity x fp32_acc
- CSV result output to `sweep_results/<op_name>_sweep.csv`
- OOM handling via `pytest.skip`

## Step 3 -- Run the Sweep

```bash
mkdir -p tests/capabilities/<model_name>/sweep_results
rm -f tests/capabilities/<model_name>/sweep_results/<op_name>_sweep.csv
pytest tests/capabilities/<model_name>/test_sweep_<op_name>.py -v --tb=no -q 2>&1 | tee sweep_output.txt
```

## Step 4 -- Analyze and Present Results

Parse the CSV, filter passing configs (PCC >= 0.999), sort by time. Present top 5 to user.

**Cross-reference with tech reports**: Check if recommended configs from `YoloV4-TTNN/yolov4.md`
(section 2.3: "set math_fidelity=ttnn.MathFidelity.LoFi unless noticeable PCC drop")
align with sweep results.

Save best config as `tests/capabilities/<model_name>/sweep_results/<op_name>_best.json`.

## Error Handling

| Problem | Resolution |
|---------|------------|
| PCC < threshold for all configs | Widen threshold temporarily; report to user |
| bfloat4_b crashes | Known incompatibility with some ops; skip |
| Device OOM | Reduce batch size or seq_len; try DRAM-only configs |
| Timing variance too high | Increase number of runs; check for background processes |
