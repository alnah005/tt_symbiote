---
name: tracy-profiling
description: Run tracy profiling on a model/module test to capture device performance data, generate ops_perf_results CSV and perf_report.txt. Profiles TTNN ops with signpost markers, warm-up run for kernel compilation, then tracy capture.
---

# Tracy Profiling

Profile a pytest test or standalone script to measure TTNN op device times.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/models/<model_name>/test_modeling_<model_name>.py`

**Test location**: Per-model tests live under `tests/models/<model_name>/` (RICH, e2e-traced)
  or `tests/experimental/<model_name>/` (partial-TTNN). For an already-brought-up model default
  to `tests/models/<model_name>/`. Profiling outputs go to `tests/models/<model_name>/profiling/`.
  - Shared capability tests: `tests/shared/` root (e.g., `test_attention.py`)
  - Auto/unit tests: `tests/auto/`

**Tracy-only device time (Req 4)**: Device time is reported SOLELY from the tracy
  `ops_perf_results_*.csv` DEVICE TIME (ns) column. Never estimate, project, or compute device
  time from theoretical hardware limits, from FLOP counts, or from any efficiency ratio.
  `GEMM_FLOPS/GEMM_FLOPS.md` is reading material only.

**Tech-Report Reading Gate (Req 8)**: BEFORE any profiling-driven new-module work, append an
  additive `references_read` record to the model's `bringup_status.json`
  (`tt_metal_commit` + `timestamp` + `tech_reports` + `reference_impls` + `consulted_paths`);
  additive only — never remove existing keys.

**Pure TTNN forward**: ALL `TTNNModule.forward()` methods must use pure `ttnn.*` ops only.
  No `torch.*` calls in the compute path. Weight preprocessing may use PyTorch.
  Reference: `$TT_METAL_HOME/models/tt_transformers/tt/mlp.py` forward().
  When profiling, VERIFY the code under test follows this convention. Torch fallbacks
  in forward() will produce misleading profiling results.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.
  - Import: `from tt_symbiote.core.module import run_on_devices, DeviceArch`
  - Default: `@run_on_devices(DeviceArch.T3K)`
  - Widen only after explicit validation on other architectures.

**License headers**: Every generated `.py` file must start with:
  ```python
  # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
  # SPDX-License-Identifier: Apache-2.0
  ```
  NOTE: Some existing framework files use the Unicode copyright symbol. Do NOT change those. Use `(C)` for all NEW files.

**PCC assertions**: NEVER rely on `compare_fn_outputs()` alone -- it only prints warnings.
  Always use `assert_pcc()` from `tests/shared/pcc_utils.py`.

**Deprecated API**: Never use `register_module_replacement_dict()`.
  Use `register_modules()` from `tt_symbiote.utils.module_replacement`.

**Import conventions**:
  - Integration modules: `from tt_symbiote.modules.ttnn_<module> import TTNN<Class>`
  - Core: `from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch`
  - Device management: `from tt_symbiote.utils.device_management import set_device`

**Device/mesh_device fixtures**: Provided by the `ttnn` pytest plugin, NOT by tt_symbiote.
  Do NOT define these fixtures. Use `@pytest.mark.parametrize("device_params", [...], indirect=True)`.

**Config system**: The typed config system (ModuleConfig, DtypeConfig, etc.) does NOT exist yet.
  Current mechanism: `_model_config: dict` on TTNNModule, set via `set_model_config()`.
  Weight dtype is controlled by selecting different TTNNLinear subclasses.

**HuggingFace**: Always pass `trust_remote_code=True` in `AutoConfig.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()`.

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

Key reports for tracy-profiling (read these FIRST):
1. `MetalProfiler/metal-profiler.md` -- Profiler tool details
2. `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` -- Trace, 2CQ, Metal Trace optimization
3. `GEMM_FLOPS/GEMM_FLOPS.md` -- Understanding matmul performance metrics
4. `ttnn/TTNN-model-bringup.md` -- Canonical model bring-up flow including profiling steps
5. `data_formats/data_formats.md` -- How data format affects profiling results
6. `memory/allocator.md` -- L1 vs DRAM performance implications
7. `tensor_sharding/tensor_sharding.md` -- Sharding impact on op performance

Read ALL remaining reports after these priority ones.

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
ls "$TT_METAL_HOME/models/tt_dit/" 2>/dev/null
ls "$TT_METAL_HOME/models/tt_cnn/tt/" 2>/dev/null
ls "$TT_METAL_HOME/models/demos/" 2>/dev/null
```

**Extract from tt_transformers**: Understand how forward() is pure ttnn -- this affects
what ops appear in the tracy profile. Note the pattern:
- `model.py`: `Transformer.forward()` -- pure ttnn ops
- `mlp.py`: `MLP.forward()` -- `ttnn.linear`, `ttnn.multiply`, `ttnn.silu`
- `attention.py`: `Attention.forward_decode/prefill()` -- ttnn attention ops
- `model_config.py`: How math fidelity and memory configs are parameterized

## Plan-Verify-Execute Loop

This skill follows a mandatory loop structure. If the loop fails 5 times, report failure to the caller.

### PLAN Phase
1. Collect inputs from user
2. Determine test file, output directory, decoder layer limits
3. Design the profiling strategy based on model architecture

### VERIFY Phase (no hardware, no user approval needed)
1. Verify tracy is importable: `python -c "from tracy import signpost; print('tracy OK')"`
2. Verify tt-perf-report is available: `which tt-perf-report`
3. Verify the test file exists and has valid imports: `python -c "import ast; ast.parse(open('<test_file>').read())"`
4. Verify no `torch.*` calls in any `forward()` bodies of the model being profiled
5. If ANY verification fails, return to PLAN with the failure details and re-plan

### EXECUTE Phase (only after VERIFY passes)
Proceed with warm-up, profiling, and report generation.

## Step 1 -- Collect Inputs

Ask the user for:

1. **Test file path**: The pytest file or standalone script to profile.
   - Validate the file exists with `ls <path>`.
   - If the user gives a model name instead of a path, look for tests at:
     `tests/models/<model_name>/test_modeling_<model_name>.py`

2. **Decoder layer limiting** (optional): For models with repeated decoder layers,
   ask whether to limit to 1-2 layers to avoid profiling the same structure N times.
   Default: Yes, limit to 2 layers.

3. **Output directory** (optional): Where to save results. Default: current working directory.

## Step 2 -- Pre-flight Checks (VERIFY)

Run these checks before proceeding:

```bash
# Verify tracy is importable (hard dependency of tt_symbiote via run_config.py)
python -c "from tracy import signpost; print('tracy OK')"

# Verify tt-perf-report is available
which tt-perf-report || echo "WARNING: tt-perf-report not found in PATH"

# Verify the test file exists
test -f "<test_file>" && echo "Test file OK" || echo "ERROR: Test file not found"

# A1 CHECK: Verify forward() methods in the model code are pure TTNN
grep -n "torch\." src/tt_symbiote/models/*/modeling_*.py 2>/dev/null | grep -v "import\|#\|preprocess_weights\|from_torch\|__init__" || echo "Pure TTNN forward: OK"
```

**If tracy import fails**: tracy is NOT a pip package. It comes from tt-metal. Ensure the
tt-metal Python environment is active (source env/activate or equivalent). Check that
`python -c 'from tracy import signpost'` works.

**If tt-perf-report is missing**: The user can still get the raw CSV but not the summary report.
Proceed with profiling and note the missing tool.

**If any VERIFY step fails**: Return to PLAN, diagnose, and re-plan.

## Step 3 -- Decoder Layer Limiting (if requested)

If the user chose to limit decoder layers:

1. Read the test file
2. Find where the model's layers are accessed (e.g., `model.model.layers`)
3. Suggest adding a line after model loading: `model.model.layers = model.model.layers[:2]`
4. **ASK USER:** "Apply this modification? (yes/no)"

**IMPORTANT**: This modification is temporary for profiling only. Do NOT commit it.

## Step 4 -- Warm-up Run (Kernel Compilation)

Run the test once without tracy to compile all kernels:

```bash
export TT_SYMBIOTE_RUN_MODE=NORMAL
pytest <test_file> -x -s --no-header -rN 2>&1 | tail -30
```

**Why**: TTNN compiles kernels on first execution. Profiling before compilation
would include compilation time in the measurements.

**If the warm-up fails**: Report the error. Do NOT proceed to profiling.
Common failures:
- Device not available: Check TT_METAL_HOME, device driver
- Import errors: Check environment
- Test logic errors: Fix the test first

## Step 5 -- Tracy Profiling Run

```bash
export TT_SYMBIOTE_SIGNPOST_MODE="1"
export TT_SYMBIOTE_RUN_MODE=NORMAL
python -m tracy -p -r -v --op-support-count 20000 -m 'pytest <test_file> -x -s --no-header -rN'
```

**Flag explanations**:
- `-p`: Enable profiling
- `-r`: Generate report
- `-v`: Verbose output
- `--op-support-count 20000`: Increase op buffer (large models need this)
- `-m`: Module mode (runs pytest as a module)

**CRITICAL**: Always profile in `NORMAL` mode, NOT `TRACED` mode.
Traced mode replays pre-recorded traces, not actual op dispatch.
`TT_SYMBIOTE_SIGNPOST_MODE="1"` activates signpost markers in run_config.py.

## Step 6 -- Generate Performance Report

```bash
tt-perf-report --ignore-signposts */ops_perf_results_*.csv > perf_report.txt
```

`--ignore-signposts` omits signpost marker rows from the report for cleaner data.

## Step 7 -- Present Results

1. Report the CSV file path and perf_report.txt path to the user
2. Parse and summarize the top 10 ops by device time from the CSV
3. Identify any ops taking >10% of total device time as optimization targets
4. Flag if any profiled ops appear to be torch fallbacks (non-ttnn ops in the trace) -- this indicates an A1 violation
5. Suggest next step: "Run op-sweep on the bottleneck ops to find optimal configs."

## Error Handling

| Problem | Diagnosis | Resolution |
|---------|-----------|------------|
| No CSV generated | Test may not have dispatched TTNN ops | Check run mode is NORMAL, verify device init |
| Empty CSV | Signpost mode not activated | Verify `TT_SYMBIOTE_SIGNPOST_MODE="1"` |
| Tracy crash/hang | Op buffer too small | Increase `--op-support-count` to 50000+ |
| Test fails during profiling | Different from warm-up failure | May be tracy interference; try `-r` only |
| `tt-perf-report` not found | Not in PATH | User can manually parse the CSV |
| torch ops in profile | Model has torch fallbacks in forward() | Fix model code to use pure TTNN forward (A1) |
