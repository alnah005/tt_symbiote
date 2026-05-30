---
name: tracy-profiling
description: Run tracy profiling on a model/module test to capture device performance data, generate ops_perf_results CSV and perf_report.txt. Profiles TTNN ops with signpost markers, warm-up run for kernel compilation, then tracy capture.
---

# Tracy Profiling

Profile a pytest test or standalone script to measure TTNN op device times.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/capabilities/<model_name>/test_modeling_<model_name>.py`

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.
  - `tests/models/` does NOT exist. Never create files there.
  - Shared capability tests: `tests/capabilities/` root (e.g., `test_attention.py`)
  - Auto/unit tests: `tests/auto/`

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
  Always use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.

**Deprecated API**: Never use `register_module_replacement_dict()`.
  Use `register_modules()` from `tt_symbiote.utils.module_replacement`.

**Import conventions**:
  - Integration modules: `from tt_symbiote.integrations.ttnn_<module> import TTNN<Class>`
  - Core: `from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch`
  - Device management: `from tt_symbiote.utils.device_management import set_device`

**Device/mesh_device fixtures**: Provided by the `ttnn` pytest plugin, NOT by tt_symbiote.
  Do NOT define these fixtures. Use `@pytest.mark.parametrize("device_params", [...], indirect=True)`.

**Config system**: The typed config system (ModuleConfig, DtypeConfig, etc.) does NOT exist yet.
  Current mechanism: `_model_config: dict` on TTNNModule, set via `set_model_config()`.
  Weight dtype is controlled by selecting different TTNNLinear subclasses.

**HuggingFace**: Always pass `trust_remote_code=True` in `AutoConfig.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()`.

## Step 1 -- Collect Inputs

Ask the user for:

1. **Test file path**: The pytest file or standalone script to profile.
   - Validate the file exists with `ls <path>`.
   - If the user gives a model name instead of a path, look for tests at:
     `tests/capabilities/<model_name>/test_modeling_<model_name>.py`

2. **Decoder layer limiting** (optional): For models with repeated decoder layers,
   ask whether to limit to 1-2 layers to avoid profiling the same structure N times.
   Default: Yes, limit to 2 layers.

3. **Output directory** (optional): Where to save results. Default: current working directory.

## Step 2 -- Pre-flight Checks

Run these checks before proceeding:

```bash
# Verify tracy is importable (hard dependency of tt_symbiote via run_config.py)
python -c "from tracy import signpost; print('tracy OK')"

# Verify tt-perf-report is available
which tt-perf-report || echo "WARNING: tt-perf-report not found in PATH"

# Verify the test file exists
test -f "<test_file>" && echo "Test file OK" || echo "ERROR: Test file not found"
```

**If tracy import fails**: tracy is NOT a pip package. It comes from tt-metal. Ensure the
tt-metal Python environment is active (source env/activate or equivalent). Check that
`python -c 'from tracy import signpost'` works.

**If tt-perf-report is missing**: The user can still get the raw CSV but not the summary report.
Proceed with profiling and note the missing tool.

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
4. Suggest next step: "Run op-sweep on the bottleneck ops to find optimal configs."

## Error Handling

| Problem | Diagnosis | Resolution |
|---------|-----------|------------|
| No CSV generated | Test may not have dispatched TTNN ops | Check run mode is NORMAL, verify device init |
| Empty CSV | Signpost mode not activated | Verify `TT_SYMBIOTE_SIGNPOST_MODE="1"` |
| Tracy crash/hang | Op buffer too small | Increase `--op-support-count` to 50000+ |
| Test fails during profiling | Different from warm-up failure | May be tracy interference; try `-r` only |
| `tt-perf-report` not found | Not in PATH | User can manually parse the CSV |
