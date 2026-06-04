---
name: config-optimize-module
description: Apply optimal config from perf-analysis to a single module via model-specific subclass override. Runs targeted sweep if needed, selects optimal TTNNLinear subclass, validates PCC, reports before/after comparison.
---

# Config Optimization (Single Module)

Fine-tune configuration for one specific module.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.

**Test location**: Per-model tests live under `tests/models/<model_name>/` (RICH, e2e-traced)
  or `tests/experimental/<model_name>/` (partial-TTNN); for an already-brought-up model default
  to `tests/models/<model_name>/`.

**Pure TTNN forward**: ALL `TTNNModule.forward()` methods must use pure `ttnn.*` ops only.
  No `torch.*` calls in the compute path.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.
  - Always add `@run_on_devices(DeviceArch.T3K)` on any overridden forward()

**License headers**: Every generated `.py` file starts with `(C)` format.

**PCC assertions**: Use `assert_pcc()` from `tests/shared/pcc_utils.py`.

**Config system**: The typed config system does NOT exist. Use subclass-based overrides:
  - Weight dtype: override `preprocess_weights_impl()`
  - Compute config: override `move_weights_to_device_impl()`
  - Memory layout: override `forward()` with appropriate memory_config

**CRITICAL**: Do NOT modify shared integration modules in `src/tt_symbiote/modules/`.

**TT_METAL_COMMIT Hash**: When modifying `modeling_<model_name>.py`, ensure
  `TT_METAL_COMMIT = '<hash>'` is present and current.

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

Key reports for config-optimize-module (read FIRST):
1. `data_formats/data_formats.md` -- Dtype impact on compute and PCC for the specific op
2. `GEMM_FLOPS/GEMM_FLOPS.md` -- matmul background for the module's dominant op (read for understanding only; device time comes from the tracy CSV, not from hardware-limit estimates)
3. `memory/allocator.md` -- L1 vs DRAM for this specific module's access pattern
4. `tensor_sharding/tensor_sharding.md` -- Optimal sharding for the module's tensor shapes
5. `YoloV4-TTNN/yolov4.md` -- Per-op optimization examples

Read ALL remaining reports after these priority ones.

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
```

**Key extraction**: Find the reference implementation of the same type of module you are
optimizing. For example:
- Optimizing a linear projection? -> Study mlp.py's forward(), note compute_kernel_config usage
- Optimizing attention? -> Study attention.py's forward_decode/prefill, note SDPA config
- Optimizing a norm? -> Study decoder.py's norm call, note the norm_config pattern

### 0c. Capture TT_METAL_COMMIT Hash

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
TT_METAL_COMMIT=$(git -C "$TT_METAL_HOME" rev-parse HEAD)
echo "TT_METAL_COMMIT=$TT_METAL_COMMIT"
```

## Plan-Verify-Execute Loop

This skill follows a mandatory loop structure. If the loop fails 5 times, report failure to the caller.

### PLAN Phase
1. Collect inputs (module to optimize, model name, optimization goal)
2. Analyze current module config (class, dtype, fidelity, memory)
3. Check for existing sweep data or plan a targeted sweep
4. Design the optimal subclass override

### VERIFY Phase (no hardware, no user approval needed)
1. Verify the module exists and can be imported
2. Verify the overridden `forward()` uses only `ttnn.*` ops (pure TTNN constraint)
3. Verify `@run_on_devices` is present on any overridden `forward()`
4. Verify no shared integration modules are modified
5. Verify `TT_METAL_COMMIT` constant is present
6. If ANY verification fails, return to PLAN with failure details and re-plan

### EXECUTE Phase (only after VERIFY passes)
Write the subclass, update register_modules, run targeted PCC validation.

## Step 1 -- Collect Inputs (ASK the user)

1. **Module to optimize**: A module class name, module path, or file:line reference
2. **Model name**: Which model this module belongs to
3. **Optimization goal**: accuracy-first, speed-first, or balanced

## Step 2 -- Analyze Current Module Config

Read the module source and identify the current TTNNLinear subclass used. Report:
```
Current config for layers[5].self_attn.q_proj:
  TTNNLinear class: TTNNLinearIColShardedWRowSharded
  weight_dtype: bfloat16
  math_fidelity: HiFi4 (default)
  memory: DRAM
```

## Step 3 -- Run Targeted Sweep (if needed)

If no sweep data exists for this module, generate a focused sweep test (using op-sweep methodology) and run it.

## Step 4 -- Apply Best Config

For a single-module override, create a model-specific subclass in `modeling_<model_name>.py`:

```python
# TT_METAL_COMMIT must be at module level
TT_METAL_COMMIT = "<40-char hash>"

class TTNNLinear<Model>QProj(TTNNLinear):
    """Optimized q_proj for <model_name>."""
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
```

If the module uses a different class than other modules, use `exclude_replacement` in `register_modules`:

```python
exclude = {"model.layers.5.self_attn.q_proj"}
register_modules(model, {nn.Linear: TTNNLinear}, exclude_replacement=exclude)
register_modules(model.model.layers[5].self_attn, {nn.Linear: TTNNLinear<Model>QProj})
```

## Step 5 -- Update TT_METAL_COMMIT

Ensure `modeling_<model_name>.py` has the current commit hash.

## Step 6 -- Validate PCC (bottom-up re-validation hook)

Run the tests that exercise this module (Tier 1 for the op, plus Tier 2/3 composites) AND any
affected higher tiers. This is the single-module re-validation hook in the bottom-up tuning
workflow: re-validate this module and the affected tiers BEFORE reporting. If PCC regresses
below threshold, ROLL BACK the config change (restore the prior subclass/override) and report
the regression. Do NOT ascend to the parent module with a regressed leaf.

## Step 7 -- Report Before/After

Timing in the before/after comparison comes SOLELY from the tracy `ops_perf_results_*.csv`
DEVICE TIME (ns) column — never from hardware-limit estimates or projected numbers.

```
Module: model.layers[5].self_attn.q_proj
Before: TTNNLinear (bfloat16/HiFi4) -> PCC=0.999823, time=2.5ms
After:  TTNNLinear<Model>QProj (bfloat8_b/HiFi2) -> PCC=0.999234, time=1.1ms
Speedup: 2.3x
PCC delta: -0.000589 (still above 0.999)
```

## Error Handling

| Problem | Resolution |
|---------|------------|
| Module not found | Search integrations/ for the class; ask user |
| PCC drops below threshold | Show PCC; ask user if acceptable |
| Per-module override not supported | Use exclude_replacement + separate register_modules call |
| torch.* in forward override (A1) | Refactor to pure ttnn immediately |
