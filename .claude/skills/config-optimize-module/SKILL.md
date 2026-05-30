---
name: config-optimize-module
description: Apply optimal config from perf-analysis to a single module via model-specific subclass override. Runs targeted sweep if needed, selects optimal TTNNLinear subclass, validates PCC, reports before/after comparison.
---

# Config Optimization (Single Module)

Fine-tune configuration for one specific module.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.

**License headers**: Every generated `.py` file starts with `(C)` format.

**PCC assertions**: Use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.

**Config system**: The typed config system does NOT exist. Use subclass-based overrides:
  - Weight dtype: override `preprocess_weights_impl()`
  - Compute config: override `move_weights_to_device_impl()`
  - Memory layout: override `forward()` with appropriate memory_config
  - Always add `@run_on_devices(DeviceArch.T3K)` on any overridden forward()

**CRITICAL**: Do NOT modify shared integration modules in `src/tt_symbiote/integrations/`.

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

## Step 5 -- Validate PCC

Run only the tests that exercise this module (Tier 1 for the op, plus Tier 2/3 composites).

## Step 6 -- Report Before/After

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
