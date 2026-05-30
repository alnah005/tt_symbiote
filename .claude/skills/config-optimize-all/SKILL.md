---
name: config-optimize-all
description: Apply optimal configs from perf-analysis to all modules via model-specific subclass overrides. Selects optimal TTNNLinear subclasses, creates model-specific overrides for compute config, re-validates PCC across all tests. Does NOT modify shared integration modules.
---

# Config Optimization (All Modules)

Apply the best-known configurations to every module in a model.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.

**License headers**: Every generated `.py` file starts with `(C)` format.

**PCC assertions**: Use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.

**Config system**: The typed config system does NOT exist. Use subclass-based overrides only:
  - Weight dtype: override `preprocess_weights_impl()` (or select existing class like TTNNLinearLLama)
  - Compute config: override `move_weights_to_device_impl()`
  - Memory layout: override `forward()` with appropriate `memory_config`

**CRITICAL**: Do NOT modify shared integration modules in `src/tt_symbiote/integrations/`.
  These are shared across ALL models.

## Prerequisites

- perf-analysis recommendation.json exists for the target model
- pcc-test-gen tests pass in NORMAL mode

## Step 1 -- Collect Inputs (ASK the user)

1. **Model name**: Which model to optimize
2. **Optimization goal**:
   - Accuracy-first: Keep bfloat16/HiFi4 where PCC is sensitive
   - Speed-first: Use fastest config that still meets PCC threshold
   - Balanced: Mix of accuracy and speed
3. **PCC threshold**: Minimum acceptable (default: 0.999)
4. **Device architecture**: Target arch (default: T3K)

## Step 2 -- Read Recommendations

Load `tests/capabilities/<model_name>/perf_results/recommendation.json`.

## Step 3 -- Apply Configs via Subclass Overrides

**IMPORTANT**: The ONLY working approach is subclass-based overrides. This follows the
established pattern (TTNNLinearLLama, TTNNBailingMoEAttention, etc.).

### For weight dtype overrides (override `preprocess_weights_impl`):

```python
# src/tt_symbiote/models/<model_name>/modeling_<model_name>.py

from tt_symbiote.integrations.ttnn_linear import TTNNLinear
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
import ttnn

class TTNNLinear<Model>(TTNNLinear):
    """Model-specific linear with optimized weight dtype."""

    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
```

### For compute config overrides (override `move_weights_to_device_impl`):

```python
from tt_symbiote.integrations.ttnn_attention import TTNNSelfAttention
import ttnn

class TTNNAttention<Model>(TTNNSelfAttention):
    """Model-specific attention with optimized compute config."""

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        self.sdpa.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            q_chunk_size=128, k_chunk_size=128,
            exp_approx_mode=True,
        )
        self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
```

### For memory config overrides (in forward):

```python
from tt_symbiote.core.module import run_on_devices, DeviceArch

class TTNNLinear<Model>L1(TTNNLinear):
    """Model-specific linear using L1 memory."""

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor):
        # ... forward with memory_config=ttnn.L1_MEMORY_CONFIG ...
```

## Step 4 -- Update register_modules Dict

Update the model's `from_torch()` or `build_module_dict()` to use new subclasses:

```python
register_modules(hf_model, {
    nn.Linear: TTNNLinear<Model>,  # was TTNNLinear
    # ...
})
```

## Step 5 -- Re-validate PCC

**ASK USER:** "Apply configs to all modules, or review each individually?"

```bash
pytest tests/capabilities/<model_name>/ -v --tb=short
```

If any test fails, revert that module's subclass and report.

## Step 6 -- Generate Optimization Report

Per-module before/after device time and PCC.

## Error Handling

| Problem | Resolution |
|---------|------------|
| No recommendation data | Use heuristic defaults |
| PCC regression | Revert specific module's subclass; report |
| Multiple arch targets | Generate separate config per arch |
