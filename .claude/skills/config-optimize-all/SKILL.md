---
name: config-optimize-all
description: Apply optimal configs from perf-analysis to all modules via model-specific subclass overrides. Selects optimal TTNNLinear subclasses, creates model-specific overrides for compute config, re-validates PCC across all tests. Does NOT modify shared integration modules.
---

# Config Optimization (All Modules)

Apply the best-known configurations to every module in a model.

## External Reference Skills (MANDATORY AWARENESS)

Before executing, you MUST read `.claude/skills/_shared/external_agentic_references.md`
and operate by the principles of these canonical tt-metal agentic skills (binding here):

- **optimize** — reconcile roofline ↔ device-time ↔ end-to-end from the SAME run; remove
  host gaps, don't just note them; **import the canonical precision/fidelity policy**
  (`tt_transformers/PERF.md`, `tt/model_config.py`) before inventing one; tune ONE dtype
  group at a time on REAL weights so regressions are assignable.
- **datatype-sweep** — full-model top-1/top-5 (here: argmax-agreement vs HF + degenerate
  fraction) is the source of truth, not PCC alone; emit a COMPLETE
  `selected_precision_config.json` (weight groups, layer exceptions, fidelities, CCL/KV
  dtype) — "bf8_b weights" alone is incomplete.
- **autofix** (on PCC/accuracy regression) — treat each suspected cause as a hypothesis;
  smallest verify/refute experiment first; keep only verified fixes.

See the shared file for links + offline `git show` commands + full digests.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`

**Test location**: Per-model tests live under `tests/models/<model_name>/` (RICH, e2e-traced)
  or `tests/experimental/<model_name>/` (partial-TTNN); for an already-brought-up model default
  to `tests/models/<model_name>/`.

**Pure TTNN forward**: ALL `TTNNModule.forward()` methods must use pure `ttnn.*` ops only.
  No `torch.*` calls in the compute path. When overriding `forward()` for memory config,
  the entire body must use `ttnn.*` ops.
  Reference: `$TT_METAL_HOME/models/tt_transformers/tt/mlp.py` forward() -- pure ttnn.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.

**License headers**: Every generated `.py` file starts with `(C)` format.

**PCC assertions**: Use `assert_pcc()` from `tests/shared/pcc_utils.py`.

**Config system**: The typed config system does NOT exist. Use subclass-based overrides only:
  - Weight dtype: override `preprocess_weights_impl()` (or select existing class like TTNNLinearLLama)
  - Compute config: override `move_weights_to_device_impl()`
  - Memory layout: override `forward()` with appropriate `memory_config`

**CRITICAL**: Do NOT modify shared integration modules in `src/tt_symbiote/modules/`.
  These are shared across ALL models.

**TT_METAL_COMMIT Hash**: When modifying `modeling_<model_name>.py`, ensure
  `TT_METAL_COMMIT = '<hash>'` is present. Update if stale:
  ```bash
  git -C "$TT_METAL_HOME" rev-parse HEAD
  ```

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

Key reports for config-optimize-all (read FIRST):
1. `data_formats/data_formats.md` -- Dtype tradeoffs for weight and activation optimization
2. `GEMM_FLOPS/GEMM_FLOPS.md` -- understanding the op for config selection (NOT for estimating device time — that comes from the tracy CSV)
3. `tensor_sharding/tensor_sharding.md` -- Sharding strategy for memory config optimization
4. `memory/allocator.md` -- L1 vs DRAM config selection
5. `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` -- Multi-technique optimization
6. `YoloV4-TTNN/yolov4.md` -- Data type optimization examples

Read ALL remaining reports after these priority ones.

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
```

**Key extraction for config optimization**: Study `model_config.py` to understand:
- `DecoderOptimizations` pattern -- per-layer dtype and math fidelity tuning
- How `get_math_fidelity()`, `get_tensor_dtype()` parameterize per-op configs
- How the config is applied at the `ttnn.linear()` call site, not as module attributes

### 0c. Capture TT_METAL_COMMIT Hash

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
TT_METAL_COMMIT=$(git -C "$TT_METAL_HOME" rev-parse HEAD)
echo "TT_METAL_COMMIT=$TT_METAL_COMMIT"
```

## Plan-Verify-Execute Loop

This skill follows a mandatory loop structure. If the loop fails 5 times, report failure to the caller.

### PLAN Phase
1. Collect inputs (model name, optimization goal, PCC threshold, target arch)
2. Read perf-analysis recommendation.json
3. Design subclass overrides for each module, ensuring forward() stays pure TTNN
4. Draft the optimized modeling file contents

### VERIFY Phase (no hardware, no user approval needed)
1. Verify recommendation.json exists and is valid JSON
2. Verify all overridden `forward()` methods use only `ttnn.*` ops (pure TTNN constraint)
3. Verify all overridden `forward()` methods have `@run_on_devices` decorator
4. Verify no shared integration modules in `src/tt_symbiote/modules/` are modified
5. Verify license headers are present on all generated files
6. Verify `TT_METAL_COMMIT` constant is present in the modeling file
7. If ANY verification fails, return to PLAN with failure details and re-plan

### EXECUTE Phase (only after VERIFY passes)
Write the optimized subclasses, update register_modules dict, run PCC validation.

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

Load `tests/models/<model_name>/perf_results/recommendation.json`.

## Step 3 -- Apply Configs via Subclass Overrides

**IMPORTANT**: The ONLY working approach is subclass-based overrides. This follows the
established pattern (TTNNLinearLLama, TTNNBailingMoEAttention, etc.).

### For weight dtype overrides (override `preprocess_weights_impl`):

```python
# src/tt_symbiote/models/<model_name>/modeling_<model_name>.py

# Commit hash for reproducibility
TT_METAL_COMMIT = "<40-char hash>"

from tt_symbiote.modules.ttnn_linear import TTNNLinear
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
from tt_symbiote.modules.ttnn_attention import TTNNSelfAttention
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

### For memory config overrides (in forward -- pure TTNN only):

```python
from tt_symbiote.core.module import run_on_devices, DeviceArch

class TTNNLinear<Model>L1(TTNNLinear):
    """Model-specific linear using L1 memory."""

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor):
        # Pure TTNN ops only -- no torch.* calls
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

## Step 5 -- Update TT_METAL_COMMIT

Ensure `modeling_<model_name>.py` has the current commit hash:

```python
TT_METAL_COMMIT = '<full 40-char hash from Step 0c>'
```

## Step 6 -- Re-validate PCC (bottom-up)

Apply optimizations BOTTOM-UP (leaves first). After each module, re-validate PCC across the
rich tiers; roll back that module's subclass/override on ANY regression before ascending to its
parent. Modify ONLY model-specific subclass overrides — never the shared integration modules in
`src/tt_symbiote/integrations/` (`src/tt_symbiote/modules/`).

**ASK USER:** "Apply configs to all modules, or review each individually?"

```bash
pytest tests/models/<model_name>/ -v --tb=short
```

If any test fails, revert that module's subclass and report.

## Step 7 -- Generate Optimization Report

Per-module before/after device time and PCC. Device time comes SOLELY from the tracy
`ops_perf_results_*.csv` DEVICE TIME (ns) column — never from hardware-limit estimates or
projected numbers.

## Error Handling

| Problem | Resolution |
|---------|------------|
| No recommendation data | Use heuristic defaults from tech reports |
| PCC regression | Revert specific module's subclass; report |
| Multiple arch targets | Generate separate config per arch |
| torch.* in forward override (A1 violation) | Refactor to pure ttnn immediately |
