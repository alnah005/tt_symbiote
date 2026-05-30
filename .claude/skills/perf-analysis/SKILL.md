---
name: perf-analysis
description: Analyze tracy profiling data and sweep results to identify performance bottlenecks, rank ops by device time, and generate optimization recommendations with specific config changes (TTNNLinear class selection and override methods).
---

# Performance Analysis

Analyze profiling data to identify bottlenecks and recommend optimizations.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/capabilities/<model_name>/test_modeling_<model_name>.py`

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.

**PCC assertions**: Use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.

**Config system**: The typed config system does NOT exist yet. Optimization is done via
  subclass selection and method overrides:
  - Weight dtype: override `preprocess_weights_impl()` (e.g., TTNNLinearLLama uses bfloat8_b)
  - Compute config: override `move_weights_to_device_impl()` for SDPA/compute kernel config
  - Memory layout: override `forward()` with appropriate memory_config

## Prerequisites

- Tracy CSV data: `ops_perf_results_*.csv` from tracy-profiling skill
- Sweep results (optional): `sweep_results/<op>_sweep.csv` from op-sweep skill

## Step 1 -- Collect Inputs (ASK the user)

1. **Tracy CSV path**: Path to `ops_perf_results_*.csv`
2. **Sweep results path** (optional): Path to sweep CSVs
3. **Model name**: For organizing output
4. **Target DeviceArch**: For arch-specific recommendations

## Step 2 -- Parse Tracy Data

```python
import csv
from collections import defaultdict

with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

op_times = defaultdict(list)
for row in rows:
    op_name = row.get("OP TYPE", row.get("op_type", "unknown"))
    device_time = float(row.get("DEVICE TIME (ns)", row.get("device_time_ns", 0)))
    op_times[op_name].append(device_time)

total_device_time = sum(sum(times) for times in op_times.values())
op_summary = []
for op, times in op_times.items():
    op_summary.append({
        "op": op, "count": len(times),
        "total_ns": sum(times), "avg_ns": sum(times) / len(times),
        "max_ns": max(times),
        "pct_total": sum(times) / total_device_time * 100 if total_device_time > 0 else 0,
    })
op_summary.sort(key=lambda x: x["total_ns"], reverse=True)
```

## Step 3 -- Cross-reference with Sweep Results

For each top-10 op, check if a faster config exists in sweep results.

## Step 4 -- Generate Recommendations

Save to `tests/capabilities/<model_name>/perf_results/recommendation.json`:

```json
{
  "device_arch": "<arch>",
  "model_name": "<model>",
  "recommendations": [
    {
      "op": "ttnn.linear",
      "module": "TTNNLinear",
      "best_config": {
        "math_fidelity": "HiFi2",
        "fp32_dest_acc_en": true,
        "weight_dtype": "bfloat8_b",
        "device_time_ns": 1200,
        "pcc": 0.9998
      },
      "implementation": "subclass",
      "override_method": "preprocess_weights_impl",
      "recommended_class": "TTNNLinearLLama",
      "how_to_apply": "In register_modules() or build_module_dict(), map nn.Linear to TTNNLinearLLama instead of TTNNLinear"
    }
  ]
}
```

The `implementation` field specifies the approach (always "subclass" since the typed config system does not exist). The `override_method` tells config-optimize skills which method to override.

## Step 5 -- Present Results

Show formatted ranking table to user. Identify ops taking >10% of total device time.
Suggest next step: "Run config-optimize-all or config-optimize-module to apply."

## Error Handling

| Problem | Resolution |
|---------|------------|
| CSV columns don't match expected format | Auto-detect column names |
| No sweep data available | Recommendations based on heuristics only |
| All ops roughly equal time | Model is well-balanced; focus on total time |
