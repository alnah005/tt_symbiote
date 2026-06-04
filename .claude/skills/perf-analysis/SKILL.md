---
name: perf-analysis
description: Analyze tracy profiling data and sweep results to identify performance bottlenecks, rank ops by device time, and generate optimization recommendations with specific config changes (TTNNLinear class selection and override methods).
---

# Performance Analysis

Analyze profiling data to identify bottlenecks and recommend optimizations.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/models/<model_name>/test_modeling_<model_name>.py`

**Test location**: Per-model tests live under `tests/models/<model_name>/` (RICH, e2e-traced)
  or `tests/experimental/<model_name>/` (partial-TTNN); for an already-brought-up model default
  to `tests/models/<model_name>/`. The `recommendation.json` is written to
  `tests/models/<model_name>/perf_results/`.

**Pure TTNN forward**: When analyzing profiles, FLAG any ops that appear to be
  torch fallbacks (non-ttnn ops in the trace). These indicate A1 violations that should
  be fixed before optimization -- optimizing around a torch fallback is wasted effort.

**PCC assertions**: Use `assert_pcc()` from `tests/shared/pcc_utils.py`.

**Config system**: The typed config system does NOT exist yet. Optimization is done via
  subclass selection and method overrides:
  - Weight dtype: override `preprocess_weights_impl()` (e.g., TTNNLinearLLama uses bfloat8_b)
  - Compute config: override `move_weights_to_device_impl()` for SDPA/compute kernel config
  - Memory layout: override `forward()` with appropriate memory_config

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

Key reports for perf-analysis (read FIRST):
1. `GEMM_FLOPS/GEMM_FLOPS.md` -- matmul/GEMM background (read for understanding only; device time is read from the tracy CSV, never computed from this report)
2. `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` -- Optimization techniques reference
3. `data_formats/data_formats.md` -- dtype impact on compute throughput
4. `memory/allocator.md` -- L1 vs DRAM bandwidth implications
5. `tensor_sharding/tensor_sharding.md` -- Sharding impact on parallelism and data movement
6. `Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md` -- DRAM bandwidth bottleneck analysis
7. `YoloV4-TTNN/yolov4.md` -- End-to-end optimization case study with perf analysis
8. `MetalProfiler/metal-profiler.md` -- Understanding profiler output format

Read ALL remaining reports after these priority ones.

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
```

**Key extraction for perf-analysis**: Study `model_config.py` (~196K bytes, ~4,236 lines)
to understand:
- How per-op and per-layer configs are selected (read for background)
- How configs are selected per-op and per-layer
- The DecoderOptimizations pattern for per-layer dtype/fidelity tuning

## Plan-Verify-Execute Loop

This skill follows a mandatory loop structure. If the loop fails 5 times, report failure to the caller.

### PLAN Phase
1. Collect inputs (CSV path, sweep results, model name, target arch)
2. Parse tracy data to identify top ops by device time
3. Cross-reference with sweep results to find faster configs
4. Draft optimization recommendations

### VERIFY Phase (no hardware, no user approval needed)
1. Verify the CSV file exists and has expected columns
2. Verify recommendations reference valid TTNNLinear subclasses and override methods
3. Verify recommended forward() implementations use only `ttnn.*` ops (pure TTNN constraint)
4. Verify any generated code has `@run_on_devices` decorators
5. If ANY verification fails, return to PLAN with failure details and re-plan

### EXECUTE Phase (only after VERIFY passes)
Generate recommendation JSON, present results to user.

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

## Step 3 -- Cross-reference with Sweep Results and Tech Reports

For each top-10 op, check if a faster config exists in sweep results.

**Also cross-reference with tech report recommendations**:
- YoloV4 report recommends LoFi math fidelity unless PCC drops
- Data formats report gives specific throughput multipliers for bfloat8_b vs bfloat16
- Sharding report gives guidance on when to switch sharding strategies

**Device-time source (Req 4)**: Rank ops SOLELY by the tracy `ops_perf_results_*.csv` DEVICE
TIME (ns) column. Do NOT derive device time from theoretical hardware limits, from FLOPS, or
from any efficiency ratio; do NOT report estimated or projected timings. Where no tracy data
exists, the answer is "profile first via tracy", NOT an estimate.

## Step 4 -- Generate Recommendations

**A1 Compliance Check**: Before recommending optimizations, verify the model uses pure TTNN
forward. If torch fallbacks are detected in the profile, flag them as PRIORITY FIX items
before any config optimization.

Save to `tests/models/<model_name>/perf_results/recommendation.json`:

```json
{
  "device_arch": "<arch>",
  "model_name": "<model>",
  "tt_metal_commit": "<40-char hash from preamble>",
  "a1_compliance": true,
  "torch_fallbacks": [],
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
      "how_to_apply": "In register_modules() or build_module_dict(), map nn.Linear to TTNNLinearLLama instead of TTNNLinear",
      "tech_report_reference": "YoloV4-TTNN section 2.3: data-type optimization"
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
| No sweep data available | Recommendations based on tech report heuristics |
| All ops roughly equal time | Model is well-balanced; focus on total time |
| torch fallbacks in profile | Flag as A1 violation; recommend fixing before optimizing |
