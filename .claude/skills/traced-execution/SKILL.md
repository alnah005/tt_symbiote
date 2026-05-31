---
name: traced-execution
description: Generate and run traced-mode tests exercising the TracedRun warm-up/capture/replay lifecycle. Validates PCC consistency between NORMAL and TRACED modes. Creates separate test files (does not modify existing tests).
---

# Traced Execution

Set up trace capture and replay for a model test for deterministic, low-overhead execution.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/capabilities/<model_name>/test_modeling_<model_name>.py`

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.
  - `tests/models/` does NOT exist. Never create files there.

**Pure TTNN forward**: ALL `TTNNModule.forward()` methods must use pure `ttnn.*` ops only.
  No `torch.*` calls in the compute path. This is ESPECIALLY critical for traced execution,
  because torch ops cannot be captured by the TTNN trace system. Any torch.* in forward()
  WILL break trace capture.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.

**License headers**: Every generated `.py` file must start with:
  ```python
  # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
  # SPDX-License-Identifier: Apache-2.0
  ```

**PCC assertions**: Use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.

**Config system**: The typed config system does NOT exist yet.

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

Key reports for traced-execution (read FIRST):
1. `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` --
   Metal Trace API details (section 1), 2CQ (section 2), combined patterns (section 3).
   This is the primary technical reference for trace implementation.
2. `ttnn/TTNN-model-bringup.md` -- Section 4.3: Trace and 2CQ
3. `ttnn/graph-tracing.md` -- TTNN graph tracing API
4. `ttnn/operation-tracing.md` -- TTNN operation tracing
5. `memory/allocator.md` -- Trace region memory requirements
6. `LLMs/llms.md` -- LLM-specific trace patterns

Read ALL remaining reports after these priority ones.

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
ls "$TT_METAL_HOME/models/tt_dit/utils/" 2>/dev/null
```

**Key reference implementations for traced execution** (study these carefully):

1. **`tt_transformers/tt/generator.py`** -- The canonical trace implementation for LLMs.
   Shows the complete 3-phase pattern:
   - Lines ~88-166: `warmup_model_prefill()` -- warmup phase runs forward to compile kernels
   - Lines ~167-209: `_capture_trace_prefill()` -- first runs forward normally for compilation,
     then runs again inside `ttnn.begin_trace_capture()` / `ttnn.end_trace_capture()` to capture
   - Lines ~248-271: `_prefill_forward_trace()` -- copies new inputs to trace buffer, then
     calls `ttnn.execute_trace()` for replay
   - Key observation: the generator manages per-sequence-length traces via `trace_id_prefill`
     dict and `trace_key` lookups

2. **`tt_dit/utils/tracing.py`** -- A clean, reusable `Tracer` class that encapsulates the
   3-phase lifecycle:
   - First call: runs function once to compile, then runs again under
     `ttnn.begin_trace_capture()`/`ttnn.end_trace_capture()` to capture
   - Subsequent calls: copies new inputs via `_update_input()`, then calls
     `ttnn.execute_trace()` for replay
   - This is the most readable reference for understanding the trace API

3. **Tech report section 1.3.1** (`AdvancedPerformanceOptimizationsForModels`):
   Shows the 3-step pattern with persistent DRAM input:
   ```python
   # Step 1: First run to compile
   output = run_model(input)
   # Step 2: Capture trace
   tid = ttnn.begin_trace_capture(device, cq_id=0)
   output = run_model(input)
   ttnn.end_trace_capture(device, tid, cq_id=0)
   # Step 3: Replay trace
   ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
   ```

## Plan-Verify-Execute Loop

This skill follows a mandatory loop structure. If the loop fails 5 times, report failure to the caller.

### PLAN Phase
1. Collect inputs (model name, test file, trace_region_size, comparison mode)
2. Read the existing test file to understand model setup
3. Verify the model has pure TTNN forward paths (no torch ops that would break tracing)
4. Draft the traced test file

### VERIFY Phase (no hardware, no user approval needed)
1. Verify the base PCC test exists: `test -f tests/capabilities/<model_name>/test_modeling_<model_name>.py`
2. Verify all imports resolve: `python -c "from tt_symbiote.core.run_config import TracedRun"`
3. Verify no `torch.*` calls in any model `forward()` methods (torch ops break trace capture):
   ```bash
   grep -n "torch\." src/tt_symbiote/models/<model_name>/modeling_<model_name>.py | grep -v "import\|#\|preprocess_weights\|from_torch\|__init__"
   ```
4. Verify `@run_on_devices` guards are present (modules without guards fall back to torch, breaking trace)
5. If ANY verification fails, return to PLAN with failure details and re-plan

### EXECUTE Phase (only after VERIFY passes)
Write the traced test file and run the tests.

## Background: TracedRun Lifecycle

`TracedRun` at `run_config.py:844` extends `LightweightRun` (which extends `NormalRun`).

Three phases, handled automatically by TracedRun when run mode is TRACED:
1. **Warm-up** (first forward call in TRACED mode): Normal execution, key added to `_warmup_keys`.
   This compiles kernels and primes JIT, CCL, and device memory allocator.
2. **Capture** (second forward call with same cache key): `ttnn.begin_trace_capture()` -> forward -> `ttnn.end_trace_capture()`.
   The system is already in steady state from warmup.
3. **Replay** (third+ forward calls): `_copy_inputs_to_trace_buffer()` then `ttnn.execute_trace()`.
   Near-zero host dispatch overhead.

**CRITICAL**: ALL 3 PHASES OCCUR IN TRACED MODE. You do NOT switch between NORMAL and TRACED
mode. `set_run_mode()` has an assertion that PREVENTS changing the mode after it is set:
```python
assert _current_run_mode is None or _current_run_mode == mode, \
    "Run mode has already been set and cannot be changed."
```
TracedRun internally handles the warmup phase (first call runs normally without trace capture),
then capture (second call), then replay (third+ calls). The mode stays TRACED throughout.

To run in TRACED mode, either:
- Set `TT_SYMBIOTE_RUN_MODE=TRACED` environment variable before the test
- Call `set_run_mode("TRACED")` once before any model forward calls

`@trace_enabled` is a **class decorator** (NOT a context manager):
```python
from tt_symbiote.core.run_config import trace_enabled

@trace_enabled
class TTNNMyModule(TTNNModule):
    ...
```

## Step 1 -- Collect Inputs (ASK the user)

1. **Model name**: For locating test files
2. **Test file to base on**: Path to existing pcc-test-gen test (typically Tier 4)
3. **trace_region_size**: Memory for trace storage (default: 200000000 ~200MB)
4. **Compare PCC between NORMAL and TRACED modes?** (default: Yes)

## Step 2 -- Verify Prerequisites

- pcc-test-gen tests exist and pass in NORMAL mode
- Verify module's `@run_on_devices` guard includes target architecture
  (modules without guards fall back to torch, breaking trace)
- **A1 CHECK**: Verify ALL forward() methods in model modules are pure TTNN.
  If any matches found, trace capture WILL FAIL. Fix model code first.

## Step 3 -- Generate Traced Test File

Create `tests/capabilities/<model_name>/test_traced_<model_name>.py` (SEPARATE from pcc-test-gen tests):

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Traced execution tests for <model_name>.

Exercises the TracedRun 3-phase lifecycle:
1. Warm-up (first forward in TRACED mode -- runs normally, compiles kernels)
2. Trace capture (second forward -- records op sequence)
3. Trace replay (third+ forward -- replays recorded trace)

IMPORTANT: ALL 3 PHASES RUN IN TRACED MODE.
TracedRun internally manages the phase transitions via _warmup_keys and _trace_cache.
set_run_mode() CANNOT be called twice -- it asserts that the mode has not been set.
The env var TT_SYMBIOTE_RUN_MODE=TRACED controls mode selection for the whole process.
"""

import os
import pytest
import torch
import ttnn
from tt_symbiote.core.run_config import TracedRun
from tt_symbiote.utils.device_management import set_device
from tests.capabilities.pcc_utils import assert_pcc

MODEL_ID = "<hf_model_id>"

# Set TRACED mode via environment variable (must be set before any model forward calls)
os.environ["TT_SYMBIOTE_RUN_MODE"] = "TRACED"


@pytest.mark.parametrize("device_params", [{
    "trace_region_size": 200000000,
    "num_command_queues": 1,
}], indirect=True)
def test_traced_3phase(mesh_device):
    """Validate 3-phase traced execution lifecycle.

    All 3 calls happen in TRACED mode. TracedRun handles phase transitions:
    - Call 1: warmup (key added to _warmup_keys, normal forward runs)
    - Call 2: capture (key found in _warmup_keys, trace captured)
    - Call 3: replay (key found in _trace_cache, trace replayed)
    """
    torch.set_grad_enabled(False)
    # ... model loading and setup ...
    set_device(model, mesh_device)

    # All 3 phases in TRACED mode -- TracedRun manages transitions internally
    # Phase 1: Warmup -- TracedRun detects key not in _warmup_keys, runs normal forward
    warmup_out = model(warmup_input)

    # Phase 2: Trace capture -- TracedRun detects key in _warmup_keys but not _trace_cache
    capture_out = model(capture_input)

    # Phase 3: Trace replay -- TracedRun detects key in _trace_cache, replays trace
    replay_out = model(replay_input)

    # Validate PCC across phases
    assert_pcc(capture_out, warmup_out, threshold=0.999, msg="capture_vs_warmup")
    assert_pcc(replay_out, warmup_out, threshold=0.999, msg="replay_vs_warmup")

    # Cleanup
    TracedRun.release_all()


@pytest.mark.parametrize("device_params", [{
    "trace_region_size": 200000000,
    "num_command_queues": 1,
}], indirect=True)
def test_traced_multiple_replays(mesh_device):
    """Validate consistency across multiple trace replays.

    After warmup and capture, replays should produce identical results.
    """
    torch.set_grad_enabled(False)
    # ... model loading and setup ...
    set_device(model, mesh_device)

    # Phase 1: Warmup (TRACED mode, TracedRun runs normal forward)
    model(warmup_input)

    # Phase 2: Capture (TRACED mode, TracedRun captures trace)
    model(capture_input)

    # Phase 3+: Multiple replays (TRACED mode, TracedRun replays trace)
    outputs = []
    for _ in range(5):
        out = model(replay_input)
        outputs.append(out)

    # All replay outputs should match
    for i in range(1, len(outputs)):
        assert_pcc(outputs[i], outputs[0], threshold=0.999, msg=f"replay_{i}_vs_replay_0")

    TracedRun.release_all()
```

## Step 4 -- Run and Compare

```bash
# Run the same model test in NORMAL mode first (baseline PCC reference)
TT_SYMBIOTE_RUN_MODE=NORMAL pytest tests/capabilities/<model_name>/test_modeling_<model_name>.py -x -s

# Run traced tests (uses TRACED mode set in the test file via os.environ)
pytest tests/capabilities/<model_name>/test_traced_<model_name>.py -x -s
```

## Step 5 -- Report PCC Delta

Compare NORMAL vs TRACED PCC values. Acceptable delta: < 0.001 (within 0.1%).

## Error Handling

| Problem | Resolution |
|---------|-----------|
| `trace_region_size` too small | Increase to 400000000+ |
| Trace capture OOM | Reduce model size or batch |
| PCC divergence between modes | Check for non-deterministic ops (dropout must be disabled) |
| Fallback modules in trace | Modules without device guards fall back to torch, breaking trace |
| `set_run_mode` assertion error | You tried to switch modes mid-process. Use env var `TT_SYMBIOTE_RUN_MODE` instead, set before any model calls. All 3 phases must be in the SAME mode (TRACED). |
| torch.* in forward (A1 violation) | Fix model code -- trace REQUIRES pure TTNN forward |
