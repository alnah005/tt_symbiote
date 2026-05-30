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

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.

**License headers**: Every generated `.py` file must start with:
  ```python
  # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
  # SPDX-License-Identifier: Apache-2.0
  ```

**PCC assertions**: Use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.

**Config system**: The typed config system does NOT exist yet.

## Background: TracedRun Lifecycle

`TracedRun` at `run_config.py:844` extends `LightweightRun` (which extends `NormalRun`).

Three phases, handled automatically by TracedRun:
1. **Warm-up** (first forward): Normal execution, key added to `_warmup_keys`
2. **Capture** (second forward): `ttnn.begin_trace_capture()` -> forward -> `ttnn.end_trace_capture()`
3. **Replay** (third+ forward): `_copy_inputs_to_trace_buffer()` then `ttnn.execute_trace()`

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

## Step 3 -- Generate Traced Test File

Create `tests/capabilities/<model_name>/test_traced_<model_name>.py` (SEPARATE from Skill 2 tests):

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Traced execution tests for <model_name>.

Exercises the TracedRun 3-phase lifecycle:
1. Warm-up (first forward)
2. Trace capture (second forward)
3. Trace replay (third+ forward)
"""

import pytest
import torch
import ttnn
from tt_symbiote.core.run_config import DispatchManager, TracedRun, set_run_mode
from tt_symbiote.utils.device_management import set_device
from tests.capabilities.pcc_utils import assert_pcc

MODEL_ID = "<hf_model_id>"


@pytest.mark.parametrize("device_params", [{
    "trace_region_size": 200000000,
    "num_command_queues": 1,
}], indirect=True)
def test_traced_3phase(mesh_device):
    """Validate 3-phase traced execution lifecycle."""
    torch.set_grad_enabled(False)
    # ... model loading and setup ...
    set_device(model, mesh_device)

    # Phase 1: Warmup (NORMAL mode)
    set_run_mode("NORMAL")
    warmup_out = model(warmup_input)

    # Phase 2: Trace capture (TRACED mode)
    set_run_mode("TRACED")
    capture_out = model(capture_input)  # First call captures

    # Phase 3: Trace replay
    replay_out = model(replay_input)  # Replays captured trace

    # Validate PCC on each phase
    assert_pcc(capture_out, warmup_out, threshold=0.999, msg="capture_vs_warmup")
    assert_pcc(replay_out, warmup_out, threshold=0.999, msg="replay_vs_warmup")

    # Cleanup
    TracedRun.release_all()


@pytest.mark.parametrize("device_params", [{
    "trace_region_size": 200000000,
    "num_command_queues": 1,
}], indirect=True)
def test_traced_multiple_replays(mesh_device):
    """Validate consistency across multiple trace replays."""
    torch.set_grad_enabled(False)
    # ... model loading and setup ...
    set_device(model, mesh_device)

    set_run_mode("NORMAL")
    model(warmup_input)

    set_run_mode("TRACED")
    model(capture_input)  # Capture

    # Multiple replays
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
# Run in NORMAL mode first (baseline)
TT_SYMBIOTE_RUN_MODE=NORMAL pytest tests/capabilities/<model_name>/test_traced_<model_name>.py -x -s

# Run traced tests
pytest tests/capabilities/<model_name>/test_traced_<model_name>.py -x -s
```

## Step 5 -- Report PCC Delta

Compare NORMAL vs TRACED PCC values. Acceptable delta: < 0.001 (within 0.1%).

## Error Handling

| Problem | Resolution |
|---------|------------|
| `trace_region_size` too small | Increase to 400000000+ |
| Trace capture OOM | Reduce model size or batch |
| PCC divergence between modes | Check for non-deterministic ops (dropout must be disabled) |
| Fallback modules in trace | Modules without device guards fall back to torch, breaking trace |
| `_TRACE_RUNNING` assertion | Weights must be preprocessed before TRACED mode (run NORMAL warmup first) |
