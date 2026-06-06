---
name: model-bringup
description: Fully autonomous deep-work driver for 0-to-100 model bring-up on Tenstorrent hardware. Embeds the user's decision profile, drives each stage as a plan-evaluate-execute cycle with automatic re-planning on failure, tracks all state in bringup_status.json. Requires only a HuggingFace model ID to run to completion.
---

# Autonomous Model Bring-Up Orchestrator

End-to-end autonomous orchestration for bringing up a new HuggingFace model on Tenstorrent hardware. This skill IS the deep-work driver -- it contains the plan-evaluate-execute-replan loop inline and spawns fresh sub-agents as executors for each stage.

## User Decision Profile

Every decision this orchestrator makes is governed by the following encoded profile. The orchestrator NEVER stops to ask the user. All decisions are made automatically per this table, and every decision is logged with its rationale in `bringup_status.json`.

| Decision | Encoded Answer | Implication |
|----------|---------------|-------------|
| PCC failure handling | NORMAL and TRACED modes only | No DPL/SEL -- those are beta. Debug by isolating modules manually via targeted Tier 1/2/3 re-runs. |
| Perf vs accuracy priority | Functionality first, then speed | Phase 1: get real samples producing correct output. Phase 2: optimize. |
| Min accuracy bar | Semantically correct output | Not just PCC numbers -- actual model output must make sense for the model type. |
| Planner count per stage | 1 planner (fast iteration) | Speed over diversity. One planner, one evaluator, one executor per stage. |
| Custom code models | Auto-proceed with trust_remote_code=True | Never pause for custom code confirmation. Always pass trust_remote_code=True. |
| Autonomy level | Fully autonomous | Never stop to ask user. Make all decisions. Log rationale. Report at end. |
| Interview per stage | Full context to planner every time | Each planner receives CLAUDE.md + the relevant skill's SKILL.md + curated tech reports + bringup_status.json. This is the autonomous equivalent of a "full interview" -- the planner gets complete context on every invocation, never a reduced or cached summary. |
| Validation samples | Model-specific defaults | Orchestrator picks sensible inputs per model type (see Validation Samples table below). |
| Context for planners | CLAUDE.md + skill SKILL.md + key tech reports + bringup_status.json | No raw deep-plan files, no prior iteration plans, no full 49-report dump. |
| Missing ops | Integration if reusable, inline if one-off | Create integration module only if 2+ models would use the op. Otherwise inline in modeling file. |
| Deep-work granularity | One iteration per skill invocation | Each skill = separate plan-evaluate-execute cycle. |
| Model directory exists | Extend (do not overwrite) | If model directory already exists, assess what is present and continue from current state. |
| Integration path | Recipe/Auto API if model is in HF transformers; Manual if custom | Auto-detect based on whether the model_type resolves in transformers.models. |
| Retry limit | 3 retries per stage | After 3 failures of the same stage, log the failure, skip to next feasible stage, and note in final report. |
| Tier 4 validation | Generate text/tokens and check semantic coherence | Not just PCC -- actually generate output and verify it is not garbage. |

### Model-Specific Validation Samples

The orchestrator automatically selects validation inputs based on model architecture:

| Model Type | Detection Heuristic | Validation Input |
|-----------|-------------------|-----------------|
| Causal LM (decoder-only) | config.is_decoder=True or "CausalLM" in class name | `"The capital of France is"` -- expect coherent continuation |
| Seq2Seq (encoder-decoder) | config.is_encoder_decoder=True | `"Translate English to French: Hello world"` |
| Vision model | "image" in config.model_type or "ViT" in class name | Load a sample image from torchvision or generate a random tensor of expected image size |
| Vision-Language | "vl" or "visual" in model_type | Text: `"Describe this image:"` + sample image tensor |
| Embedding model | "Embedding" in class name | `"This is a test sentence for embedding."` |
| Default fallback | None of the above match | `"Hello, world. This is a test of the model."` |

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/models/<model_name>/Tier4/test_modeling_<model_name>.py` (tier-dir layout:
    Tier1=ops, Tier2=composites, Tier3=decoder, Tier4=full model; each tier dir has an `__init__.py`;
    ROOT keeps `test_config.json`/`shapes.json`/`op_map.json`; tier files load ROOT artifacts via
    `Path(__file__).parent.parent`)

**Test location**: Per-model RICH tests go under `tests/models/<model_name>/` (e2e-traced
  correct); partial-TTNN bring-ups under `tests/experimental/<model_name>/` (MINIMAL floor:
  `__init__.py` + `test_config.json`). Write to `tests/models/` ONLY once e2e traced correctness
  is proven, else `tests/experimental/`. The old per-model capabilities tree was removed.
  Every per-model dir carries a `test_config.json` (5 keys: `tt_metal_commit`, `device_arch`,
  `pcc_threshold`, `hf_model_id`, `hf_revision`). Promotion experimental→models requires PROVEN
  e2e traced correctness (all RICH-tier PCC green in TRACED at 0.99 default / 0.999 bring-up +
  traced-mode PCC matches NORMAL + semantic validation) AND upgrade to the RICH floor
  (`shapes.json`, `op_map.json`, `test_ops/composites/decoder/modeling/traced_<model_name>.py`)
  with a populated `test_config.json` (pinned `tt_metal_commit`); use `git mv` to preserve history.

**Tracy-only device time (Req 4)**: Device time is sourced SOLELY from the tracy
  `ops_perf_results_*.csv` DEVICE TIME (ns) column. Never estimate, project, or compute device
  time from theoretical hardware limits, from FLOP counts, or from any efficiency ratio.
  `GEMM_FLOPS/GEMM_FLOPS.md` is reading material only.

**Decorator-only tracing (Req 5)**: Enable tracing SOLELY via the `@trace_enabled` class
  decorator on the ACTUAL trace unit, checked at runtime via `is_trace_enabled(<unit>)`. Never
  use ad-hoc instance flags (e.g. `self._trace_enabled`). Never decorate a parent/wrapper module
  just to flag a child — check `is_trace_enabled(self.<child>)` instead.

**Pure TTNN forward**: ALL `TTNNModule.forward()` methods MUST use pure `ttnn.*` ops only.
  No `torch.*` calls in the compute/forward path. This is mandatory and non-negotiable.
  - `forward()`: Pure `ttnn.*` ops only (ttnn.linear, ttnn.matmul, ttnn.add, ttnn.multiply,
    ttnn.silu, ttnn.reshape, ttnn.deallocate, ttnn.to_memory_config, etc.)
  - `preprocess_weights_impl()`: May use PyTorch for weight transformation
  - `from_torch()`: May use PyTorch for weight extraction
  Reference: `$TT_METAL_HOME/models/tt_transformers/tt/mlp.py` forward().

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.
  - Import: `from tt_symbiote.core.module import run_on_devices, DeviceArch`
  - Default: `@run_on_devices(DeviceArch.T3K)`

**License headers**: Every generated `.py` file must start with:
  ```python
  # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
  # SPDX-License-Identifier: Apache-2.0
  ```

**PCC assertions**: Use `assert_pcc()` from `tests/shared/pcc_utils.py`.
  NEVER rely on `compare_fn_outputs()`.

**Deprecated API**: Never use `register_module_replacement_dict()`.
  Use `register_modules()` from `tt_symbiote.utils.module_replacement`.

**Import conventions**:
  - Integration modules: `from tt_symbiote.modules.ttnn_<module> import TTNN<Class>`
  - Core: `from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch`
  - Tier 1-3 tests: `from tt_symbiote.utils.device_management import set_device`
  - Tier 4 (Auto API): `from tt_symbiote import AutoModelForCausalLM, set_device`

**Device/mesh_device fixtures**: Provided by the `ttnn` pytest plugin, NOT by tt_symbiote.
  Do NOT define these fixtures.

**Config system**: The typed config system (ModuleConfig, DtypeConfig, etc.) does NOT exist.
  Use `_model_config: dict` via `set_model_config()`, and subclass-based overrides.

**Tuning Workflow (Req 6 — functional-first, then bottom-up)**:
  - **Phase A (functional-first)**: get ALL tier PCC green (0.99 default / 0.999 bring-up) plus
    semantic validation BEFORE any performance tuning. This is a HARD precondition — do NOT begin
    Phase B until Phase A passes.
  - **Phase B (bottom-up, leaves-first)**: profile via tracy, then tune a module ONLY IF its tracy
    device-time % exceeds the descent gate (knob `phase_b_descent_gate_pct`, default **5%**). After
    tuning each module, re-validate PCC for that module and the affected tiers BEFORE ascending to
    its parent; roll back the change on any PCC regression; re-profile via tracy. All numbers come
    from the tracy `ops_perf_results_*.csv` DEVICE TIME column — never estimates.

**Tech-Report Reading Gate (Req 8)**: BEFORE writing or replacing ANY new TTNN module, append an
  additive `references_read` record to `bringup_status.json`. The orchestrator is BLOCKED until
  this is logged; the entry is additive and never removes existing keys:
  ```json
  "references_read": [{"tt_metal_commit":"<hash>","timestamp":"<iso8601>",
    "tech_reports":["ttnn/TTNN-model-bringup.md","..."],
    "reference_impls":["models/tt_transformers/tt/attention.py","..."],
    "consulted_paths":["models/demos/.../..."]}]
  ```

**HuggingFace**: Always pass `trust_remote_code=True`. (Decision profile: auto-proceed.)

**TT_METAL_COMMIT Hash**: Every `modeling_<model_name>.py` MUST contain:
  ```python
  TT_METAL_COMMIT = '<full 40-char git hash>'
  ```
  Captured from: `git -C "$TT_METAL_HOME" rev-parse HEAD`

## Amendment Composition Matrix

How the 7 amendments apply to each skill when invoked by this orchestrator:

| Amendment | model-bringup | pcc-test-gen | op-sweep | config-optimize-all | config-optimize-module | tracy-profiling | perf-analysis | traced-execution |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A1 Pure TTNN | ENFORCE | ENFORCE | ENFORCE | ENFORCE | ENFORCE | VERIFY | VERIFY | VERIFY |
| A2 Explore $TT_METAL_HOME | FULL | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| A3 Read Tech Reports | FULL | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| A4 Dynamic Orchestration | PRIMARY | - | - | - | - | - | - | - |
| A5 Fresh Sub-Agent | SPAWNER | TARGET | TARGET | TARGET | TARGET | TARGET | TARGET | TARGET |
| A6 Plan-Verify-Execute | FULL | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| A7 Commit Hash | ENFORCE | ENFORCE | - | ENFORCE | ENFORCE | - | - | - |

## Step 0 -- Mandatory Exploration Preamble

**This step is NON-NEGOTIABLE. Complete it IN FULL before proceeding.**

### 0a. Read Key Tech Reports

Read the tech reports relevant to orchestrator-level decisions from `$TT_METAL_HOME/tech_reports/`. The orchestrator reads a CURATED subset (not all 49). Each sub-agent reads reports relevant to its own skill during its own Step 0.

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"

# Priority reports for the orchestrator:
cat "$TT_METAL_HOME/tech_reports/ttnn/TTNN-model-bringup.md"
cat "$TT_METAL_HOME/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md"
cat "$TT_METAL_HOME/tech_reports/data_formats/data_formats.md"
cat "$TT_METAL_HOME/tech_reports/tensor_sharding/tensor_sharding.md"
cat "$TT_METAL_HOME/tech_reports/tensor_layouts/tensor_layouts.md"
cat "$TT_METAL_HOME/tech_reports/memory/allocator.md"
cat "$TT_METAL_HOME/tech_reports/LLMs/llms.md"
cat "$TT_METAL_HOME/tech_reports/FlashAttention/FlashAttention.md"
cat "$TT_METAL_HOME/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md"
```

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"

# 1. tt_transformers: THE canonical LLM pattern
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
# Key files: model.py, attention.py, mlp.py, decoder.py, model_config.py, generator.py

# 2. tt_dit: DiT model patterns (if bringing up a diffusion model)
ls "$TT_METAL_HOME/models/tt_dit/" 2>/dev/null

# 3. tt_cnn: CNN model patterns (if bringing up a CNN)
ls "$TT_METAL_HOME/models/tt_cnn/tt/" 2>/dev/null

# 4. demos: Production model examples
ls "$TT_METAL_HOME/models/demos/" 2>/dev/null
```

**Extract patterns relevant to the model being brought up**:
- For LLMs: Study tt_transformers/{model.py, attention.py, mlp.py, decoder.py}
- For CNNs: Study tt_cnn/ and its bringup guide
- For DiT: Study tt_dit/
- Note how each model type uses pure ttnn forward()

### 0c. Capture TT_METAL_COMMIT Hash

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
TT_METAL_COMMIT=$(git -C "$TT_METAL_HOME" rev-parse HEAD)
echo "TT_METAL_COMMIT=$TT_METAL_COMMIT"
```

This hash will be embedded in the generated modeling file.

### 0d. Structure Preflight

Before scaffolding, run the check-only structure auditor SCOPED to this model to learn which
tier dirs/files already exist and to seed the scaffold planner with concrete failure codes. The
script is check-only -- it never mutates the filesystem.

```bash
REPORT=$(python scripts/check_tier_structure.py --model "$MODEL" --skip-lint --format json 2>/dev/null) || true
python - "$MODEL" "$REPORT" <<'PY'
import json, sys
name, raw = sys.argv[1], sys.argv[2]
try: r = json.loads(raw)
except Exception: print("CONTRACT_UNAVAILABLE"); sys.exit(0)
if r.get("schema_version") != 1: print("CONTRACT_UNKNOWN_VERSION"); sys.exit(0)
fails = [f for grp in r["checks"].values() for f in grp.get("failures", [])
         if name in f.get("path","")]
print("STRUCTURE_OK" if not fails else "MISSING:" + ",".join(f["code"] for f in fails))
PY
```

Parse the FROZEN report contract (`schema_version: 1`; see pcc-test-gen for the closed `code`
vocabulary). Key ONLY off `summary.errors` and `failures[].code` -- never off English text. Seed
`bringup_status.json.structure_preflight` with the parsed result and feed each `failures[].code`
(e.g. `MISSING_TIER_DIR`, `MISSING_CONFIG`) to the scaffold planner so scaffolding creates the
missing `Tier1..Tier4/` dirs + ROOT `test_config.json`. A `--model`-scoped preflight does NOT
exercise the whole-repo invariants or check-mode lint -- it is a preflight, not full validation.

A FINAL structure gate runs AFTER `traced_execution` completes, also `--model`-scoped (so an
unrelated dir cannot block this bring-up): require `summary.errors == 0` for THIS model before
marking the bring-up complete.

## Architecture: Autonomous Deep-Work Driver

This skill IS the deep-work loop. It does not "call" the deep-work skill -- it implements the pattern inline. Each stage of model bring-up is one deep-work cycle:

```
For each stage:
  1. PLAN:     Spawn 1 planner sub-agent
               Input: CLAUDE.md + relevant skill SKILL.md + key tech reports + bringup_status.json
               Output: An execution plan for this stage
  2. EVALUATE: Spawn 1 evaluator sub-agent
               Input: The plan from step 1 + skill SKILL.md + bringup_status.json
               Output: Approved plan (or rejection with amendments)
               If rejected: Return to PLAN with evaluator feedback (max 2 rejections per retry)
  3. EXECUTE:  Spawn 1 executor sub-agent with the approved plan
               Input: Approved plan + full skill SKILL.md + model context
               Output: Artifacts + pass/fail result
  4. ASSESS:   If execution failed:
               - Increment retry count in bringup_status.json
               - If retries < 3: Return to PLAN with failure context
               - If retries >= 3: Log failure, skip stage, continue to next feasible stage
  5. UPDATE:   Write results to bringup_status.json
               Log decision rationale in decision_log
```

### Nested Retry Semantics

The deep-work cycle has two nested retry loops. Their interaction is defined precisely here:

```
OUTER LOOP: Stage retries (max 3)
  Each outer retry represents one full attempt at a stage: plan -> evaluate -> execute.
  The retry counter in bringup_status.json tracks outer retries only.

  INNER LOOP: Evaluator rejections (max 2 rejections = 3 plan attempts per outer retry)
    Within a single outer retry, the evaluator may reject the planner's output up to 2 times.
    On each rejection, the planner re-plans with evaluator feedback.
    After 2 rejections (3 total plan attempts), the evaluator MUST accept the best plan
    available and annotate it with residual concerns.

  INTERACTION:
    - Evaluator rejections are INTERNAL to a single outer retry. They do NOT consume
      outer retry budget.
    - After a plan is approved (by acceptance or by exhausting inner retries), execution
      proceeds. If execution FAILS, the outer retry counter increments.
    - A single stage can therefore produce up to 3 outer retries x 3 plan attempts each
      = 9 total plan attempts before the stage is skipped.
    - Evaluator rejections reset to 0 at the start of each outer retry.

  EXAMPLE FLOW:
    Outer retry 0:
      Plan attempt 1 -> Evaluator rejects (rejection 1)
      Plan attempt 2 -> Evaluator approves
      Execution -> FAILS
      (outer retry counter: 0 -> 1)
    Outer retry 1:
      Plan attempt 1 -> Evaluator approves
      Execution -> FAILS
      (outer retry counter: 1 -> 2)
    Outer retry 2:
      Plan attempt 1 -> Evaluator rejects (rejection 1)
      Plan attempt 2 -> Evaluator rejects (rejection 2, max reached)
      Plan attempt 3 -> Evaluator MUST accept (annotates residual concerns)
      Execution -> PASSES
      Stage complete!
```

### What Planners Receive

Each planner sub-agent is given ONLY:
- The full content of `CLAUDE.md` (repository conventions)
- The full content of the relevant skill's `SKILL.md` (the skill being invoked -- read explicitly by the orchestrator before composing the prompt; see Main Orchestration Loop step 3a)
- Key tech reports from `$TT_METAL_HOME/tech_reports/` (curated per stage, see Tech Report Selection table)
- The current `bringup_status.json` (state of the bring-up)
- Model context: HF model ID, model_name, device_arch, TT_METAL_COMMIT, integration_path

Planners do NOT receive:
- Raw deep-plan files from `/tmp/deep-plan-*/`
- Prior deep-work iteration plans from `./deep-work/`
- The full 49 tech report dump (only stage-relevant reports)
- Other skills' SKILL.md files (only the one being planned for)

### Tech Report Selection Per Stage

| Stage | Key Tech Reports (passed to planner) |
|-------|--------------------------------------|
| scaffold | TTNN-model-bringup.md, LLMs/llms.md, data_formats.md, tensor_layouts/tensor_layouts.md |
| pcc_test_gen | TTNN-model-bringup.md, data_formats.md, tensor_sharding.md, FlashAttention.md |
| op_sweep | data_formats.md, GEMM_FLOPS.md, memory/allocator.md, tensor_sharding.md, tensor_layouts/tensor_layouts.md, YoloV4-TTNN/yolov4.md |
| tracy_profiling | MetalProfiler/metal-profiler.md, AdvancedPerformanceOptimizations.md, GEMM_FLOPS.md |
| perf_analysis | GEMM_FLOPS.md, AdvancedPerformanceOptimizations.md, data_formats.md, Saturating_DRAM_bandwidth.md |
| config_optimize | data_formats.md, GEMM_FLOPS.md, tensor_sharding.md, memory/allocator.md, YoloV4-TTNN/yolov4.md |
| traced_execution | AdvancedPerformanceOptimizations.md, TTNN-model-bringup.md, graph-tracing.md, operation-tracing.md |

**Note on tensor_layouts/tensor_layouts.md**: This report covers TILE vs ROW_MAJOR layout fundamentals. It is assigned to scaffold (choosing initial tensor layouts for module I/O) and op_sweep (sweep parameters include memory layout selection). The path is `$TT_METAL_HOME/tech_reports/tensor_layouts/tensor_layouts.md`.

## Dynamic Decision Tree

The orchestrator uses a dynamic decision tree -- NOT a fixed linear pipeline. At each iteration of the main loop, it reads `bringup_status.json` and determines the next action.

```
MAIN LOOP:
  |
  v
[Read bringup_status.json]
  |
  v
[Assess current state]
  |
  |-- No model directory exists
  |     --> Stage: SCAFFOLD
  |     Decision: "Model directory does not exist. Creating scaffold."
  |
  |-- Model directory exists, no bringup_status.json
  |     --> Create bringup_status.json by scanning existing artifacts
  |     --> Re-assess from the top
  |
  |-- scaffold:completed, pcc_test_gen:not_started
  |     --> Stage: PCC_TEST_GEN
  |     Decision: "Scaffold complete. Generating tiered PCC tests."
  |
  |-- pcc_test_gen:completed, Tier 1 failing
  |     --> Stage: PCC_TEST_GEN (retry)
  |     Decision: "Tier 1 tests failing. Re-running pcc-test-gen with failure context.
  |               Debug mode: isolate failing modules via targeted Tier 1 re-runs (NORMAL mode only)."
  |
  |-- pcc_test_gen:completed, Tier 1 passing, Tier 2 failing
  |     --> Stage: PCC_TEST_GEN (retry)
  |     Decision: "Tier 2 failing. Check integration class compatibility and from_torch signatures."
  |
  |-- pcc_test_gen:completed, Tier 1-2 passing, Tier 3 failing
  |     --> Stage: PCC_TEST_GEN (retry)
  |     Decision: "Tier 3 failing. Check decoder layer residual connections and norm placement."
  |
  |-- pcc_test_gen:completed, Tier 1-3 passing, op_sweep:not_started
  |     --> Stage: OP_SWEEP
  |     Decision: "Tier 1-3 tests pass. Running op parameter sweep."
  |
  |-- op_sweep:completed, tracy_profiling:not_started
  |     --> Stage: TRACY_PROFILING
  |     Decision: "Sweep complete. Profiling with tracy."
  |
  |-- tracy_profiling:completed, perf_analysis:not_started
  |     --> Stage: PERF_ANALYSIS
  |     Decision: "Profiling complete. Analyzing performance data."
  |
  |-- perf_analysis:completed, config_optimize:not_started
  |     --> Stage: CONFIG_OPTIMIZE
  |     Decision: "Analysis complete. Applying optimal configurations."
  |
  |-- config_optimize:completed, traced_execution:not_started
  |     --> Stage: TRACED_EXECUTION
  |     Decision: "Optimization complete. Setting up traced execution."
  |
  |-- All phases completed
  |     --> Run Tier 4 full model validation with semantic check
  |     --> Generate final summary report
  |
  |-- Any stage failed 3+ times
  |     --> Log: "Stage <X> failed 3 times. Skipping to next feasible stage."
  |     --> Decision: Skip failed stage, continue with next dependency-free stage
  |     --> Note in final report as known issue
```

At EACH decision point, the orchestrator:
1. Reads bringup_status.json
2. Evaluates the decision tree
3. Logs the decision with rationale in decision_log
4. Executes the deep-work cycle for the selected stage
5. Updates bringup_status.json with results
6. Returns to the top of the loop

## Phase 1: Model Discovery (Autonomous)

### Step 1 -- Collect Model Information

The only required input is the **HuggingFace model ID** (e.g., `meta-llama/Llama-3.1-8B`). Everything else is derived automatically.

```python
from transformers import AutoConfig
import importlib

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model_type = getattr(config, 'model_type', None)

try:
    importlib.import_module(f"transformers.models.{model_type}")
    model_name = model_type
    integration_path = "recipe"  # Auto-decision: Recipe path for standard HF models
except ImportError:
    model_name = model_type
    integration_path = "manual"  # Auto-decision: Manual path for custom models
```

**Autonomous decisions made here**:
- Model name: Derived from `config.model_type`. No confirmation needed.
- Custom code: Always proceeds with `trust_remote_code=True` (decision profile).
- Integration path: Recipe if model is in HF transformers; Manual if custom code.
- Device architecture: Default T3K.
- Existing directory: If exists, extend from current state (decision profile: never overwrite).

Log all decisions:
```json
{
  "timestamp": "<ISO>",
  "decision": "model_discovery",
  "reason": "model_type='llama' found in transformers.models. Using recipe integration path. Device: T3K.",
  "result": "pass"
}
```

### Step 2 -- Read Model Architecture

```python
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
```

Analyze architecture autonomously:
- Architecture summary (attention type, MLP type, norm type, MoE or not)
- Cross-reference with `$TT_METAL_HOME/models/tt_transformers/tt/`:
  - GQA attention? -> See `attention.py` n_local_kv_heads pattern
  - SwiGLU MLP? -> See `mlp.py` gate/up/down pattern
  - MoE? -> See `mixtral_moe.py` and `mixtral_mlp.py`
- Map to existing tt_symbiote integration modules
- Identify modules needing new implementations
- Auto-select initial run mode: NORMAL_WITH_FALLBACK
- Auto-select validation samples from Model-Specific Validation Samples table

## Phase 2: Scaffold -- Deep-Work Cycle

### Scaffold Planner Input

The planner sub-agent receives:
```
You are a planner for the SCAFFOLD stage of model bring-up.

## REPOSITORY CONVENTIONS
<full CLAUDE.md content>

## SKILL INSTRUCTIONS
<full content of .claude/skills/model-bringup/SKILL.md, read by orchestrator in step 3a>

## CURRENT STATE
<bringup_status.json content>

## MODEL CONTEXT
Model ID: <hf_model_id>
Model name: <model_name>
Device: T3K
TT_METAL_COMMIT: <commit_hash>
Integration path: <recipe|manual>
Architecture analysis: <from Step 2>

## KEY TECH REPORTS
<content of TTNN-model-bringup.md>
<content of LLMs/llms.md>
<content of data_formats.md>
<content of tensor_layouts/tensor_layouts.md>

## YOUR TASK
Produce a detailed plan for scaffolding the model directory, including:
1. Directory structure
2. Module class list with from_torch signatures
3. Recipe registration (if recipe path)
4. All file contents with license headers, TT_METAL_COMMIT, @run_on_devices decorators
5. Initial tensor layout decisions (informed by tensor_layouts tech report)
6. Verification steps (imports resolve, no torch in forward, etc.)
```

### Scaffold Evaluator

The evaluator checks:
1. All forward() skeletons are pure TTNN (no torch.* placeholders in forward body)
2. @run_on_devices is on every forward()
3. License headers are planned for every file
4. TT_METAL_COMMIT will be embedded in the modeling file
5. Integration path is consistent (recipe has @register_recipe, manual has register_modules)
6. __init__.py imports all necessary classes
7. _RECIPE_BEARING_SUBPACKAGES is updated (recipe path only)
8. Tensor layout choices are consistent with the tensor_layouts tech report

### Scaffold Executor

The executor sub-agent receives the approved plan and executes it:

```
You are an executor for the SCAFFOLD stage of model bring-up.

## SKILL INSTRUCTIONS
<full model-bringup SKILL.md -- conventions section only>

## APPROVED PLAN
<the evaluated and approved plan>

## CONTEXT
Model: <model_name>
Model ID: <hf_model_id>
Device: T3K
TT_METAL_COMMIT: <commit_hash>

## YOUR TASK
Execute the plan exactly. Create all files. Run verification commands.
Report: files created, verification results, any issues.
```

The executor creates:
1. `src/tt_symbiote/models/<model_name>/` directory
2. `modeling_<model_name>.py` with TTNN module classes
3. `__init__.py` with exports
4. Updates `models/__init__.py` for recipe path
5. `tests/models/<model_name>/` directory
6. `bringup_status.json` initialized

Verification commands:
```bash
python -c "from tt_symbiote.models.<model_name> import TTNN<ModelName>Model; print('Import OK')"

# If Recipe path:
python -c "
from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY
assert '<HFModelClass>ForCausalLM' in TT_MODEL_REGISTRY, 'Recipe not registered!'
print('Recipe registered')
"

# Verify TT_METAL_COMMIT is present
grep 'TT_METAL_COMMIT' src/tt_symbiote/models/<model_name>/modeling_<model_name>.py
```

### Scaffold Template

The modeling file follows this template (adjusted per model architecture):

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN bring-up for <HFModelClass>.

Model: <model_id>
Device target: T3K (8x Wormhole)
"""

import ttnn

from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.modules.ttnn_normalization import TTNNRMSNorm
from tt_symbiote.utils.module_replacement import register_modules

TT_METAL_COMMIT = '<full 40-char git hash from Step 0c>'


class TTNN<ModelName>Attention(TTNNModule):
    """TTNN implementation of <ModelName>Attention.
    Reference: $TT_METAL_HOME/models/tt_transformers/tt/attention.py
    """

    @classmethod
    def from_torch(cls, torch_layer):
        new_module = cls()
        new_module._fallback_torch_layer = torch_layer
        return new_module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        raise NotImplementedError("Implement TTNN attention forward using only ttnn.* ops")


class TTNN<ModelName>MLP(TTNNModule):
    """TTNN implementation of <ModelName>MLP.
    Reference: $TT_METAL_HOME/models/tt_transformers/tt/mlp.py
    Pattern: gate_proj -> silu -> multiply(up_proj) -> down_proj (all ttnn ops)
    """

    @classmethod
    def from_torch(cls, torch_layer):
        new_module = cls()
        new_module._fallback_torch_layer = torch_layer
        return new_module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        raise NotImplementedError("Implement TTNN MLP forward using only ttnn.* ops")


class TTNN<ModelName>DecoderLayer(TTNNModule):
    """TTNN implementation of a single <ModelName> decoder layer.
    Reference: $TT_METAL_HOME/models/tt_transformers/tt/decoder.py
    """

    @classmethod
    def from_torch(cls, torch_layer):
        new_module = cls()
        new_module._fallback_torch_layer = torch_layer
        new_module.self_attn = TTNN<ModelName>Attention.from_torch(torch_layer.self_attn)
        new_module.mlp = TTNN<ModelName>MLP.from_torch(torch_layer.mlp)
        return new_module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        raise NotImplementedError("Implement decoder layer forward using only ttnn.* ops")


class TTNN<ModelName>Model(TTNNModule):
    """TTNN wrapper for the full <ModelName> model.
    Reference: $TT_METAL_HOME/models/tt_transformers/tt/model.py
    """

    @classmethod
    def from_torch(cls, hf_model):
        register_modules(hf_model, {
            type(hf_model.model.layers[0]): TTNN<ModelName>DecoderLayer,
            type(hf_model.model.norm): TTNNRMSNorm,
        })
        new_module = cls()
        new_module._fallback_torch_layer = hf_model
        new_module.model = hf_model
        return new_module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
```

If Recipe/Auto API path, also generate:

```python
from typing import Dict, Type
from torch import nn
from tt_symbiote.models.auto.auto_mappings import register_recipe

@register_recipe(hf_class_name="<HFModelClass>ForCausalLM")
class <ModelName>Recipe:
    """Recipe for <model_name> model bring-up."""

    def build_module_dict(self, model) -> Dict[Type, Type]:
        return {
            type(model.model): TTNN<ModelName>Model,
            nn.Linear: TTNNLinear,
        }

    def post_register(self, model):
        import torch
        type(model).device = property(lambda self: torch.device("cpu"))
        for module in model.modules():
            if isinstance(module, TTNNModule):
                module._bypass_tensor_wrapping = True

    def make_kv_cache(self, model, device, batch_size: int = 1, **kwargs):
        return None
```

## Phase 3: Dynamic Skill Orchestration -- Deep-Work Cycles

### Main Orchestration Loop

```
LOOP:
  1. Read bringup_status.json
  2. Evaluate decision tree to determine next_stage and the skill_name it maps to
  3. Gather context for the planner:
     3a. Read the target skill's SKILL.md:
         cat .claude/skills/<skill_name>/SKILL.md
         Store the full content as $SKILL_CONTENT.
         This step is MANDATORY -- the planner cannot produce a valid plan without
         the skill's instructions. If the file does not exist, log an error and
         skip the stage.
     3b. Read the curated tech reports for this stage (see Tech Report Selection table):
         For each report path in the table row for next_stage:
           cat $TT_METAL_HOME/tech_reports/<path>
         Store each report's content.
     3c. Read bringup_status.json (refresh -- may have been updated by a prior iteration).
     3d. Read CLAUDE.md for repository conventions.
  4. Log decision: {"timestamp": ..., "decision": next_stage, "reason": "...", "result": "pending"}
  5. Execute deep-work cycle for next_stage:
     a. PLAN:     Spawn planner sub-agent.
                  Compose prompt from: CLAUDE.md + $SKILL_CONTENT + tech reports + bringup_status.json + model context.
                  The planner receives the FULL content of the skill's SKILL.md (not a summary).
     b. EVALUATE: Spawn evaluator sub-agent to check the plan against $SKILL_CONTENT.
                  If rejected (max 2 rejections per retry): return to PLAN with feedback.
     c. EXECUTE:  Spawn executor sub-agent with approved plan + $SKILL_CONTENT.
     d. ASSESS:   Check execution result. On failure, increment outer retry (max 3).
  6. Update bringup_status.json with results
  7. If all stages complete or no more feasible stages: break
  8. LOOP back to step 1
```

### Fresh Sub-Agent Architecture

When invoking a skill, spawn a fresh sub-agent via the Agent tool:

```
Agent tool call:
  prompt: |
    You are a specialized agent for the '<skill_name>' skill.

    ## SKILL INSTRUCTIONS
    <full content of the skill's SKILL.md, obtained in step 3a>

    ## CONTEXT
    Model: <model_name>
    Model ID: <hf_model_id>
    Device: <device_arch>
    TT_METAL_COMMIT: <commit_hash>
    Integration path: <recipe|manual>

    ## CURRENT STATE
    <content of bringup_status.json>

    ## AUTONOMOUS DECISIONS (from user decision profile)
    - trust_remote_code: always True
    - PCC failure: use NORMAL mode only (no DPL/SEL)
    - Missing ops: integration module if reusable by 2+ models, inline otherwise
    - Do NOT ask any questions. Make all decisions autonomously.
    - Log every decision with rationale.

    ## YOUR TASK
    Execute the skill following its instructions. Complete the plan-verify-execute loop.

    IMPORTANT: The skill's SKILL.md contains "ASK the user" prompts. IGNORE those.
    Instead, make all decisions autonomously using the decision profile above
    and the model context provided.

    Report back: what you did, what passed, what failed, what artifacts were created.
```

**CRITICAL**: Each sub-agent runs in isolation with its own plan-verify-execute loop.
The sub-agent does NOT spawn further sub-agents. The PVE loop is INTERNAL to the
sub-agent's single session. If the PVE loop fails 5 times within a sub-agent,
the sub-agent reports failure back to the orchestrator.

### Per-Stage Deep-Work Cycle Details

#### Stage: PCC_TEST_GEN

**Skill invoked**: `pcc-test-gen`

**Pre-condition**: scaffold:completed

**Planner context (in addition to base context)**:
- Full pcc-test-gen SKILL.md (read in step 3a: `cat .claude/skills/pcc-test-gen/SKILL.md`)
- Tech reports: TTNN-model-bringup.md, data_formats.md, tensor_sharding.md, FlashAttention.md
- HF model source code analysis (module hierarchy, shapes)

**Autonomous decisions injected**:
- Input shapes: batch_sizes=[1], seq_lengths=[32, 128]
- Device architecture: T3K (mesh_device fixture)
- Model directory name: already derived in Step 1
- Module hierarchy: auto-derived, no user confirmation needed
- compare_fn_outputs migration: defer to later (do not migrate existing shared tests now)

**Expected artifacts**: shapes.json, op_map.json, test_ops_*.py, test_composites_*.py, test_decoder_*.py, test_modeling_*.py, test_device_guards_*.py, pcc_utils.py (if stub)

**Success criteria**: At least Tier 1 tests can be collected by pytest (import validation). Full PCC pass is NOT required at this stage -- that comes during the validation sub-loop.

**After execution**: Run Tier 1 tests. If they fail, analyze failure, update bringup_status.json, and re-enter the deep-work cycle with failure context. If Tier 1 passes, run Tier 2, then Tier 3. Update status at each tier.

#### Stage: OP_SWEEP

**Skill invoked**: `op-sweep`

**Pre-condition**: pcc_test_gen:completed AND at least Tier 1-3 tests passing

**Planner context**:
- Full op-sweep SKILL.md (read in step 3a: `cat .claude/skills/op-sweep/SKILL.md`)
- Tech reports: data_formats.md, GEMM_FLOPS.md, memory/allocator.md, tensor_sharding.md, tensor_layouts/tensor_layouts.md, YoloV4-TTNN/yolov4.md
- shapes.json and op_map.json from pcc-test-gen

**Autonomous decisions injected**:
- Which ops to sweep: `linear` (the most impactful op for LLMs)
- Sweep mode: quick (~18 configs: 3 dtypes x 3 fidelities x 2 fp32_acc)
- PCC threshold: 0.999

**Expected artifacts**: sweep_results/*.csv, *_best.json

**Success criteria**: At least one passing config per swept op

#### Stage: TRACY_PROFILING

**Skill invoked**: `tracy-profiling`

**Pre-condition**: op_sweep:completed OR (pcc_test_gen:completed AND Tier 4 test exists)

**Planner context**:
- Full tracy-profiling SKILL.md (read in step 3a: `cat .claude/skills/tracy-profiling/SKILL.md`)
- Tech reports: MetalProfiler/metal-profiler.md, AdvancedPerformanceOptimizations.md, GEMM_FLOPS.md
- Test file path (Tier 4 test)

**Autonomous decisions injected**:
- Test file: `tests/models/<model_name>/Tier4/test_modeling_<model_name>.py`
- Decoder layer limiting: Yes, limit to 2 layers
- Output directory: `tests/models/<model_name>/` (tier subdirs Tier1..Tier4/)

**Expected artifacts**: ops_perf_results_*.csv, perf_report.txt

**Success criteria**: CSV generated with valid data rows

#### Stage: PERF_ANALYSIS

**Skill invoked**: `perf-analysis`

**Pre-condition**: tracy_profiling:completed AND CSV exists

**Planner context**:
- Full perf-analysis SKILL.md (read in step 3a: `cat .claude/skills/perf-analysis/SKILL.md`)
- Tech reports: GEMM_FLOPS.md, AdvancedPerformanceOptimizations.md, data_formats.md, Saturating_DRAM_bandwidth.md
- Tracy CSV path
- Sweep results path (if available)

**Autonomous decisions injected**:
- Model name: from bringup_status.json
- Target DeviceArch: T3K

**Expected artifacts**: recommendation.json in perf_results/

**Success criteria**: recommendation.json with at least one recommendation

#### Stage: CONFIG_OPTIMIZE

**Skill invoked**: `config-optimize-all`

**Pre-condition**: perf_analysis:completed AND recommendation.json exists

**Planner context**:
- Full config-optimize-all SKILL.md (read in step 3a: `cat .claude/skills/config-optimize-all/SKILL.md`)
- Tech reports: data_formats.md, GEMM_FLOPS.md, tensor_sharding.md, memory/allocator.md, YoloV4-TTNN/yolov4.md
- recommendation.json
- Current modeling_*.py

**Autonomous decisions injected**:
- Optimization goal: Balanced (functionality first, then speed -- per decision profile)
- PCC threshold: 0.999
- Apply to all modules at once (no per-module review)

**Expected artifacts**: Updated modeling_*.py with optimized subclasses

**Success criteria**: PCC tests still pass with new configs

#### Stage: TRACED_EXECUTION

**Skill invoked**: `traced-execution`

**Pre-condition**: config_optimize:completed

**Planner context**:
- Full traced-execution SKILL.md (read in step 3a: `cat .claude/skills/traced-execution/SKILL.md`)
- Tech reports: AdvancedPerformanceOptimizations.md, TTNN-model-bringup.md, graph-tracing.md, operation-tracing.md
- Current modeling file and Tier 4 test file

**Autonomous decisions injected**:
- Test file to base on: Tier 4 test
- trace_region_size: 200000000 (~200MB)
- Compare PCC between NORMAL and TRACED: Yes

**Expected artifacts**: test_traced_*.py

**Success criteria**: Trace tests pass, PCC delta < 0.001 vs NORMAL

## Phase 4: Final Validation and Summary

After all stages complete (or all feasible stages complete):

### Tier 4 Semantic Validation

Run the full model with a real validation sample (auto-selected from the Model-Specific Validation Samples table):

```python
# For Causal LM:
input_text = "The capital of France is"
# Tokenize, run through model, decode output
# Check: output is coherent text continuation (not garbage/NaN/repeated tokens)

# Semantic check (autonomous -- no user judgment needed):
# 1. Output is not empty
# 2. Output does not contain NaN or inf
# 3. Output is not all the same token repeated
# 4. Output length > 0 tokens
# 5. If possible, check that output is linguistically plausible
```

Log the validation result in bringup_status.json.

### Generate Final Summary

```
MODEL BRING-UP SUMMARY
======================
Model: <model_name> (<model_id>)
Device: T3K
TT_METAL_COMMIT: <hash>
Integration path: <recipe|manual>
Started: <timestamp>
Completed: <timestamp>

FILES CREATED/MODIFIED:
  - src/tt_symbiote/models/<model_name>/modeling_<model_name>.py
  - src/tt_symbiote/models/<model_name>/__init__.py
  - src/tt_symbiote/models/__init__.py (if recipe path)
  - tests/models/<model_name>/shapes.json            (ROOT artifact)
  - tests/models/<model_name>/op_map.json            (ROOT artifact)
  - tests/models/<model_name>/Tier1/test_ops_<model_name>.py
  - tests/models/<model_name>/Tier2/test_composites_<model_name>.py
  - tests/models/<model_name>/Tier3/test_decoder_<model_name>.py
  - tests/models/<model_name>/Tier4/test_modeling_<model_name>.py
  - tests/models/<model_name>/Tier4/test_device_guards_<model_name>.py
  - tests/models/<model_name>/Tier4/test_traced_<model_name>.py
  - tests/models/<model_name>/sweep_results/*.csv
  - tests/models/<model_name>/perf_results/recommendation.json
  - tests/models/<model_name>/bringup_status.json

PCC RESULTS:
  Tier 1 (ops):       <pass/fail> (PCC: <value>)
  Tier 2 (composites): <pass/fail> (PCC: <value>)
  Tier 3 (decoder):    <pass/fail> (PCC: <value>)
  Tier 4 (full model): <pass/fail> (PCC: <value>)

PERFORMANCE OPTIMIZATION:
  Subclasses applied: <list>
  Override methods: <list>
  Speedup: <value>

TRACED EXECUTION:
  Status: <pass/fail>
  PCC delta vs NORMAL: <value>

SEMANTIC VALIDATION:
  Input: "<validation sample>"
  Output: "<model output>"
  Status: <coherent/incoherent>

DECISION LOG: <count> decisions made autonomously
  See bringup_status.json for full log.

KNOWN ISSUES:
  - <any stages that failed 3+ times>
  - <any stages skipped>

NEXT STEPS:
  - <recommended follow-up actions>
```

## State Management: bringup_status.json

Located at `tests/models/<model_name>/bringup_status.json`:

```json
{
  "model_name": "<model_name>",
  "model_id": "<hf_model_id>",
  "device_arch": "T3K",
  "tt_metal_commit": "<commit_hash>",
  "integration_path": "recipe|manual",
  "started_at": "<ISO timestamp>",
  "validation_sample": "<auto-selected sample text/description>",
  "phases": {
    "scaffold": {
      "status": "not_started|in_progress|completed|failed|skipped",
      "artifacts": [],
      "notes": "",
      "started_at": null,
      "completed_at": null
    },
    "pcc_test_gen": {
      "status": "...",
      "artifacts": [],
      "test_results": {
        "tier1": {"status": "not_run|pass|fail", "pcc": null, "details": ""},
        "tier2": {"status": "not_run|pass|fail", "pcc": null, "details": ""},
        "tier3": {"status": "not_run|pass|fail", "pcc": null, "details": ""},
        "tier4": {"status": "not_run|pass|fail", "pcc": null, "details": ""}
      },
      "started_at": null,
      "completed_at": null
    },
    "op_sweep": {
      "status": "...",
      "artifacts": [],
      "notes": "",
      "started_at": null,
      "completed_at": null
    },
    "tracy_profiling": {
      "status": "...",
      "artifacts": [],
      "notes": "",
      "started_at": null,
      "completed_at": null
    },
    "perf_analysis": {
      "status": "...",
      "artifacts": [],
      "notes": "",
      "started_at": null,
      "completed_at": null
    },
    "config_optimize": {
      "status": "...",
      "artifacts": [],
      "notes": "",
      "started_at": null,
      "completed_at": null
    },
    "traced_execution": {
      "status": "...",
      "artifacts": [],
      "notes": "",
      "started_at": null,
      "completed_at": null
    }
  },
  "decision_log": [
    {
      "timestamp": "<ISO>",
      "decision": "<stage or decision name>",
      "reason": "<why this decision was made, referencing the decision profile>",
      "result": "pass|fail|skipped"
    }
  ],
  "retry_counts": {
    "scaffold": 0,
    "pcc_test_gen": 0,
    "op_sweep": 0,
    "tracy_profiling": 0,
    "perf_analysis": 0,
    "config_optimize": 0,
    "traced_execution": 0
  },
  "semantic_validation": {
    "input": "",
    "output": "",
    "status": "not_run|coherent|incoherent"
  }
}
```

**Schema notes**:
- `integration_path`: One of `"recipe"` or `"manual"`. `"recipe"` uses `@register_recipe` + `build_module_dict()` + Auto API. `"manual"` uses explicit `from_torch()` chains + `register_modules()`.
- Phase keys use underscores (`pcc_test_gen`) while skill directory names use hyphens (`pcc-test-gen`).
- `test_results` is only defined for `pcc_test_gen` since other phases have different output structures.
- `decision_log` is append-only. Every autonomous decision is logged with its rationale.
- `retry_counts` are per-phase. Max 3 retries per phase (decision profile). See "Nested Retry Semantics" section for interaction with evaluator rejections.

## Error Handling

| Problem | Autonomous Resolution | Decision Profile Reference |
|---------|----------------------|---------------------------|
| Model not in HF transformers | Use `trust_remote_code=True`; manual integration path | Custom code: auto-proceed |
| Model directory already exists | Extend from current state; never overwrite | Directory exists: extend |
| Recipe registration fails | Check __init__.py imports; check _RECIPE_BEARING_SUBPACKAGES; retry | Retry limit: 3 |
| PCC failures during bring-up | Isolate via targeted tier re-runs in NORMAL mode only | PCC failure: NORMAL mode only |
| from_torch signature mismatch | Read source of each integration class's from_torch | -- |
| Existing test uses deprecated API | Do NOT copy; use register_modules | -- |
| Sub-agent fails (PVE loop exhausted) | Update status, increment retry count, re-assess decision tree | Retry limit: 3 |
| Stage fails 3+ times | Skip stage, log as known issue, continue to next feasible stage | Retry limit: 3, then skip |
| torch.* in forward (A1 violation) | Fix immediately -- all downstream skills depend on pure TTNN | -- |
| TT_METAL_COMMIT stale | Update from git rev-parse HEAD | -- |
| Missing TTNN op | If 2+ models would use it: create integration module. If 1 model: inline. | Missing ops: integration if reusable |
| Semantic validation fails | Log as known issue, do not block completion | Min accuracy: semantic correctness |
| tracy/tt-perf-report not available | Skip profiling stages, log as known issue | Retry limit: 3, then skip |
| Skill SKILL.md file not found | Log error, skip stage, note in final report as infrastructure issue | -- |
