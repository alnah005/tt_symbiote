# tt_symbiote

## What This Is

Python package for bringing HuggingFace transformer models to Tenstorrent TTNN hardware.
Pip-installable: `pip install -e .`

Converts PyTorch modules to TTNN-accelerated equivalents via a `TTNNModule` base class
with automatic device management and fallback.

## Repository Layout

```
src/tt_symbiote/
  _hf_compat.py    # Compatibility shims for upstream transformers API drift
  core/            # TTNNModule base class, DeviceArch, run_config, tensor, arch, ccl, utils
    module.py      # TTNNModule, DeviceArch enum, @run_on_devices decorator
    run_config.py  # Run modes (NORMAL, DPL, TRACED, etc.), DispatchManager, TracedRun
    tensor.py      # TorchTTNNTensor (torch.Tensor subclass wrapping TTNN)
    utils.py       # compare_fn_outputs (WARNING: only prints, does NOT assert PCC)
  integrations/    # Shared TTNN building blocks (NOT per-model)
    ttnn_linear.py          # TTNNLinear and variants (sharded, bfloat8_b, etc.)
    ttnn_linear_intelligent.py  # SmartTTNNLinear auto-selecting variants
    ttnn_attention.py       # SDPA, FusedQKV, self-attention variants
    ttnn_normalization.py   # RMSNorm, LayerNorm, distributed variants
    ttnn_moe.py             # Mixture-of-experts routing and expert execution
    ttnn_embedding.py       # Embeddings and rotary position encodings
    ttnn_rope.py            # Rotary position embeddings
    ttnn_conv.py            # Conv2d, MaxPool2d
    ttnn_activation.py      # TTNNSilu, TTNNReLU, TTNNGelu
    ttnn_tensor.py          # TTNNPermute, TTNNReshape, TTNNAdd
    tt_cnn/                 # CNN-specific pipeline (builder, executor, pipeline)
  models/          # Per-model compositions and recipes
    <model_name>/modeling_<model_name>.py
    __init__.py    # _RECIPE_BEARING_SUBPACKAGES list
  auto/            # AutoModelForCausalLM, recipe registry (@register_recipe)
  utils/           # device_management.py (set_device), module_replacement.py (register_modules)
  generation/      # Text generation utilities (stub)

tests/
  auto/            # Software-only tests (no hardware needed, mock TTNN)
  capabilities/    # Hardware tests: shared + per-model subdirectories
    <model_name>/  # Per-model test directories
    pcc_utils.py   # PCC assertion helpers (stub -- filled by pcc-test-gen skill)
    shared_configs.py  # Config presets (stub -- filled by config skills)
    conftest.py    # Shared fixtures (pcc_threshold = 0.99)
    test_attention.py, test_conv.py, test_moe.py, test_rope.py  # Shared capability tests
```

## Architecture: The TTNNModule Lifecycle

```python
# Phase 1: Create TTNN module from PyTorch
ttnn_module = TTNNSomeModule.from_torch(pytorch_module)

# Phase 2: Bind to device (orchestrates preprocess_weights + move_weights_to_device)
set_device(ttnn_module, device)

# Phase 3: Forward pass
output = ttnn_module(input_tensor)
```

Full internal sequence: `from_torch()` -> `set_device()` -> `preprocess_weights()` -> `move_weights_to_device()` -> `forward()`

## Two Integration Paths

- **Recipe/Auto API** (bailing_moe_v2 pattern): `@register_recipe(hf_class_name="...")` with `build_module_dict()`. Uses `register_modules()` for automatic replacement. Recipe methods are INSTANCE methods (NOT @staticmethod).
- **Manual from_torch** (gemma4, qwen3_moe pattern): Nested `TTNN<Class>.from_torch()` calls building the tree manually. Requires explicit `override_children_module_names()` and `set_model_config()` on the root after tree construction.

## Hard Conventions (ALL new code must follow these)

### Pure TTNN Forward Path (Mandatory)

ALL `TTNNModule.forward()` implementations MUST use pure TTNN operations. No `torch.*` calls
are permitted in the forward/compute path. This matches the `tt_transformers` pattern in
`$TT_METAL_HOME/models/tt_transformers/tt/` where `forward()` methods use only `ttnn.linear()`,
`ttnn.matmul()`, `ttnn.add()`, `ttnn.reshape()`, `ttnn.deallocate()`, etc.

- **forward()**: Pure `ttnn.*` ops only. No `torch.matmul`, `torch.nn.functional.*`, `torch.cat`, etc.
- **preprocess_weights_impl()**: May use PyTorch for weight transformation (transpose, pad, cast).
- **from_torch()**: May use PyTorch for weight extraction and module construction.
- **__init__()**: May use PyTorch for setup.

Reference: `$TT_METAL_HOME/models/tt_transformers/tt/mlp.py` forward() uses `ttnn.linear()`,
`ttnn.multiply()`, `ttnn.silu()`, `ttnn.deallocate()` -- zero torch calls.

### $TT_METAL_HOME Exploration (Mandatory Preamble)

Before writing ANY code, every code-generating agent MUST explore `$TT_METAL_HOME` for
reference implementations and patterns. This is NOT optional.

```bash
# 1. Capture the current tt-metal commit for reproducibility
TT_METAL_COMMIT=$(git -C $TT_METAL_HOME rev-parse HEAD)
echo "tt-metal commit: $TT_METAL_COMMIT"

# 2. Read relevant reference implementations
ls $TT_METAL_HOME/models/tt_transformers/tt/   # LLM patterns (attention, mlp, model, decoder)
ls $TT_METAL_HOME/models/tt_dit/               # DiT/diffusion patterns
ls $TT_METAL_HOME/models/tt_cnn/               # CNN patterns
ls $TT_METAL_HOME/models/demos/                # Demo models (deepseek_v3, qwen25_vl, qwen3_vl, vit, etc.)
ls $TT_METAL_HOME/ttnn/examples/               # TTNN op examples
```

Key reference files for LLM bring-up:
- `models/tt_transformers/tt/attention.py` -- Pure TTNN attention with SDPA
- `models/tt_transformers/tt/mlp.py` -- Pure TTNN MLP (gate/up/down projections)
- `models/tt_transformers/tt/decoder.py` -- Decoder layer composition
- `models/tt_transformers/tt/model.py` -- Full transformer model
- `models/tt_transformers/tt/common.py` -- Mode enum, utility functions
- `models/tt_transformers/tt/rope.py` -- Rotary embeddings in TTNN
- `models/tt_transformers/tt/lm_head.py` -- LM head implementation
- `models/tt_transformers/tt/generator.py` -- Trace capture/replay for LLMs
- `models/tt_transformers/tt/model_config.py` -- Config patterns (~196K bytes, ~4,236 lines)

### Tech Reports (Mandatory Reading)

Every agent MUST read the relevant tech reports from `$TT_METAL_HOME/tech_reports/` before
starting work. There are 49 tech report files across 30 directories:

| # | Path | Topic |
|---|------|-------|
| 1 | `ttnn/TTNN-model-bringup.md` | **TTNN model bring-up guide** |
| 2 | `ttnn/ttnn.md` | TTNN overview and API |
| 3 | `ttnn/comparison-mode.md` | TTNN comparison/debug mode |
| 4 | `ttnn/graph-tracing.md` | TTNN graph tracing |
| 5 | `ttnn/operation-tracing.md` | TTNN operation tracing |
| 6 | `LLMs/llms.md` | LLM optimization on Tenstorrent |
| 7 | `LLMs/vLLM_integration.md` | vLLM integration guide |
| 8 | `data_formats/data_formats.md` | Data format details (bfloat16, bfloat8_b, etc.) |
| 9 | `data_formats/reconfig_data_format.md` | Runtime data format reconfiguration |
| 10 | `memory/allocator.md` | Memory allocator internals |
| 11 | `tensor_layouts/tensor_layouts.md` | Tensor layout guide (ROW_MAJOR, TILE) |
| 12 | `tensor_sharding/tensor_sharding.md` | Tensor sharding strategies |
| 13 | `tensor_serialization/tensor_serialization.md` | Tensor serialization |
| 14 | `tensor_accessor/tensor_accessor.md` | Tensor accessor API |
| 15 | `tensor_accessor/tensor_accessor_iterator.md` | Tensor accessor iterator |
| 16 | `GEMM_FLOPS/GEMM_FLOPS.md` | GEMM performance and FLOPS |
| 17 | `matrix_engine/matrix_engine.md` | Matrix engine architecture |
| 18 | `FlashAttention/FlashAttention.md` | Flash attention implementation |
| 19 | `FlashAttention/FlashDecode.md` | Flash decode optimization |
| 20 | `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` | Advanced perf optimizations |
| 21 | `Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md` | DRAM bandwidth saturation |
| 22 | `MetalProfiler/metal-profiler.md` | Metal profiler usage |
| 23 | `Debugging/Kernel_Debugging_Tips.md` | Kernel debugging tips |
| 24 | `EthernetMultichip/BasicEthernetGuide.md` | Ethernet multi-chip guide |
| 25 | `Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md` | Mesh device programming |
| 26 | `Programming_Multiple_Meshes/Programming_Multiple_Meshes.md` | Multiple mesh programming |
| 27 | `CNNs/cnn_optimizations.md` | CNN optimization techniques |
| 28 | `CNNs/ttcnn.md` | TTCNN framework |
| 29 | `ViT-TTNN/vit.md` | Vision Transformer on TTNN |
| 30 | `ViT-TTNN/vit_bh.md` | ViT on Blackhole |
| 31 | `YoloV4-TTNN/yolov4.md` | YOLOv4 on TTNN |
| 32 | `Blackhole/BlackholeBringUpProgrammingGuide.md` | Blackhole bring-up guide |
| 33 | `SubDevices/SubDevices.md` | Sub-device programming |
| 34 | `TT-Distributed/MultiHostMeshRuntime.md` | Multi-host mesh runtime |
| 35 | `TT-Distributed/TT-Distributed-Architecture-1219.md` | TT-Distributed architecture |
| 36 | `TT-Distributed/TTMeshMigrationGuide.md` | TT Mesh migration guide |
| 37 | `TT-Fabric/TT-Fabric-Architecture.md` | TT-Fabric architecture |
| 38 | `Handling_Special_Value/special_values.md` | NaN/Inf special value handling |
| 39 | `code-indexing/kernel-code-indexing.md` | Kernel code indexing |
| 40 | `op_kernel_dev/accuracy_tips/accuracy_tips.md` | Op kernel accuracy tips |
| 41 | `ttnn_operators/intimg.md` | TTNN integer image ops |
| 42 | `prog_examples/NoC_tile_transfer/NoC_tile_transfer.md` | NoC tile transfer |
| 43 | `prog_examples/matmul_multi_core_optimized/matmul_multi_core_optimized.md` | Optimized multi-core matmul |
| 44 | `prog_examples/matmul_multi_core_optimized/data_mcast.md` | Data multicast for matmul |
| 45 | `prog_examples/matmul_multi_core_optimized/data_reuse.md` | Data reuse for matmul |
| 46 | `prog_examples/multicast/multicast.md` | Multicast programming |
| 47 | `prog_examples/pad_multi_core/pad_multi_core.md` | Multi-core padding |
| 48 | `prog_examples/sfpu_eltwise_chain/sfpu_eltwise_chain.md` | SFPU eltwise chain |
| 49 | `prog_examples/shard_data_rm/shard_data_rm.md` | Row-major data sharding |

**Priority reading order** (read these first, remaining as needed):
1. `ttnn/TTNN-model-bringup.md` -- the primary bringup guide
2. `LLMs/llms.md` -- LLM-specific patterns
3. `data_formats/data_formats.md` -- dtype choices
4. `tensor_layouts/tensor_layouts.md` -- layout fundamentals
5. `tensor_sharding/tensor_sharding.md` -- sharding strategies
6. `FlashAttention/FlashAttention.md` -- attention implementation
7. `GEMM_FLOPS/GEMM_FLOPS.md` -- matmul performance
8. `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md` -- perf guide
9. `memory/allocator.md` -- memory management
10. `MetalProfiler/metal-profiler.md` -- profiling

### TT_METAL_HOME Commit Hash Tracking

Every model file generated by model-bringup or code-writing skills MUST include:

```python
# At module level in modeling_<model_name>.py, after imports:
TT_METAL_COMMIT = "<full 40-character git hash>"
```

Captured via:
```bash
git -C $TT_METAL_HOME rev-parse HEAD
```

This records which tt-metal version the model was developed against.

### Plan-Verify-Execute Loop (All Skills)

Every skill follows a mandatory loop:

```
PLAN: Determine what to do (read code, analyze, draft changes)
  |
  v
VERIFY: Check the plan (imports resolve, shapes match, no torch in forward, etc.)
  |
  +--> If verify fails: return to PLAN with failure details (max 5 attempts)
  |
  v
EXECUTE: Write files, run commands, produce artifacts
```

Verification checks (done WITHOUT hardware, WITHOUT user approval):
- All imports resolve: `python -c "from tt_symbiote.integrations.ttnn_linear import TTNNLinear"`
- No `torch.*` calls in `forward()` bodies (pure TTNN constraint)
- Shapes are consistent with `shapes.json` and HF config
- `@run_on_devices` decorator present on all `forward()` methods
- License headers present
- No deprecated API usage (`register_module_replacement_dict`, `compare_fn_outputs` for assertions)

If the PVE loop fails 5 times within a single sub-agent session, the sub-agent reports
failure back to the orchestrator rather than continuing indefinitely.

### Naming
- Model directories: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
- `<model_name>` MUST match `transformers.models.<model_name>` where an HF equivalent exists.
- Integration modules: `src/tt_symbiote/integrations/ttnn_<capability>.py` (NOT per-model)
- TTNN classes: `TTNN<ModelName><Component>` (e.g., `TTNNGemma4Attention`)
- Completed renames: owlvit (not owl_vit), speecht5 (not speech_t5), qwen3_omni_moe (not qwen_omni). gr00t: DELETED, do not reference.

### Device Guards
ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices` decorator.
```python
from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch

class TTNNMyModule(TTNNModule):
    @run_on_devices(DeviceArch.T3K)  # Default; widen after validation
    def forward(self, input_tensor):
        ...
```
Guard goes on forward() ONLY, not on __init__, from_torch, or preprocess_weights.

### Test Location
- Per-model tests: `tests/capabilities/<model_name>/test_modeling_<model_name>.py`
- Shared capability tests: `tests/capabilities/` (root level)
- Software-only tests: `tests/auto/`
- `tests/models/` NO LONGER EXISTS.

### PCC Testing
- `compare_fn_outputs()` from `core/utils.py` only prints warnings -- it does NOT assert. NEVER use it as sole validation.
- Always use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.
- Default threshold: 0.99 (matches conftest.py fixture and pcc_utils.py).
- Pass `threshold=0.999` explicitly for stricter bring-up validation.
- Known tech debt: Existing shared tests (test_attention.py, test_conv.py, test_moe.py, test_rope.py) still use `compare_fn_outputs()` instead of `assert_pcc()`. These should be migrated. New tests MUST NOT use `compare_fn_outputs()`.

### Config System (Current State)
The hierarchical typed config system (DtypeConfig, ComputeConfig, MemoryConfig, ModuleConfig) is PLANNED but NOT yet implemented. None of these classes exist.

Current mechanisms:
1. **Subclass-based overrides** (primary): Model-specific subclasses override `preprocess_weights_impl()` for dtype or `move_weights_to_device_impl()` for compute config. Example: `TTNNLinearLLama` uses bfloat8_b vs `TTNNLinear`'s bfloat16.
2. **`_model_config` dict**: Set via `set_model_config()`, accessed as `self.model_config[self.module_name]`. Passed through `register_modules(model, dict, model_config=...)`.

### Recipe Registration (for Auto API)
1. Create `modeling_<model_name>.py` with `@register_recipe(hf_class_name="...")`
2. Recipe methods are INSTANCE methods (NOT @staticmethod):
   - `def build_module_dict(self, model) -> Dict[Type, Type]` (required)
   - `def post_register(self, model)` (optional)
   - `def make_kv_cache(self, model, device, batch_size=1, **kwargs)` (optional)
3. Create `__init__.py` that imports the recipe class
4. Add `"<model_name>"` to `_RECIPE_BEARING_SUBPACKAGES` in `models/__init__.py`

### Run Modes (TT_SYMBIOTE_RUN_MODE env var)
| Mode | Description |
|------|-------------|
| LIGHTWEIGHT | Always torch fallback, no TTNN attempted |
| NORMAL | Standard TTNN execution |
| NORMAL_WITH_FALLBACK | Try TTNN, fall back to torch on failure |
| SEL | Returns TTNN result but logs PCC comparison |
| DPL | Debug Per Layer -- compares torch vs TTNN at every boundary |
| DPL_NO_ERROR_PROP | Like DPL but re-materializes inputs from torch (prevents drift) |
| CPU | CPU-only execution with tensor wrapping |
| TRACED | Trace capture/replay for production performance |

### Deprecated APIs
| Deprecated | Replacement | Location |
|-----------|-------------|----------|
| `register_module_replacement_dict()` | `register_modules()` | `utils/module_replacement.py` |
| `compare_fn_outputs()` (for test assertions) | `assert_pcc()` | `core/utils.py` -> `tests/capabilities/pcc_utils.py` |

### License Headers
New files use `(C)`:
```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
```
Some existing framework files use the Unicode copyright symbol -- do not change those.

### Test Fixtures
- `device` and `mesh_device` fixtures come from the **ttnn pytest plugin** (installed with tt-metal). NOT defined in this repo's conftest.py.
- Configure via `@pytest.mark.parametrize("device_params", [{...}], indirect=True)`
- Single-device tests (N150, P150): use `device` fixture
- Multi-device tests (N300, T3K, TG, etc.): use `mesh_device` fixture

## Model Bring-Up State Tracking

Active model bring-ups track progress in `tests/capabilities/<model_name>/bringup_status.json`.
This file records:
- Which phases (scaffold, pcc_test_gen, op_sweep, etc.) are completed
- Test results per tier
- Artifacts created
- Decision log for debugging
- Retry counts per phase

Do not manually edit this file -- it is managed by the model-bringup skill.

## Autonomous Orchestration Pattern

The model-bringup skill implements the deep-work pattern (plan -> evaluate -> execute -> 
re-plan on failure) as an autonomous orchestrator. Key properties:

1. **Single input**: Only a HuggingFace model ID is required. Everything else is derived.
2. **Decision profile**: All decisions are pre-encoded (integration path, PCC handling, 
   retry limits, validation samples, etc.). The orchestrator never stops to ask the user.
3. **Stage isolation**: Each stage (scaffold, pcc-test-gen, op-sweep, tracy-profiling, 
   perf-analysis, config-optimize, traced-execution) runs as a separate deep-work cycle 
   with its own planner, evaluator, and executor sub-agents.
4. **Curated context**: Planners receive CLAUDE.md + the relevant skill SKILL.md + curated 
   tech reports + bringup_status.json. They do NOT receive raw deep-plan files, prior 
   iteration plans, or the full 49 tech report dump. The orchestrator explicitly reads each
   skill's SKILL.md file before composing the planner prompt.
5. **State tracking**: bringup_status.json records phase status, artifacts, test results, 
   decision log, and retry counts. It is the single source of truth for orchestration state.
6. **Failure handling**: Each stage retries up to 3 times (outer loop). Within each retry, 
   the evaluator may reject the plan up to 2 times (inner loop). After 3 outer failures, the 
   stage is skipped and noted in the final report. The orchestrator continues to the next 
   feasible stage.
7. **Semantic validation**: The final step generates actual model output and checks for 
   coherence (not just PCC numbers).

## Available Skills

| Skill | Invoke With | Description |
|-------|-------------|-------------|
| Tracy Profiling | `tracy-profiling` | Profile a test with tracy, generate perf CSV + report |
| PCC Test Generation | `pcc-test-gen` | Generate tiered PCC tests for a HF model |
| Op Parameter Sweep | `op-sweep` | Sweep TTNN op configs (dtype, fidelity, memory) |
| Traced Execution | `traced-execution` | Validate trace capture/replay lifecycle |
| Performance Analysis | `perf-analysis` | Analyze tracy data, rank ops, identify bottlenecks |
| Config Optimization (All) | `config-optimize-all` | Apply best configs to all modules in a model |
| Config Optimization (Module) | `config-optimize-module` | Optimize config for one specific module |
| Model Bring-Up | `model-bringup` | Fully autonomous deep-work driver for 0-to-100 model bring-up. Embeds user decision profile, drives each stage as plan-evaluate-execute cycle, tracks state in bringup_status.json. Requires only a HF model ID. |

**Skill Invocation**: model-bringup is a fully autonomous deep-work driver. It spawns each
skill as a FRESH sub-agent via the Agent tool, passing the skill's SKILL.md, CLAUDE.md,
curated tech reports, and bringup_status.json as context. Each sub-agent operates independently
with its own plan-verify-execute loop. The orchestrator makes all decisions autonomously per
its encoded user decision profile -- it never stops to ask the user. All decisions are logged
with rationale in bringup_status.json. Each stage (scaffold, test-gen, sweep, profile,
analyze, optimize, trace) is a separate deep-work cycle: plan (1 planner) -> evaluate ->
execute -> re-plan on failure (up to 3 retries per stage, with up to 2 evaluator rejections
per retry attempt).

## Common Commands

```bash
# Run software-only tests
pytest tests/auto/ -x -v

# Run a specific model's capability tests
pytest tests/capabilities/<model_name>/ -x -v

# Run with DPL mode for debugging
TT_SYMBIOTE_RUN_MODE=DPL pytest <test_file> -x -s

# Profile with tracy
export TT_SYMBIOTE_SIGNPOST_MODE="1"
python -m tracy -p -r -v --op-support-count 20000 -m 'pytest <test_file> -x -s'

# Generate perf report
tt-perf-report --ignore-signposts */ops_perf_results_*.csv > perf_report.txt

# Capture tt-metal commit hash
git -C $TT_METAL_HOME rev-parse HEAD
```

## Do Not
- Do NOT use `register_module_replacement_dict` (deprecated alias for `register_modules`)
- Do NOT put tests in `tests/models/` (directory removed)
- Do NOT create `TTNNModule.forward()` without `@run_on_devices`
- Do NOT use `torch.*` calls inside `TTNNModule.forward()` -- pure TTNN only
- Do NOT reference or recreate anything related to gr00t
- Do NOT modify Makefile or pre-commit hooks
- Do NOT import from `src.tt_symbiote...` (use `tt_symbiote...` -- package is pip-installed)
- Do NOT import from `tt_symbiote.core.config` (module does not exist)
- Do NOT define `device` or `mesh_device` fixtures (they come from ttnn pytest plugin)
- Do NOT write code without first exploring `$TT_METAL_HOME` for reference patterns
- Do NOT skip reading tech reports before starting implementation work
