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
| Model Bring-Up | `model-bringup` | Full model bring-up: scaffold, tests, config, recipe |

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
```

## Do Not
- Do NOT use `register_module_replacement_dict` (deprecated alias for `register_modules`)
- Do NOT put tests in `tests/models/` (directory removed)
- Do NOT create `TTNNModule.forward()` without `@run_on_devices`
- Do NOT reference or recreate anything related to gr00t
- Do NOT modify Makefile or pre-commit hooks
- Do NOT import from `src.tt_symbiote...` (use `tt_symbiote...` -- package is pip-installed)
- Do NOT import from `tt_symbiote.core.config` (module does not exist)
- Do NOT define `device` or `mesh_device` fixtures (they come from ttnn pytest plugin)
