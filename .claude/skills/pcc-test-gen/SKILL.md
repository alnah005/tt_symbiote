---
name: pcc-test-gen
description: Read a HuggingFace transformers model, decompose into all module tiers (ops, composites, decoder layers, full model), generate tiered PCC tests with explicit PCC assertions, and implement the assert_pcc utility. Generates shapes.json, op_map.json, and per-tier test files under tests/models/<model_name>/.
---

# PCC Test Generation

Generate comprehensive tiered PCC tests for a HuggingFace model's TTNN bring-up.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/models/<model_name>/test_modeling_<model_name>.py`

**Test location**: Generate tiered tests into `tests/models/<model_name>/` (RICH tree) once
  end-to-end traced correctness is proven, otherwise `tests/experimental/<model_name>/`
  (MINIMAL tree). Emit `test_config.json` (5 keys: `tt_metal_commit`, `device_arch`,
  `pcc_threshold`, `hf_model_id`, `hf_revision`) + `shapes.json` + `op_map.json` alongside the
  tier files. STOP writing to the removed per-model capabilities tree.
  - Shared capability tests: `tests/shared/` root (e.g., `test_attention.py`)
  - Auto/unit tests: `tests/auto/`

**Pure TTNN forward**: ALL `TTNNModule.forward()` methods must use pure `ttnn.*` ops only.
  No `torch.*` calls in the compute path. Weight preprocessing may use PyTorch.
  Reference: `$TT_METAL_HOME/models/tt_transformers/tt/mlp.py` forward().

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.
  - Import: `from tt_symbiote.core.module import run_on_devices, DeviceArch`
  - Default: `@run_on_devices(DeviceArch.T3K)`
  - Widen only after explicit validation on other architectures.

**License headers**: Every generated `.py` file must start with:
  ```python
  # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
  # SPDX-License-Identifier: Apache-2.0
  ```
  NOTE: Some existing framework files use the Unicode copyright symbol. Do NOT change those. Use `(C)` for all NEW files.

**PCC assertions**: NEVER rely on `compare_fn_outputs()` alone -- it only prints warnings.
  Always use `assert_pcc()` from `tests/shared/pcc_utils.py`.

**Deprecated API**: Never use `register_module_replacement_dict()`.
  Use `register_modules()` from `tt_symbiote.utils.module_replacement`.
  WARNING: Some existing tests (e.g., gemma4, qwen3_moe) still use the deprecated name. Do NOT copy those patterns.

**Import conventions**:
  - Integration modules: `from tt_symbiote.modules.ttnn_<module> import TTNN<Class>`
  - Core: `from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch`
  - Tier 1-3 tests: `from tt_symbiote.utils.device_management import set_device`
  - Tier 4 (full model with Auto API): `from tt_symbiote import AutoModelForCausalLM, set_device`

**Device/mesh_device fixtures**: Provided by the `ttnn` pytest plugin, NOT by tt_symbiote.
  Do NOT define these fixtures. Use `@pytest.mark.parametrize("device_params", [...], indirect=True)`.
  - Single-device tests (N150, P150): use the `device` fixture
  - Multi-device tests (N300, T3K, TG, etc.): use the `mesh_device` fixture

**Config system**: The typed config system (ModuleConfig, DtypeConfig, etc.) does NOT exist yet.
  Current mechanism: `_model_config: dict` on TTNNModule, set via `set_model_config()`.
  Weight dtype is controlled by selecting different TTNNLinear subclasses.

**HuggingFace**: Always pass `trust_remote_code=True` in `AutoConfig.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()`.

**TT_METAL_COMMIT Hash**: When generating or modifying `modeling_<model_name>.py`, include:
  ```python
  TT_METAL_COMMIT = '<full 40-char git hash from $TT_METAL_HOME>'
  ```
  Capture via: `git -C "$TT_METAL_HOME" rev-parse HEAD`

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

Key reports for pcc-test-gen (read FIRST):
1. `ttnn/TTNN-model-bringup.md` -- Unit test structure, PCC methodology
2. `data_formats/data_formats.md` -- Understanding dtype impact on PCC
3. `tensor_sharding/tensor_sharding.md` -- How sharding affects output precision
4. `LLMs/llms.md` -- LLM-specific testing patterns
5. `FlashAttention/FlashAttention.md` -- SDPA testing considerations

Read ALL remaining reports after these priority ones.

### 0b. Explore $TT_METAL_HOME Reference Implementations

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
ls "$TT_METAL_HOME/models/tt_transformers/tt/"
ls "$TT_METAL_HOME/models/tt_dit/" 2>/dev/null
ls "$TT_METAL_HOME/models/tt_cnn/tt/" 2>/dev/null
ls "$TT_METAL_HOME/models/demos/" 2>/dev/null
```

**Extract from tt_transformers**:
- How forward() methods are structured (pure ttnn)
- How attention, MLP, decoder, embedding are decomposed (informs tier classification)
- How model_config.py parameterizes dtypes and memory (informs test shape selection)
- **Cross-reference with HF model**: Compare module structure against tt_transformers patterns
  (GQA attention? SwiGLU MLP? MoE? See attention.py, mlp.py, mixtral_moe.py)

### 0c. Capture TT_METAL_COMMIT Hash

```bash
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/salnahari/testing_dir/tt-metal}"
TT_METAL_COMMIT=$(git -C "$TT_METAL_HOME" rev-parse HEAD)
echo "TT_METAL_COMMIT=$TT_METAL_COMMIT"
```

Store this value -- it will be embedded in any modeling files created or modified.

## Plan-Verify-Execute Loop

This skill follows a mandatory loop structure. If the loop fails 5 times, report failure to the caller.

### PLAN Phase
1. Collect inputs (model ID, target device, shapes)
2. Read HuggingFace model source to enumerate all modules across 4 tiers
3. Map each module to existing tt_symbiote integration classes
4. Draft test file contents for each tier

### VERIFY Phase (no hardware, no user approval needed)
1. Verify all imports resolve: `python -c "from tt_symbiote.modules.ttnn_linear import TTNNLinear; ..."`
2. Verify shapes are consistent with HuggingFace config dimensions
3. Verify no `torch.*` calls in any generated `forward()` bodies
4. Verify `@run_on_devices` decorator is present on all generated `forward()` methods
5. Verify license headers on all generated files
6. Verify `assert_pcc()` is used (not `compare_fn_outputs()`) in all test assertions
7. If ANY verification fails, return to PLAN with the failure details and re-plan

### EXECUTE Phase (only after VERIFY passes)
Write all files (pcc_utils.py, shapes.json, op_map.json, test files).

## Step 1 -- Collect Inputs (ASK the user)

1. **HuggingFace model identifier**: e.g., `google/gemma-4-12b`, `Qwen/Qwen3-30B-A3B`
   - Or a local path to the model source files

2. **Target device architecture** (determines fixtures):
   - Single device: N150, P150 -> tests use `device` fixture
   - Multi-device: N300, T3K, TG, P300, P150x4, P150x8, BHGLX -> tests use `mesh_device` fixture
   - Default: T3K
   - NOTE: These fixtures come from the ttnn pytest plugin (not defined locally)

3. **Model directory name**: Derive using the HF naming convention:
   ```python
   from transformers import AutoConfig
   import importlib
   config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
   model_type = getattr(config, 'model_type', None)
   try:
       importlib.import_module(f"transformers.models.{model_type}")
       model_name = model_type  # Canonical HF name
   except ImportError:
       # ASK USER: "Model name '<model_type>' not found in HF transformers. Proceed?"
       model_name = model_type
   ```
   Show the derived name to the user and confirm.

4. **Input shapes** to test:
   - Batch sizes (suggest: `[1]`)
   - Sequence lengths (suggest: `[32, 128]` for initial bring-up)

## Step 2 -- Read Model Source

1. Read the HuggingFace model's source code:
   ```python
   from transformers import AutoConfig, AutoModelForCausalLM
   import inspect
   config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
   model_class = type(AutoModelForCausalLM.from_config(config))
   source_file = inspect.getfile(model_class)
   ```
   Use the Read tool to examine `source_file`.

2. Enumerate ALL modules at 4 tiers using this classification:
   - **Tier 1 (Ops)**: Leaf modules with no child TTNNModules. Wrap single TTNN ops. Examples: linear projections (q_proj, k_proj, v_proj, o_proj), embeddings, RMSNorm, activations (SiLU, GELU), RoPE.
     - Heuristic: module has no nn.Module children besides parameters, OR its forward() calls <= 3 TTNN ops.
   - **Tier 2 (Composites)**: Modules composing multiple Tier 1 modules. Examples: Attention (QKV + SDPA + output proj), MLP (gate + up + activation + down), MoE (router + experts).
     - Heuristic: contains 2+ Tier 1 child modules and is NOT a full decoder layer.
   - **Tier 3 (Decoder layers)**: Single decoder/encoder layer including attention + MLP + residuals + normalization.
     - Heuristic: named `*DecoderLayer` or `*EncoderLayer`, contains attention + MLP composites.
   - **Tier 4 (Full model)**: The complete model wrapping all layers, embeddings, and final head.
     - Heuristic: top-level model class (e.g., `*ForCausalLM`, `*Model`).

3. **Present module hierarchy to user as tree, ASK:** "Does this look correct? Any modules to add/remove?"

4. Map each module to existing tt_symbiote integration classes:
   - `nn.Linear` -> `TTNNLinear` from `tt_symbiote.modules.ttnn_linear`
   - `RMSNorm` -> `TTNNRMSNorm` from `tt_symbiote.modules.ttnn_normalization`
   - `LayerNorm` -> `TTNNLayerNorm` from `tt_symbiote.modules.ttnn_normalization`
   - Attention -> Check `ttnn_attention.py` (TTNNSelfAttention, TTNNFusedQKVSelfAttention, etc.)
   - MoE -> Check `ttnn_moe.py` (TTNNMoE, TTNNExperts, etc.)
   - If no integration exists, flag it to the user

**When to use `from_torch` vs `register_modules`**: Use `from_torch` when an integration class
provides a complete TTNN replacement for a module (e.g., `TTNNSelfAttention.from_torch(torch_attn)`).
Use `register_modules` when you need to recursively replace leaf modules within a composite
(e.g., replacing all `nn.Linear` children inside an MLP with `TTNNLinear`).

## Step 3 -- Implement assert_pcc Utility

Check if `tests/shared/pcc_utils.py` still has the `NotImplementedError` stub:

```bash
grep "NotImplementedError" tests/shared/pcc_utils.py
```

If it does, replace the stub with the full implementation. **CRITICAL**: The implementation
MUST preserve the existing stub's function signature `(actual, expected, threshold=0.99, msg="")`
to avoid breaking any code that may already reference it.

Write this content to `tests/shared/pcc_utils.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared PCC (Pearson Correlation Coefficient) assertion utilities.

CRITICAL: The framework's compare_fn_outputs() at core/utils.py only prints
warnings when PCC < 0.999 -- it NEVER raises or asserts. Tests using only
compare_fn_outputs will silently pass even with catastrophically wrong outputs.
Use assert_pcc() instead.
"""

import torch
from tt_symbiote.core.tensor import TorchTTNNTensor


def compute_pcc(actual, expected):
    """Compute PCC between actual (TTNN) and expected (PyTorch) outputs.

    Handles single tensors, TorchTTNNTensor instances, and nested collections.
    Returns list of (pcc_value, max_abs_diff) tuples, one per output tensor pair.

    NOTE: TorchTTNNTensor.to_torch is a @property, not a method.
    """
    def _extract_tensors(output, force_readback=False):
        tensors = []
        if isinstance(output, TorchTTNNTensor):
            if force_readback:
                output.elem = None  # Force readback from device
            tensors.append(output.to_torch)  # @property, no parens
        elif isinstance(output, torch.Tensor):
            tensors.append(output)
        elif isinstance(output, (list, tuple)):
            for item in output:
                tensors.extend(_extract_tensors(item, force_readback))
        elif output is None:
            pass  # Skip None outputs (e.g., optional attention weights)
        return tensors

    actual_tensors = _extract_tensors(actual, force_readback=True)
    expected_tensors = _extract_tensors(expected, force_readback=False)

    assert len(actual_tensors) == len(expected_tensors), (
        f"Mismatched output count: {len(actual_tensors)} vs {len(expected_tensors)}"
    )

    results = []
    for a, e in zip(actual_tensors, expected_tensors):
        a = a.to(torch.float32).flatten()
        e = e.to(torch.float32).flatten()
        assert a.shape == e.shape, f"Shape mismatch: {a.shape} vs {e.shape}"
        pcc = torch.corrcoef(torch.stack([a, e]))[0, 1]
        max_diff = torch.max(torch.abs(a - e)).item()
        results.append((pcc.item(), max_diff))
    return results


def assert_pcc(actual, expected, threshold=0.99, msg=""):
    """Assert PCC >= threshold between actual (TTNN) and expected (PyTorch) outputs.

    Handles single tensors, TorchTTNNTensor instances, and collections.

    Args:
        actual: TTNN output (TorchTTNNTensor, torch.Tensor, or nested collection)
        expected: PyTorch reference output (same structure)
        threshold: Minimum acceptable PCC (default 0.99; pass 0.999 for bring-up)
        msg: Optional message prefix for assertion errors
    """
    results = compute_pcc(actual, expected)
    prefix = f"{msg}: " if msg else ""
    assert len(results) > 0, f"{prefix}No output tensors to compare"
    for i, (pcc, max_diff) in enumerate(results):
        assert not torch.tensor(pcc).isnan(), (
            f"{prefix}output[{i}]: PCC is NaN (max_abs_diff={max_diff:.6f})"
        )
        assert pcc >= threshold, (
            f"{prefix}output[{i}]: PCC {pcc:.6f} < {threshold} "
            f"(max_abs_diff={max_diff:.6f})"
        )
```

## Step 4 -- Generate Shapes Manifest

Create `tests/models/<model_name>/shapes.json`:

```json
{
  "model_name": "<model_name>",
  "model_id": "<hf_model_id>",
  "device_arch": "<T3K|N150|etc>",
  "tt_metal_commit": "<TT_METAL_COMMIT from Step 0c>",
  "config": {
    "hidden_size": "<from HF config>",
    "num_attention_heads": "<from HF config>",
    "num_key_value_heads": "<from HF config>",
    "intermediate_size": "<from HF config>",
    "num_hidden_layers": "<from HF config>",
    "head_dim": "<from HF config or hidden_size // num_heads>",
    "vocab_size": "<from HF config>"
  },
  "test_shapes": {
    "batch_sizes": [1],
    "seq_lengths": [32, 128]
  },
  "module_shapes": {
    "q_proj": {"input": [1, 128, "<hidden_size>"], "weight": ["<num_heads*head_dim>", "<hidden_size>"]},
    "k_proj": {"input": [1, 128, "<hidden_size>"], "weight": ["<num_kv_heads*head_dim>", "<hidden_size>"]},
    "gate_proj": {"input": [1, 128, "<hidden_size>"], "weight": ["<intermediate_size>", "<hidden_size>"]}
  }
}
```

Also create `tests/models/<model_name>/op_map.json`:
```json
{
  "TTNNLinear": {
    "ttnn_ops": ["ttnn.linear"],
    "used_by": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "import_path": "tt_symbiote.modules.ttnn_linear.TTNNLinear",
    "device_constraint": null
  },
  "TTNNRMSNorm": {
    "ttnn_ops": ["ttnn.rms_norm"],
    "used_by": ["input_layernorm", "post_attention_layernorm", "norm"],
    "import_path": "tt_symbiote.modules.ttnn_normalization.TTNNRMSNorm",
    "device_constraint": null
  }
}
```

## Step 5 -- Generate Test Files

### 5a. Create test directory and __init__.py

```bash
mkdir -p tests/models/<model_name>
touch tests/models/<model_name>/__init__.py
```

### 5b. Tier 1 -- Op/Simple Module Tests

Generate `tests/models/<model_name>/test_ops_<model_name>.py` following the pattern from the PLAN phase. Each test:
1. Creates a PyTorch module with correct shape from shapes.json
2. `torch.set_grad_enabled(False)` and `.eval()`
3. `TTNNModule.from_torch(torch_module)`
4. `set_device(ttnn_module, device)`
5. Wraps input in `TorchTTNNTensor`
6. Runs both forward passes
7. `assert_pcc(ttnn_output, torch_output, threshold=0.999, msg="<op_name>")`

### 5c. Tier 2 -- Composite Module Tests

Generate `tests/models/<model_name>/test_composites_<model_name>.py`.

Two patterns are used for TTNN conversion:
- **Direct from_torch**: when an integration class provides complete TTNN replacement
  (e.g., `TTNNSelfAttention.from_torch(torch_attn)`)
- **register_modules**: when replacing leaf modules within a composite
  (e.g., replacing all `nn.Linear` inside an MLP with `TTNNLinear`)

### 5d. Tier 3 -- Decoder Layer Tests

Generate `tests/models/<model_name>/test_decoder_<model_name>.py` with both
prefill (seq_len=32,128) and decode (seq_len=1) tests.

### 5e. Tier 4 -- Full Model Test

Generate `tests/models/<model_name>/test_modeling_<model_name>.py`. Two paths:

**Path A (Recipe/Auto API)**: Uses `from tt_symbiote import AutoModelForCausalLM, set_device`.

**Path B (Manual replacement)**: Uses explicit `register_modules` calls with
`from tt_symbiote.utils.device_management import set_device`.

### 5f. Generate Device Guard Verification Test

Generate `tests/models/<model_name>/test_device_guards_<model_name>.py` to verify
`@run_on_devices` guards are present on all TTNN module forward() methods.

## Step 6 -- compare_fn_outputs Migration Guidance

After generating new tests, report existing tests that still use `compare_fn_outputs`:

- `tests/shared/test_attention.py` (2 call sites)
- `tests/shared/test_conv.py` (3 call sites)
- `tests/shared/test_moe.py` (1 call site)
- `tests/shared/test_rope.py` (4 call sites)

**ASK USER:** "These 4 existing test files use `compare_fn_outputs()` which only prints warnings and never asserts. Should I migrate them to use `assert_pcc()` now? (yes/no/later)"

If yes, for each file:
1. Replace `from tt_symbiote.core.utils import compare_fn_outputs` with `from tests.shared.pcc_utils import assert_pcc`
2. Replace `compare_fn_outputs(torch_out, ttnn_out, "Name")` with `assert_pcc(ttnn_out, torch_out, msg="Name")`
3. Note argument order difference: `compare_fn_outputs(torch, ttnn, name)` vs `assert_pcc(actual=ttnn, expected=torch, msg=name)`

## Step 7 -- Validate (Final VERIFY)

```bash
# Verify pytest can discover the tests
pytest --collect-only tests/models/<model_name>/ 2>&1 | tail -10

# Verify shapes.json is valid JSON
python -c "import json; json.load(open('tests/models/<model_name>/shapes.json'))"

# Verify no import errors
python -c "import tests.models.<model_name>"
```

If any validation fails, return to PLAN, diagnose, and iterate.

## Error Handling

| Problem | Resolution |
|---------|------------|
| Model not in HF hub | Ask user for local path; read source directly |
| No TTNN integration for a module | Flag to user, skip that tier, suggest creating one |
| `from_torch` signature varies | Read source of each `from_torch` to get correct params |
| Model requires specific transformers version | Add version check at top of generated test files |
| torch.* in generated forward() | A1 violation -- refactor to use ttnn.* equivalents |
