---
name: pcc-test-gen
description: Read a HuggingFace transformers model, decompose into all module tiers (ops, composites, decoder layers, full model), generate tiered PCC tests with explicit PCC assertions, and implement the assert_pcc utility. Generates shapes.json, op_map.json, and per-tier test files under tests/capabilities/<model_name>/.
---

# PCC Test Generation

Generate comprehensive tiered PCC tests for a HuggingFace model's TTNN bring-up.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/capabilities/<model_name>/test_modeling_<model_name>.py`

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.
  - `tests/models/` does NOT exist. Never create files there.
  - Shared capability tests: `tests/capabilities/` root (e.g., `test_attention.py`)
  - Auto/unit tests: `tests/auto/`

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
  Always use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.

**Deprecated API**: Never use `register_module_replacement_dict()`.
  Use `register_modules()` from `tt_symbiote.utils.module_replacement`.
  WARNING: Some existing tests (e.g., gemma4, qwen3_moe) still use the deprecated name. Do NOT copy those patterns.

**Import conventions**:
  - Integration modules: `from tt_symbiote.integrations.ttnn_<module> import TTNN<Class>`
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
   - `nn.Linear` -> `TTNNLinear` from `tt_symbiote.integrations.ttnn_linear`
   - `RMSNorm` -> `TTNNRMSNorm` from `tt_symbiote.integrations.ttnn_normalization`
   - `LayerNorm` -> `TTNNLayerNorm` from `tt_symbiote.integrations.ttnn_normalization`
   - Attention -> Check `ttnn_attention.py` (TTNNSelfAttention, TTNNFusedQKVSelfAttention, etc.)
   - MoE -> Check `ttnn_moe.py` (TTNNMoE, TTNNExperts, etc.)
   - If no integration exists, flag it to the user

## Step 3 -- Implement assert_pcc Utility

Check if `tests/capabilities/pcc_utils.py` still has the `NotImplementedError` stub:

```bash
grep "NotImplementedError" tests/capabilities/pcc_utils.py
```

If it does, replace the stub with the full implementation. **CRITICAL**: The implementation
MUST preserve the existing stub's function signature `(actual, expected, threshold=0.99, msg="")`
to avoid breaking any code that may already reference it.

Write this content to `tests/capabilities/pcc_utils.py`:

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

Create `tests/capabilities/<model_name>/shapes.json`:

```json
{
  "model_name": "<model_name>",
  "model_id": "<hf_model_id>",
  "device_arch": "<T3K|N150|etc>",
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

Also create `tests/capabilities/<model_name>/op_map.json`:
```json
{
  "TTNNLinear": {
    "ttnn_ops": ["ttnn.linear"],
    "used_by": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "import_path": "tt_symbiote.integrations.ttnn_linear.TTNNLinear",
    "device_constraint": null
  },
  "TTNNRMSNorm": {
    "ttnn_ops": ["ttnn.rms_norm"],
    "used_by": ["input_layernorm", "post_attention_layernorm", "norm"],
    "import_path": "tt_symbiote.integrations.ttnn_normalization.TTNNRMSNorm",
    "device_constraint": null
  }
}
```

## Step 5 -- Generate Test Files

### 5a. Create test directory and __init__.py

```bash
mkdir -p tests/capabilities/<model_name>
touch tests/capabilities/<model_name>/__init__.py
```

### 5b. Tier 1 -- Op/Simple Module Tests

Generate `tests/capabilities/<model_name>/test_ops_<model_name>.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 PCC tests: individual ops and simple modules for <model_name>."""

import json
import pytest
import torch
from pathlib import Path

from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.integrations.ttnn_linear import TTNNLinear
from tt_symbiote.integrations.ttnn_normalization import TTNNRMSNorm
from tt_symbiote.utils.device_management import set_device
from tests.capabilities.pcc_utils import assert_pcc

SHAPES = json.loads((Path(__file__).parent / "shapes.json").read_text())


@pytest.mark.parametrize("seq_len", SHAPES["test_shapes"]["seq_lengths"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_linear_q_proj(device, seq_len):
    """Q projection linear: PCC against PyTorch reference."""
    torch.set_grad_enabled(False)
    hidden_size = SHAPES["config"]["hidden_size"]
    num_heads = SHAPES["config"]["num_attention_heads"]
    head_dim = SHAPES["config"].get("head_dim", hidden_size // num_heads)

    torch_linear = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    torch_linear = torch_linear.to(torch.bfloat16).eval()

    input_tensor = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    torch_output = torch_linear(input_tensor)

    ttnn_linear = TTNNLinear.from_torch(torch_linear)
    set_device(ttnn_linear, device)

    ttnn_input = TorchTTNNTensor(input_tensor)
    ttnn_output = ttnn_linear(ttnn_input)

    assert_pcc(ttnn_output, torch_output, threshold=0.999, msg="q_proj_linear")


# Repeat for k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, rms_norm, etc.
# Each test follows the same pattern:
# 1. Create PyTorch module with correct shape from shapes.json
# 2. torch.set_grad_enabled(False) and .eval()
# 3. TTNNModule.from_torch(torch_module)
# 4. set_device(ttnn_module, device)
# 5. Wrap input in TorchTTNNTensor
# 6. Run both forward passes
# 7. assert_pcc(ttnn_output, torch_output, threshold=0.999, msg="<op_name>")
```

### 5c. Tier 2 -- Composite Module Tests

Generate `tests/capabilities/<model_name>/test_composites_<model_name>.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 PCC tests: composite modules (attention, MLP) for <model_name>."""

import copy
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.integrations.ttnn_linear import TTNNLinear
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules
from tests.capabilities.pcc_utils import assert_pcc

MODEL_ID = "<hf_model_id>"


def _create_minimal_model():
    """Create a 1-layer model with random weights for testing composites."""
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
    torch.set_grad_enabled(False)
    return model, config


def _create_position_embeddings(model, config, seq_len):
    """Generate rotary position embeddings for attention tests."""
    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    rotary_emb = getattr(model.model, 'rotary_emb', None)
    if rotary_emb is None:
        rotary_emb = getattr(model.model.layers[0].self_attn, 'rotary_emb', None)

    if rotary_emb is not None:
        cos, sin = rotary_emb(hidden, position_ids)
        return (cos, sin)
    return None


@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_attention(device, seq_len):
    """Attention block PCC: Q/K/V projections + SDPA + O projection."""
    model, config = _create_minimal_model()
    torch_attn = model.model.layers[0].self_attn
    position_embeddings = _create_position_embeddings(model, config, seq_len)

    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)

    torch_kwargs = {}
    if position_embeddings is not None:
        torch_kwargs["position_embeddings"] = position_embeddings
    torch_out = torch_attn(hidden, **torch_kwargs)
    torch_attn_out = torch_out[0] if isinstance(torch_out, tuple) else torch_out

    # Map to appropriate TTNN attention class from ttnn_attention.py
    # Read the source to find the right class for this model's attention pattern
    from tt_symbiote.integrations.ttnn_attention import TTNNSelfAttention
    ttnn_attn = TTNNSelfAttention.from_torch(torch_attn)
    set_device(ttnn_attn, device)

    ttnn_hidden = TorchTTNNTensor(hidden)
    ttnn_out = ttnn_attn(ttnn_hidden, **torch_kwargs)
    ttnn_attn_out = ttnn_out[0] if isinstance(ttnn_out, tuple) else ttnn_out

    assert_pcc(ttnn_attn_out, torch_attn_out, threshold=0.999, msg="attention")


@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_mlp(device, seq_len):
    """MLP/FFN block PCC: gate + up projections, activation, down projection."""
    model, config = _create_minimal_model()
    torch_mlp = model.model.layers[0].mlp

    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    torch_out = torch_mlp(hidden)

    # Replace MLP's linear layers with TTNN equivalents
    mlp_copy = copy.deepcopy(torch_mlp)
    register_modules(mlp_copy, {torch.nn.Linear: TTNNLinear})
    set_device(mlp_copy, device)

    ttnn_input = TorchTTNNTensor(hidden)
    ttnn_out = mlp_copy(ttnn_input)

    assert_pcc(ttnn_out, torch_out, threshold=0.999, msg="mlp")
```

### 5d. Tier 3 -- Decoder Layer Tests

Generate `tests/capabilities/<model_name>/test_decoder_<model_name>.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 PCC tests: full decoder layer for <model_name>."""

import copy
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.integrations.ttnn_linear import TTNNLinear
from tt_symbiote.integrations.ttnn_normalization import TTNNRMSNorm
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules
from tests.capabilities.pcc_utils import assert_pcc

MODEL_ID = "<hf_model_id>"


@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer(device, seq_len):
    """Full decoder layer PCC: attention + MLP + norms + residual."""
    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()

    torch_layer = model.model.layers[0]
    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # Generate position embeddings
    rotary_emb = getattr(model.model, 'rotary_emb', None)
    position_embeddings = None
    if rotary_emb is not None:
        cos, sin = rotary_emb(hidden, position_ids)
        position_embeddings = (cos, sin)

    # Run torch reference
    torch_kwargs = {"position_embeddings": position_embeddings} if position_embeddings else {}
    torch_out = torch_layer(hidden, **torch_kwargs)
    torch_hidden_out = torch_out[0] if isinstance(torch_out, tuple) else torch_out

    # Create TTNN decoder layer via module replacement
    ttnn_layer = copy.deepcopy(torch_layer)
    replacement_dict = {torch.nn.Linear: TTNNLinear}
    if hasattr(torch_layer, 'input_layernorm'):
        replacement_dict[type(torch_layer.input_layernorm)] = TTNNRMSNorm
    register_modules(ttnn_layer, replacement_dict)
    set_device(ttnn_layer, device)

    ttnn_input = TorchTTNNTensor(hidden)
    ttnn_out = ttnn_layer(ttnn_input, **torch_kwargs)
    ttnn_hidden_out = ttnn_out[0] if isinstance(ttnn_out, tuple) else ttnn_out

    assert_pcc(ttnn_hidden_out, torch_hidden_out, threshold=0.999, msg="decoder_layer")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer_decode_step(device):
    """Decoder layer with single-token decode (seq_len=1)."""
    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()

    torch_layer = model.model.layers[0]
    seq_len = 1
    past_seq_len = 64

    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.tensor([[past_seq_len]])

    rotary_emb = getattr(model.model, 'rotary_emb', None)
    position_embeddings = None
    if rotary_emb is not None:
        cos, sin = rotary_emb(hidden, position_ids)
        position_embeddings = (cos, sin)

    torch_kwargs = {"position_embeddings": position_embeddings} if position_embeddings else {}
    torch_out = torch_layer(hidden, **torch_kwargs)
    torch_hidden_out = torch_out[0] if isinstance(torch_out, tuple) else torch_out

    import copy
    ttnn_layer = copy.deepcopy(torch_layer)
    replacement_dict = {torch.nn.Linear: TTNNLinear}
    if hasattr(torch_layer, 'input_layernorm'):
        replacement_dict[type(torch_layer.input_layernorm)] = TTNNRMSNorm
    register_modules(ttnn_layer, replacement_dict)
    set_device(ttnn_layer, device)

    ttnn_input = TorchTTNNTensor(hidden)
    ttnn_out = ttnn_layer(ttnn_input, **torch_kwargs)
    ttnn_hidden_out = ttnn_out[0] if isinstance(ttnn_out, tuple) else ttnn_out

    assert_pcc(ttnn_hidden_out, torch_hidden_out, threshold=0.999, msg="decoder_layer_decode")
```

### 5e. Tier 4 -- Full Model Test

Generate `tests/capabilities/<model_name>/test_modeling_<model_name>.py`. Two paths:

**Path A (Recipe/Auto API)**:
```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 PCC test: full model end-to-end for <model_name> (Auto API path)."""

import os
import pytest
import torch
from transformers import AutoTokenizer
import ttnn

from tt_symbiote import AutoModelForCausalLM, set_device
from tt_symbiote.core.run_config import DispatchManager, TracedRun

MODEL_ID = "<hf_model_id>"

@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000000,
    "num_command_queues": 1}], indirect=True)
def test_full_model(mesh_device):
    """Full model generation test."""
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    set_device(model, mesh_device)
    model.eval()

    inputs = tokenizer("What is machine learning?", return_tensors="pt")
    inputs.pop("token_type_ids", None)

    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True)
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    assert len(decoded.strip()) > 0, "Generated output should not be empty"
    TracedRun.release_all()
```

**Path B (Manual replacement)**: Use explicit `register_modules` calls with
`from tt_symbiote.utils.device_management import set_device`.

### 5f. Generate Device Guard Verification Test

Generate `tests/capabilities/<model_name>/test_device_guards_<model_name>.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verify @run_on_devices guards on all TTNN modules for <model_name>."""

def test_device_guards_present():
    """All TTNNModule.forward methods must have @run_on_devices."""
    from tt_symbiote.integrations.ttnn_linear import TTNNLinear
    # Import all TTNN classes used by this model
    # ...

    classes_to_check = [TTNNLinear]  # Add all classes used

    missing = []
    for cls in classes_to_check:
        forward = getattr(cls, 'forward', None)
        if forward and not hasattr(forward, '__tt_allowed_archs__'):
            missing.append(cls.__name__)

    # NOTE: Some base classes (e.g., TTNNLinear) may not have guards --
    # only their device-specific subclasses do.
    if missing:
        import warnings
        warnings.warn(f"Modules missing @run_on_devices: {missing}")
```

## Step 6 -- compare_fn_outputs Migration Guidance

After generating new tests, report existing tests that still use `compare_fn_outputs`:

- `tests/capabilities/test_attention.py` (2 call sites)
- `tests/capabilities/test_conv.py` (3 call sites)
- `tests/capabilities/test_moe.py` (1 call site)
- `tests/capabilities/test_rope.py` (4 call sites)

**ASK USER:** "These 4 existing test files use `compare_fn_outputs()` which only prints warnings and never asserts. Should I migrate them to use `assert_pcc()` now? (yes/no/later)"

If yes, for each file:
1. Replace `from tt_symbiote.core.utils import compare_fn_outputs` with `from tests.capabilities.pcc_utils import assert_pcc`
2. Replace `compare_fn_outputs(torch_out, ttnn_out, "Name")` with `assert_pcc(ttnn_out, torch_out, msg="Name")`
3. Note argument order difference: `compare_fn_outputs(torch, ttnn, name)` vs `assert_pcc(actual=ttnn, expected=torch, msg=name)`

## Step 7 -- Validate

```bash
# Verify pytest can discover the tests
pytest --collect-only tests/capabilities/<model_name>/ 2>&1 | tail -10

# Verify shapes.json is valid JSON
python -c "import json; json.load(open('tests/capabilities/<model_name>/shapes.json'))"

# Verify no import errors
python -c "import tests.capabilities.<model_name>"
```

## Error Handling

| Problem | Resolution |
|---------|------------|
| Model not in HF hub | Ask user for local path; read source directly |
| No TTNN integration for a module | Flag to user, skip that tier, suggest creating one |
| `from_torch` signature varies | Read source of each `from_torch` to get correct params |
| Model requires specific transformers version | Add version check at top of generated test files |
