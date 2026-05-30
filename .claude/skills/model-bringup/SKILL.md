---
name: model-bringup
description: Interactive full-model bring-up orchestrator. Given a HuggingFace model, scaffolds directory structure, generates TTNNModule skeletons with @run_on_devices guards, creates the recipe for Auto API with instance methods, registers in _RECIPE_BEARING_SUBPACKAGES, and guides through skills 1-7 step by step.
---

# Full Model Bring-Up Orchestrator

End-to-end orchestration for bringing up a new HuggingFace model on Tenstorrent hardware.

## Conventions (apply to ALL artifacts this skill creates)

**File naming**: Model directories use HuggingFace `transformers` snake_case naming.
  - Model source: `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`
  - Model tests: `tests/capabilities/<model_name>/test_modeling_<model_name>.py`

**Test location**: All per-model tests go under `tests/capabilities/<model_name>/`.
  - `tests/models/` does NOT exist. Never create files there.

**Device guards**: ALL new `TTNNModule.forward()` methods MUST have `@run_on_devices`.
  - Import: `from tt_symbiote.core.module import run_on_devices, DeviceArch`
  - Default: `@run_on_devices(DeviceArch.T3K)`

**License headers**: Every generated `.py` file must start with:
  ```python
  # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
  # SPDX-License-Identifier: Apache-2.0
  ```

**PCC assertions**: Use `assert_pcc()` from `tests/capabilities/pcc_utils.py`.
  NEVER rely on `compare_fn_outputs()`.

**Deprecated API**: Never use `register_module_replacement_dict()`.
  Use `register_modules()` from `tt_symbiote.utils.module_replacement`.
  WARNING: Some existing tests (gemma4, qwen3_moe) still use the deprecated name. Do NOT copy.

**Import conventions**:
  - Integration modules: `from tt_symbiote.integrations.ttnn_<module> import TTNN<Class>`
  - Core: `from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch`
  - Tier 1-3 tests: `from tt_symbiote.utils.device_management import set_device`
  - Tier 4 (Auto API): `from tt_symbiote import AutoModelForCausalLM, set_device`

**Device/mesh_device fixtures**: Provided by the `ttnn` pytest plugin, NOT by tt_symbiote.
  Do NOT define these fixtures.

**Config system**: The typed config system (ModuleConfig, DtypeConfig, etc.) does NOT exist.
  Use `_model_config: dict` via `set_model_config()`, and subclass-based overrides.

**HuggingFace**: Always pass `trust_remote_code=True`.

## Overview

This skill orchestrates the ENTIRE model bring-up pipeline:

1. **Scaffold**: Create model directory, TTNNModule skeletons, recipe
2. **Test**: Generate PCC tests (invoke pcc-test-gen)
3. **Sweep**: Run parameter sweeps (invoke op-sweep)
4. **Profile**: Profile with tracy (invoke tracy-profiling)
5. **Analyze**: Analyze performance (invoke perf-analysis)
6. **Optimize**: Apply optimal configs (invoke config-optimize-all or config-optimize-module)
7. **Trace**: Enable traced execution (invoke traced-execution)

Each step is interactive -- Claude asks the user before proceeding.

## Phase 1: Model Discovery

### Step 1 -- Collect Model Information (ASK the user)

1. **HuggingFace model ID**: e.g., `meta-llama/Llama-3.1-8B`

2. **Derive canonical model name**:
   ```python
   from transformers import AutoConfig
   import importlib

   config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
   model_type = getattr(config, 'model_type', None)

   try:
       importlib.import_module(f"transformers.models.{model_type}")
       model_name = model_type
   except ImportError:
       # ASK USER: "Model name '<model_type>' not found in HF transformers. Proceed?"
       model_name = model_type
   ```
   Show derived name to user and confirm.

3. **Check for existing model directory**:
   ```bash
   ls -d src/tt_symbiote/models/<model_name> 2>/dev/null && echo "EXISTS" || echo "NEW"
   ```
   If exists, warn user. The models directory contains:
   - `bailing_moe_v2` (recipe-registered in _RECIPE_BEARING_SUBPACKAGES)
   - `gemma4` (NOT recipe-registered -- uses manual from_torch)
   - `qwen3_moe` (NOT recipe-registered -- uses manual from_torch)
   **ASK USER:** "Model directory already exists. Overwrite? Extend? Or use a different name?"

4. **Target device architecture**: Default T3K

5. **Integration path**:
   - **Recipe/Auto API** (recommended): `@register_recipe` + `build_module_dict()`
   - **Manual replacement**: `from_torch()` chain + explicit `register_modules`

### Step 2 -- Read Model Architecture

```python
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
```

Present analysis to user:
- Architecture summary (attention type, MLP type, norm type, MoE or not)
- Which existing tt_symbiote integration modules can be reused
- Which modules need NEW implementations
- Suggested replacement dict for `register_modules()`
- Recommended initial run mode: NORMAL_WITH_FALLBACK

## Phase 2: Directory Scaffolding

### Step 3 -- Create Model Directory

**ASK USER:** "Create model directory at `src/tt_symbiote/models/<model_name>/`? (yes/no)"

```bash
mkdir -p src/tt_symbiote/models/<model_name>
```

### Step 4 -- Generate modeling file

Create `src/tt_symbiote/models/<model_name>/modeling_<model_name>.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN bring-up for <HFModelClass>.

Model: <model_id>
Device target: T3K (8x Wormhole)
"""

from typing import Dict, Type

import torch
from torch import nn

from tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from tt_symbiote.integrations.ttnn_linear import TTNNLinear
from tt_symbiote.integrations.ttnn_normalization import TTNNRMSNorm
from tt_symbiote.utils.module_replacement import register_modules


class TTNN<ModelName>Attention(TTNNModule):
    """TTNN implementation of <ModelName>Attention."""

    @classmethod
    def from_torch(cls, torch_layer):
        new_module = cls()
        new_module._fallback_torch_layer = torch_layer
        # TODO: Extract weights, create child TTNN modules
        return new_module

    @run_on_devices(DeviceArch.T3K)  # TODO: Widen after N150/N300 validation
    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        raise NotImplementedError("Implement TTNN attention forward")


class TTNN<ModelName>MLP(TTNNModule):
    """TTNN implementation of <ModelName>MLP."""

    @classmethod
    def from_torch(cls, torch_layer):
        new_module = cls()
        new_module._fallback_torch_layer = torch_layer
        return new_module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        raise NotImplementedError("Implement TTNN MLP forward")


class TTNN<ModelName>DecoderLayer(TTNNModule):
    """TTNN implementation of a single <ModelName> decoder layer."""

    @classmethod
    def from_torch(cls, torch_layer):
        new_module = cls()
        new_module._fallback_torch_layer = torch_layer
        new_module.self_attn = TTNN<ModelName>Attention.from_torch(torch_layer.self_attn)
        new_module.mlp = TTNN<ModelName>MLP.from_torch(torch_layer.mlp)
        return new_module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        raise NotImplementedError("Implement decoder layer forward")


class TTNN<ModelName>Model(TTNNModule):
    """TTNN wrapper for the full <ModelName> model."""

    @classmethod
    def from_torch(cls, hf_model):
        # CRITICAL: register_modules BEFORE wrapping
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

**CRITICAL ordering**: `register_modules` BEFORE wrapping (matches bailing_moe_v2 pattern).

### Step 5 -- Generate Recipe (if Recipe/Auto API path)

Add to the modeling file:

```python
from tt_symbiote.auto.auto_mappings import register_recipe

# IMPORTANT: Recipe methods must be INSTANCE methods (with self), NOT @staticmethod.
# The @register_recipe decorator creates an instance, then calls instance methods.
# Reference: BailingMoEV2Recipe at models/bailing_moe_v2/modeling_bailing_moe_v2.py

@register_recipe(hf_class_name="<HFModelClass>ForCausalLM")
class <ModelName>Recipe:
    """Recipe for <model_name> model bring-up."""

    def build_module_dict(self, model) -> Dict[Type, Type]:
        """Return mapping of PyTorch classes to TTNN replacements."""
        return {
            type(model.model): TTNN<ModelName>Model,
            nn.Linear: TTNNLinear,
        }

    def post_register(self, model):
        """Post-registration hook."""
        type(model).device = property(lambda self: torch.device("cpu"))
        for module in model.modules():
            if isinstance(module, TTNNModule):
                module._bypass_tensor_wrapping = True

    def make_kv_cache(self, model, device, batch_size: int = 1, **kwargs):
        """Create KV cache (optional). Return None if not needed."""
        return None
```

### Step 6 -- Create __init__.py

Generate `src/tt_symbiote/models/<model_name>/__init__.py`:

```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""<model_name> model package."""

from tt_symbiote.models.<model_name>.modeling_<model_name> import (
    TTNN<ModelName>Model,
    TTNN<ModelName>Attention,
    TTNN<ModelName>MLP,
    TTNN<ModelName>DecoderLayer,
)
```

If using Recipe path, also import the Recipe class.

### Step 7 -- Register in _RECIPE_BEARING_SUBPACKAGES (Recipe path only)

**Only if the model has a `@register_recipe` decorator:**

Edit `src/tt_symbiote/models/__init__.py`:

```python
_RECIPE_BEARING_SUBPACKAGES = (
    "bailing_moe_v2",
    "<model_name>",  # NEW
)
```

NOTE: `gemma4` and `qwen3_moe` are correctly absent -- they do NOT have `@register_recipe`.

### Step 8 -- Create test directory

```bash
mkdir -p tests/capabilities/<model_name>
touch tests/capabilities/<model_name>/__init__.py
```

### Step 9 -- Verify Scaffold

```bash
python -c "from tt_symbiote.models.<model_name> import TTNN<ModelName>Model; print('Import OK')"

# If Recipe path:
python -c "
from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY
assert '<HFModelClass>ForCausalLM' in TT_MODEL_REGISTRY, 'Recipe not registered!'
print('Recipe registered')
"
```

## Phase 3: Skill Orchestration

### Step 10 -- Guide Through Skills

**ASK at each step:**

a. "Shall I generate PCC tests? (invoke: pcc-test-gen)"
   - This will implement `assert_pcc()` in pcc_utils.py and generate tiered tests

b. "Run PCC tests?" -> If issues, recommend DPL or DPL_NO_ERROR_PROP mode

c. "Sweep op parameters? (invoke: op-sweep) Which modules?"

d. "Profile sweep results? (invoke: tracy-profiling + perf-analysis)"

e. "Apply optimal configs? All modules or individual? (invoke: config-optimize-all or config-optimize-module)"

f. "Run traced execution validation? (invoke: traced-execution)"

g. "Run full model PCC test?"

After each step, ASK: "What would you like to do next?"

## Phase 4: Summary

Generate model bring-up summary:
- All files created/modified
- PCC results per tier
- Performance optimizations applied (which subclasses, which override methods)
- Traced execution status
- Next steps / known issues

## Error Handling

| Problem | Resolution |
|---------|------------|
| Model not in HF transformers | Use `trust_remote_code=True`; derive name from model_type |
| Model directory already exists | Warn user; ask to overwrite, extend, or rename |
| Recipe registration fails | Check __init__.py imports; check _RECIPE_BEARING_SUBPACKAGES |
| PCC failures during bring-up | Use DPL mode to isolate; check TTNN op compatibility |
| from_torch signature mismatch | Read source of each integration class's from_torch |
| Existing test uses deprecated API | Do NOT copy; use register_modules |
