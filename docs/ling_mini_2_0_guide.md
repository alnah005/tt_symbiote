# Ling-mini-2.0 in `tt_symbiote`

A complete guide to **(A)** how `inclusionAI/Ling-mini-2.0` is implemented as
the first reference port in `tt_symbiote`, and **(B)** how to run it on a
Tenstorrent T3K through the public `from tt_symbiote import
AutoModelForCausalLM` API.

> Audience: someone who has bootstrapped the standalone `tt_symbiote`
> venv via [`scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh) (or
> who develops against tt-metal HEAD per §B.1), has access to T3K
> hardware, and wants to either run the model or add a new one following
> the same pattern. No `tt-metal` source checkout is required for the
> standalone path.

---

## TL;DR — running the model

```python
import os
os.environ.setdefault("MESH_DEVICE", "T3K")

import torch
import ttnn
from transformers import AutoTokenizer
from tt_symbiote import AutoModelForCausalLM, set_device

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    trace_region_size=200_000_000,
    num_command_queues=1,
)

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0", trust_remote_code=True, dtype="auto",
)
set_device(model, mesh_device)
assert hasattr(model, "_tt_kv_cache")

model.eval()
torch.set_grad_enabled(False)

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is your favorite condiment?"}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
).to(model.device)
inputs.pop("token_type_ids", None)

out = model.generate(
    **inputs,
    max_new_tokens=128,
    use_cache=True,
    past_key_values=model._tt_kv_cache,
)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:]))

ttnn.close_mesh_device(mesh_device)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
```

That's the entire user surface. Three calls do all the heavy lifting:

1. `AutoModelForCausalLM.from_pretrained(...)` — loads the HF model and
   swaps its modules for TTNN equivalents (via the registered recipe).
2. `set_device(model, mesh_device)` — binds every TTNN module to the mesh,
   preprocesses/moves weights, allocates the paged KV cache.
3. `model.generate(...)` — vanilla HF generation, accelerated end-to-end.

---

# Part A — Implementation

## A.1 — Architecture overview

`tt_symbiote` is structurally a thin shell around `transformers` that
injects TTNN modules into the HF model graph at load time. Three concepts
make that possible:

| Concept | Where it lives | Job |
| --- | --- | --- |
| **`Auto*` factory** | `tt_symbiote/models/auto/auto_factory.py` | Mirrors every `transformers.Auto*` class. After HF loads the model, looks up the registered recipe and applies it. |
| **`Recipe`** | One per model, decorated with `@register_recipe(hf_class_name=...)` | Declares **which** PyTorch classes to swap for **which** TTNN classes, plus optional post-load patches and an optional KV-cache builder. |
| **`set_device`** | `tt_symbiote/utils/device_management.py` | Mandatory device-binding step. Walks the model graph, binds TTNN modules to the mesh, calls `preprocess_weights` / `move_weights_to_device`, and invokes `Recipe.make_kv_cache` if present. |

The contract from the user's point of view is exactly two extra calls
compared to vanilla `transformers`: an `Auto*.from_pretrained` (with the
`tt_symbiote` namespace instead of `transformers`) and a `set_device(...)`.

## A.2 — File layout for Ling-mini-2.0

```
src/tt_symbiote/
├── __init__.py                            # eager-imports tt_symbiote.models -> triggers recipe registration
├── _hf_compat.py                          # compat shims for `transformers` API drift
├── auto/
│   ├── auto_factory.py                    # _BaseAutoModelClass.from_pretrained (recipe dispatch)
│   └── auto_mappings.py                   # Recipe protocol + register_recipe decorator + TT_MODEL_REGISTRY
├── models/
│   ├── __init__.py                        # imports every recipe-bearing model subpackage
│   └── bailing_moe_v2/
│       ├── __init__.py                    # imports modeling_bailing_moe_v2 -> fires @register_recipe
│       └── modeling_bailing_moe_v2.py     # TTNN modules + BailingMoEV2Recipe
└── utils/
    └── device_management.py               # set_device + KV-cache provisioning
```

## A.3 — The recipe

The Ling-mini-2.0 recipe is the file
`src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py`. The
entire public-facing surface is the trailing 25 lines:

```610:633:src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py
@register_recipe(hf_class_name="BailingMoeV2ForCausalLM")
class BailingMoEV2Recipe:
    def build_module_dict(self, model):
        return {
            type(model.model): TTNNBailingMoeV2Model,
            nn.Linear: TTNNLinearIColShardedWRowSharded,
        }

    def post_register(self, model):
        type(model).device = property(lambda self: torch.device("cpu"))

    def make_kv_cache(self, model, device, batch_size: int = 1, **kwargs):
        config = PagedAttentionConfig(
            block_size=kwargs.get("block_size", 64),
            max_num_blocks=kwargs.get("max_num_blocks", 32),
            batch_size=batch_size,
        )
        return TTNNPagedAttentionKVCache(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            config=config,
            device=None,
        ).to_device(device)
```

Three responsibilities, one per method:

### `build_module_dict(model)` — what to swap

A **single flat dict** keyed by the PyTorch class, valued by its TTNN
replacement. The dict is consumed in one pass by
`tt_symbiote.utils.module_replacement.register_modules`, applied to the
root `BailingMoeV2ForCausalLM` instance. For Ling-mini-2.0 only two outer
swaps are declared:

- `type(model.model) → TTNNBailingMoeV2Model` swaps the outer
  `BailingMoeV2Model` wrapper.
- `nn.Linear → TTNNLinearIColShardedWRowSharded` catches the bare
  `lm_head` at the root.

The decoder layers, the final RMS norm, `nn.Embedding`, and the rotary
embedding are **not** in this dict — `TTNNBailingMoeV2Model.from_torch`
owns the conversion of everything inside its own subtree (see A.4).

This is the **Option 1 / single-pass** contract resolved in docs/internal/PROJECT_PROPOSAL.md
OQ-2: each TTNN wrapper class is responsible for the conversion of its
own children. Recipes only declare outer-level swaps. See
[`docs/internal/migration_notes.md`](./internal/migration_notes.md) §Phase 5 for the
options that were considered and why Option 1 won.

### `post_register(model)` — patch the model

Runs immediately after `register_modules` completes. For Ling-mini-2.0
the only patch is rewriting the `model.device` property:

```python
type(model).device = property(lambda self: torch.device("cpu"))
```

HF's `generate` calls `self.device` to decide where to place the
`input_ids` tensor it constructs internally. After replacement, the
model has no torch parameters left, so the default implementation
(`next(self.parameters()).device`) would raise `StopIteration`. We
return CPU so HF builds CPU tensors; the TTNN modules accept those
without complaint.

### `make_kv_cache(model, device, **kwargs)` — paged attention

Optional hook (docs/internal/PROJECT_PROPOSAL.md OQ-9 resolution). Returns a
`TTNNPagedAttentionKVCache` configured from `model.config`:

- `num_layers = config.num_hidden_layers`
- `num_kv_heads = config.num_key_value_heads`
- `head_dim = config.head_dim`
- `block_size` / `max_num_blocks` / `batch_size` from kwargs (sane
  defaults: 64 / 32 / 1)

`set_device` calls this *after* the device is bound and weights are
moved, then attaches the result as `model._tt_kv_cache`. The user
passes it as `past_key_values=model._tt_kv_cache` to `model.generate`.

The `**kwargs` that reach `make_kv_cache` are sourced (in order, last
wins per-key) from:

1. `model._tt_kv_cache_kwargs`, which `AutoModelForCausalLM.from_pretrained`
   populates from its `kv_cache_kwargs=` keyword. This is the
   recommended path — the cache shape is a model-config decision that
   pairs naturally with model loading.
2. `kwargs["kv_cache_kwargs"]` on the `set_device` call site, kept as
   an escape hatch for A/B-testing different cache budgets without
   reloading the model.

If a recipe has no `make_kv_cache` (most non-LM models), the no-op
default installed by the `@register_recipe` decorator quietly returns
`None` and `model._tt_kv_cache` is never set.

## A.4 — Single-pass subtree conversion (`TTNNBailingMoeV2Model.from_torch`)

The interesting part of Option 1 is that wrappers *own their subtree*.
Here is how `TTNNBailingMoeV2Model` adopts the HF model and rewrites its
children:

```70:97:src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py
        register_modules(
            hf_model,
            {
                type(hf_model.layers[0]): TTNNBailingMoEDecoderLayerPadded,
                type(hf_model.norm): TTNNDistributedRMSNorm,
                nn.Embedding: TTNNBailingPaddedEmbedding,
                type(hf_model.rotary_emb): TTNNBailingRotaryEmbedding,
            },
            model_config=None,
        )

        new_model = cls()
        new_model._fallback_torch_layer = hf_model
        new_model.model = hf_model

        # Bypass tensor wrapping/unwrapping for decoder layers.
        # These sit under the HF BailingMoeV2Model (nn.Module), so
        # set_device() would give them _bypass_tensor_wrapping=False.
        # Bypassing is safe: no PyTorch ops touch hidden_states between
        # layer calls, and each layer's forward already works with raw
        # ttnn.Tensor objects.
        for layer in hf_model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        if isinstance(hf_model.norm, TTNNModule):
            hf_model.norm._bypass_tensor_wrapping = True

        return new_model
```

What happened here, in order:

1. **Inner `register_modules` call**: rewrites the HF
   `BailingMoeV2Model`'s children — every `BailingMoeV2DecoderLayer`
   becomes `TTNNBailingMoEDecoderLayerPadded`, `BailingMoeV2RMSNorm`
   becomes `TTNNDistributedRMSNorm`, etc. This is the conversion that
   the *outer* recipe explicitly does **not** declare.
2. **Wrap and adopt**: a fresh `TTNNBailingMoeV2Model` is created, the
   HF model is stashed as `_fallback_torch_layer` (used if the device
   architecture isn't supported — see `set_device` step 2) and also as
   `.model` (used by `call()` to access config, layers, etc.).
3. **Bypass tensor wrapping** for inner TTNN modules: because the
   decoder layers sit *under* an `nn.Module` (the HF wrapper), the
   default `set_device` pass would mark them
   `_bypass_tensor_wrapping=False`. We override that — the decoder
   `call()` already works with raw `ttnn.Tensor` objects and never
   touches PyTorch ops between layers.

The net effect: after `BailingMoEV2Recipe.build_module_dict` declares
two outer swaps, the inner `from_torch` call rewires 20 decoder layers,
1 final norm, 1 embedding, and 1 rotary in a single sweep. No
multi-pass orchestration anywhere in user-visible code.

## A.5 — `set_device(model, mesh_device)`

`set_device` (`src/tt_symbiote/utils/device_management.py`) is the
mandatory final step. It does six things in order:

1. **Walks the model graph**, collecting every `TTNNModule`.
2. **Architecture-gates each module** via
   `forward.__tt_allowed_archs__`. If the active arch (resolved from
   `MESH_DEVICE`) isn't allowed, the module is swapped in place with
   its `_fallback_torch_layer` and a warning is logged.
3. **`to_device(device)`** and (for multi-device meshes)
   `set_device_state(...)` on every remaining TTNN module.
4. **`preprocess_weights()`** then **`move_weights_to_device()`** on
   every TTNN module. This subsumes the per-test loop that callers
   previously had to write by hand.
5. **`make_kv_cache(...)`**: if a recipe is registered for
   `type(obj).__name__` and exposes `make_kv_cache`, the result is
   built and attached as `obj._tt_kv_cache` (docs/internal/PROJECT_PROPOSAL.md Q9).
6. **Marks the model**: `_tt_symbiote_device_set = True` on the root
   and on every visited TTNN module.

Hard-error contract: `run_config.module_run` asserts
`self._device is not None` with a message naming `set_device`, so a
forward called before `set_device` fails with a clear diagnostic
instead of an inscrutable `AttributeError`.

## A.6 — Top-level registration (HF-style side effect)

For `AutoModelForCausalLM.from_pretrained` to find the recipe, the
`@register_recipe` decorator must have run before that call. We do this
the same way `transformers` does — at top-level package import:

```94:103:src/tt_symbiote/__init__.py
try:
    from tt_symbiote import models as _models  # noqa: F401
except Exception as _e:  # pragma: no cover - exercised on broken installs
    import warnings as _warnings

    _warnings.warn(
        f"tt_symbiote: failed to load model recipes ({type(_e).__name__}: {_e}); "
        f"AutoModel*.from_pretrained will fall back to unmodified HF models.",
        stacklevel=2,
    )
```

Importing `tt_symbiote.models` cascades to
`tt_symbiote.models.bailing_moe_v2`, which imports
`modeling_bailing_moe_v2`, which runs the
`@register_recipe(hf_class_name="BailingMoeV2ForCausalLM")` decorator,
populating `TT_MODEL_REGISTRY["BailingMoeV2ForCausalLM"]` =
`BailingMoEV2Recipe` instance. Each per-model import is wrapped in
`try/except` so a broken model file warns but doesn't poison the whole
`tt_symbiote` import.

## A.7 — Inside `Auto*.from_pretrained`

Everything wired together:

```37:70:src/tt_symbiote/models/auto/auto_factory.py
    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the HF model, apply the tt_symbiote recipe if one is registered."""
        if cls._HF_AUTO_CLASS is None:
            raise NotImplementedError(
                f"{cls.__name__} has no HF counterpart configured (set _HF_AUTO_CLASS)."
            )

        # Install compat shims before HF's dynamic remote-code loader runs:
        # Hub modeling files authored against older transformers releases
        # frequently import symbols (e.g. ``is_torch_fx_available``) that
        # have since been removed. See ``tt_symbiote/utils/hf_compat.py``.
        from tt_symbiote.utils.hf_compat import install_transformers_shims

        install_transformers_shims()

        model = cls._HF_AUTO_CLASS.from_pretrained(pretrained_name_or_path, *args, **kwargs)

        hf_class_name = type(model).__name__
        recipe = TT_MODEL_REGISTRY.get(hf_class_name)
        if recipe is None:
            warnings.warn(
                f"No tt_symbiote recipe for {hf_class_name!r}; returning unmodified HF model. "
                f"set_device() will be a no-op for this model.",
                stacklevel=2,
            )
            return model

        module_dict = recipe.build_module_dict(model)
        register_modules(model, module_dict)
        recipe.post_register(model)
        # Marker read by tt_symbiote.set_device for the hard-error contract.
        model._tt_symbiote_has_recipe = True
        return model
```

Five steps, in order:

1. Install `_hf_compat` shims (see A.8).
2. Delegate to the real HF `Auto*` for download, weight load,
   tokenizer hookup, etc.
3. Look up `TT_MODEL_REGISTRY[type(model).__name__]`. If empty, warn
   and return the unmodified HF model — `set_device` becomes a no-op.
4. Apply the recipe: `register_modules(model, recipe.build_module_dict(model))`,
   then `recipe.post_register(model)`.
5. Stamp `_tt_symbiote_has_recipe = True` so `set_device` can hard-error
   if invoked on a model with no recipe.

## A.8 — `_hf_compat`: shim layer for upstream API drift

The Ling-mini-2.0 Hub modeling file (loaded via
`trust_remote_code=True`) was authored against `transformers ≈ 4.x`,
but `tt_symbiote` pins `transformers == 5.9.0`. Two symbols the Hub
file imports were removed upstream between those versions. Rather than
asking the Hub author to update the file, `tt_symbiote/utils/hf_compat.py`
re-installs the missing symbols **before** HF's dynamic remote-code
loader runs.

| Shim | Why | Replacement |
| --- | --- | --- |
| `transformers.utils.import_utils.is_torch_fx_available` | Removed between 4.x and 5.x. The Hub file imports it as a feature gate before `torch.fx.wrap(_prepare_4d_causal_attention_mask)`. | `lambda: hasattr(torch, "fx")` (always `True` on modern PyTorch). |
| `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"]` | The legacy unscaled-RoPE entry was dropped from the dict but the Hub file falls back to `self.rope_type = "default"` when `config.rope_scaling is None`, then raises `KeyError`. | Canonical formula `inv_freq = 1 / base ** (arange(0, dim, 2) / dim)`, reading config via `getattr` to stay compatible with the legacy config shape. |

Both shims are guarded (`if not hasattr / if key not in dict`), so
they're no-ops if a future `transformers` release brings the originals
back. The installer is also idempotent — gated by a module-level
`_INSTALLED` flag.

**Extension point**: when porting a new remote-code model that fails
on a missing/moved upstream symbol, add a guarded entry to
`install_transformers_shims` in `hf_compat.py` — that's the entire
escape hatch.

---

# Part B — Running it

## B.1 — Prerequisites

1. **Hardware**: Tenstorrent T3K (1×8 mesh, 8 chips). Other meshes
   work too but the recipe was validated on T3K.

2. **System sfpi toolchain** at `/opt/tenstorrent/sfpi/`. This is the
   apt-installable Tenstorrent RISC-V compiler that ttnn uses to
   JIT-compile firmware kernels at first device-open time. It is a
   *system* prerequisite (same role CUDA plays for GPU users), not a
   tt-metal dependency. If you don't have it, install via
   Tenstorrent's official installer / apt repo:

   ```bash
   /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version
   # → riscv-tt-elf-g++ (tenstorrent/sfpi:7.35.3[426]) 15.1.0
   ```

   The version string after `tenstorrent/sfpi:` must match the
   `SFPI_REQUIRED` line in
   [`scripts/ttnn-pin.txt`](../scripts/ttnn-pin.txt); the bootstrap
   script verifies this and bails with a clear message if not.

3. **Bootstrap the venv** — one command:

   ```bash
   cd /home/aroberge/tt_symbiote
   ./scripts/bootstrap_venv.sh
   source .venv/bin/activate
   ```

   `scripts/bootstrap_venv.sh` reads `scripts/ttnn-pin.txt`, probes the
   system sfpi, creates `tt_symbiote/.venv`, then `pip install`s
   `ttnn==<pinned>`, `torch`, `transformers==5.9.0`, and `tt_symbiote`
   (editable). The smoke check at the end confirms
   `AutoModelForCausalLM`, `set_device`, and
   `TT_MODEL_REGISTRY["BailingMoeV2ForCausalLM"]` are all wired up.

   **No tt-metal source checkout or tt-metal Python env is required.**
   Verified end-to-end on T3K: Ling-mini-2.0 loads, binds across the
   mesh, and generates coherent text from a fresh venv built solely
   by this script.

4. **HF auth (optional)**: `inclusionAI/Ling-mini-2.0` is public, but
   set `HF_TOKEN` to avoid rate limits on big downloads.

### Alternative — develop against tt-metal HEAD

If you're actively developing tt-metal and want to use *its*
locally-built `ttnn` (instead of the PyPI wheel), skip
`bootstrap_venv.sh` and install into tt-metal's venv directly:

```bash
source /home/aroberge/tt-metal/python_env/bin/activate
cd /home/aroberge/tt_symbiote && pip install -e .
```

This works but has two trade-offs: it upgrades transformers in
tt-metal's venv to 5.9.0 (tt-metal pins `4.53.0`, which can break
tt-metal's own model tests in that venv), and you're now tied to
whichever `ttnn` snapshot tt-metal built — not the pinned PyPI
version. If `_ttnncpp.so` and tt-metal's Python sources fall out of
sync, rebuild with `ninja -C build_Release install` from the tt-metal
root.

## B.2 — The runnable script

The canonical runnable example is in the repo at
[`examples/e2e/run_ling_mini_2_0.py`](../examples/e2e/run_ling_mini_2_0.py):

```python
import os
os.environ.setdefault("MESH_DEVICE", "T3K")

import torch
import ttnn
from transformers import AutoTokenizer
from tt_symbiote import AutoModelForCausalLM, set_device

# Fabric config must be set BEFORE open_mesh_device in current tt-metal HEAD;
# `fabric_config=` is no longer a kwarg to `open_mesh_device` (it was removed
# in favor of the `ttnn.set_fabric_config(...)` setter).
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    trace_region_size=200_000_000,
    num_command_queues=1,
)

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0",
    trust_remote_code=True,
    dtype="auto",
)
set_device(model, mesh_device)
assert hasattr(model, "_tt_kv_cache"), "Phase 5 recipe should have allocated this"

model.eval()
torch.set_grad_enabled(False)

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is your favorite condiment?"}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
).to(model.device)
inputs.pop("token_type_ids", None)

out = model.generate(
    **inputs,
    max_new_tokens=128,
    use_cache=True,
    past_key_values=model._tt_kv_cache,
)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:]))

ttnn.close_mesh_device(mesh_device)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
```

Run it (assumes you've already bootstrapped a venv per §B.1):

```bash
source /home/<you>/tt_symbiote/.venv/bin/activate
python examples/e2e/run_ling_mini_2_0.py
```

## B.3 — Line-by-line walkthrough

1. **`os.environ["MESH_DEVICE"] = "T3K"`** — read by `tt_symbiote`'s
   architecture gate inside `set_device` (`MeshShapeToDeviceArch`) and
   by the pytest fixture in
   `tests/capabilities/bailing_moe_v2/test_modeling_bailing_moe_v2.py`.
   Other accepted values: `N150`, `N300`, `N150x4`, `TG`, `P150`,
   `P300`, `P150x4`, `P150x8`, `BHGLX`.

2. **`ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)`** —
   must come **before** `open_mesh_device`. Older `tt-metal` releases
   accepted `fabric_config=` as a keyword on `open_mesh_device`; that
   was removed in favor of this setter. Mirrors how `tt-metal`'s pytest
   `mesh_device` fixture handles fabric in `tt-metal/conftest.py::set_fabric`.

3. **`ttnn.open_mesh_device(...)`** — opens the 1×8 T3K mesh with a
   200 MB trace region and a single command queue. Returns a
   `MeshDevice` that `set_device` will bind every TTNN module to.

4. **`AutoTokenizer.from_pretrained(... trust_remote_code=True)`** —
   needed because Ling ships a custom tokenizer class on the Hub.

5. **`AutoModelForCausalLM.from_pretrained(...)`** — the only line in
   this script that uses `tt_symbiote`'s namespace. Under the hood:
   - `_hf_compat.install_transformers_shims()` registers
     `is_torch_fx_available` and `ROPE_INIT_FUNCTIONS["default"]`.
   - HF downloads / executes the Hub modeling file, builds
     `BailingMoeV2ForCausalLM` on CPU.
   - `TT_MODEL_REGISTRY["BailingMoeV2ForCausalLM"]` =
     `BailingMoEV2Recipe` instance (populated at import time).
   - `recipe.build_module_dict(model)` returns the 2-entry outer dict.
   - `register_modules(model, ...)` applies it in one pass; the
     embedded `TTNNBailingMoeV2Model.from_torch` does the inner sweep
     for decoder layers / norm / embedding / rotary.
   - `recipe.post_register(model)` patches `model.device` to CPU.
   - Returns a `BailingMoeV2ForCausalLM` whose internals are TTNN
     modules but whose Python type is unchanged (so HF's `generate`
     still works).

6. **`set_device(model, mesh_device)`** — binds every TTNN module to
   the mesh, calls `preprocess_weights` and `move_weights_to_device`
   on each, invokes `BailingMoEV2Recipe.make_kv_cache(model,
   mesh_device)`, and attaches the result as `model._tt_kv_cache`.
   The `assert hasattr(model, "_tt_kv_cache")` immediately after is a
   sanity check that the Phase 5 KV-cache hook actually ran.

7. **`tokenizer.apply_chat_template(...).to(model.device)`** — builds
   CPU tensors (because `post_register` made `model.device` return
   CPU). The TTNN modules accept CPU input tensors and handle the
   host-to-device transfer themselves.

8. **`model.generate(..., past_key_values=model._tt_kv_cache)`** —
   plain HF `generate`. Because the modules are TTNN, the entire
   forward pass runs on the mesh; only token sampling/branching runs
   on the host. The paged KV cache lives on-device for the full
   generation.

9. **Teardown**: `close_mesh_device` then
   `set_fabric_config(DISABLED)`. The order matters — disabling
   fabric while the mesh is still open leaves the next process with
   inconsistent state.

## B.4 — Expected output

```
TTNNBailingMoEDecoderLayerPadded: model.model._modules[layers]._modules[0] on device MeshDevice(1x8 grid, 8 devices)
TTNNBailingMoEDecoderLayer: model.model._modules[layers]._modules[0].layer on device MeshDevice(1x8 grid, 8 devices)
TTNNDistributedRMSNorm: model.model._modules[layers]._modules[0].layer.input_layernorm on device MeshDevice(1x8 grid, 8 devices)
TTNNBailingMoEAttention: model.model._modules[layers]._modules[0].layer.attention on device MeshDevice(1x8 grid, 8 devices)
...   (one block like this per layer, 0..19, plus norms / linears / MoE / experts ...)
TTNNDistributedRMSNorm: model.model._modules[norm] on device MeshDevice(1x8 grid, 8 devices)
TTNNLinearIColShardedWRowSharded: lm_head on device MeshDevice(1x8 grid, 8 devices)

As an AI, I don't have personal preferences or taste buds, so I don't have a favorite condiment.
However, I can provide information on various condiments and their uses if you're interested!<|role_end|>
```

The trailing `nanobind: leaked N instances!` block at process exit is a
known upstream `ttnn` teardown quirk and does not affect correctness or
the exit code.

## B.5 — Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `AttributeError: partially initialized module 'ttnn' has no attribute 'global_avg_pool2d'` | Stale `_ttnncpp.so` (built before a Python source update in `tt-metal/ttnn/operations/pool.py`). | `cd /home/aroberge/tt-metal && ninja -C build_Release install` |
| `TypeError: open_mesh_device() got an unexpected keyword argument 'fabric_config'` | Code uses the old `tt-metal` API. | Use `ttnn.set_fabric_config(...)` **before** `open_mesh_device`, drop `fabric_config=` from the kwargs. |
| `ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'` | `_hf_compat.install_transformers_shims()` did not run (e.g. you called `transformers.AutoModelForCausalLM.from_pretrained` directly instead of going through `tt_symbiote`). | Call `from tt_symbiote.utils.hf_compat import install_transformers_shims; install_transformers_shims()` once before any HF auto factory load, **or** use `tt_symbiote.AutoModelForCausalLM` (which installs the shims automatically). |
| `KeyError: 'default'` inside `BailingMoeV2RotaryEmbedding.__init__` | Same root cause as above (shims not installed). | Same fix as above. |
| Warning: `No tt_symbiote recipe for 'BailingMoeV2ForCausalLM'; returning unmodified HF model.` | Top-level eager import of `tt_symbiote.models` failed silently. Check `import tt_symbiote` for warnings. | Investigate the warning emitted at import time; it carries the original exception. |
| `set_device` raises `_device is not None` failure during `model.generate` | `set_device` was never called on this model. | Add `set_device(model, mesh_device)` immediately after `from_pretrained`. |

---

# Part C — Reference

## C.1 — Public API

| Symbol | Source | Purpose |
| --- | --- | --- |
| `tt_symbiote.AutoModelForCausalLM` (and 42 other `Auto*` classes) | `tt_symbiote/models/auto/auto_factory.py` | Drop-in replacement for `transformers.Auto*`; applies a recipe if one is registered for the loaded HF class. |
| `tt_symbiote.set_device(model, device, **kwargs)` | `tt_symbiote/utils/device_management.py` | Mandatory device-binding step. Binds, preprocesses weights, allocates KV cache. |
| `tt_symbiote.register_modules(model, dict, model_config=None)` | `tt_symbiote/utils/module_replacement.py` | Lower-level utility used by both the recipe dispatch and wrappers like `TTNNBailingMoeV2Model.from_torch`. |
| `tt_symbiote.register_recipe(hf_class_name)` | `tt_symbiote/models/auto/auto_mappings.py` | Decorator. Installs a recipe class instance in `TT_MODEL_REGISTRY`. |
| `tt_symbiote.Recipe` | `tt_symbiote/models/auto/auto_mappings.py` | Runtime-checkable Protocol describing what a recipe must expose. |
| `tt_symbiote.TT_MODEL_REGISTRY` | `tt_symbiote/models/auto/auto_mappings.py` | Read-only-ish dict mapping HF class name → recipe instance. |

## C.2 — Where to extend

To add a new model `Foo` (HF class `FooForCausalLM`):

1. Create `src/tt_symbiote/models/foo/modeling_foo.py` with TTNN
   replacements for every PyTorch class you want swapped, plus a
   recipe class:
   ```python
   @register_recipe(hf_class_name="FooForCausalLM")
   class FooRecipe:
       def build_module_dict(self, model):
           return { type(model.model): TTNNFooModel, nn.Linear: TTNNLinear, ... }

       def post_register(self, model):
           ...  # optional patches (model.device fix, etc.)

       def make_kv_cache(self, model, device, **kwargs):
           ...  # optional, only if you need a paged cache or similar
   ```
2. Create `src/tt_symbiote/models/foo/__init__.py` that re-imports
   from `modeling_foo` so the `@register_recipe` decorator fires:
   ```python
   from tt_symbiote.models.foo.modeling_foo import FooRecipe, TTNNFooModel
   ```
3. Add `"foo"` to the eager-import list in
   `src/tt_symbiote/models/__init__.py`.
4. If the Hub modeling file fails to import on transformers 5.9.0,
   add a guarded entry to
   [`src/tt_symbiote/utils/hf_compat.py::install_transformers_shims`](../src/tt_symbiote/utils/hf_compat.py).
5. Add a hardware-free test
   `tests/auto/test_foo_recipe.py` along the lines of
   `tests/auto/test_ling_recipe.py`, and a hardware smoke test
   `tests/capabilities/foo/test_modeling_foo.py`.

## C.3 — File pointers (the whole story in 7 files)

- [`src/tt_symbiote/__init__.py`](../src/tt_symbiote/__init__.py) — eager-imports `tt_symbiote.models`.
- [`src/tt_symbiote/models/__init__.py`](../src/tt_symbiote/models/__init__.py) — imports each recipe subpackage.
- [`src/tt_symbiote/models/bailing_moe_v2/__init__.py`](../src/tt_symbiote/models/bailing_moe_v2/__init__.py) — fires `@register_recipe`.
- [`src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py`](../src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py) — TTNN modules + `BailingMoEV2Recipe` (lines 610–633).
- [`src/tt_symbiote/models/auto/auto_factory.py`](../src/tt_symbiote/models/auto/auto_factory.py) — `Auto*.from_pretrained` recipe dispatch.
- [`src/tt_symbiote/utils/device_management.py`](../src/tt_symbiote/utils/device_management.py) — `set_device` (the six-step mandatory binding pass).
- [`src/tt_symbiote/utils/hf_compat.py`](../src/tt_symbiote/utils/hf_compat.py) — `is_torch_fx_available` + `ROPE_INIT_FUNCTIONS["default"]` shims.

For deeper background on why each piece looks the way it does, see
[`docs/internal/migration_notes.md`](./internal/migration_notes.md) §Phase 5 and
[`docs/internal/PROJECT_PROPOSAL.md`](../docs/internal/PROJECT_PROPOSAL.md) §§4 (public API) and §10
(dependency policy).
