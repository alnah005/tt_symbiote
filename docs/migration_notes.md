# Phase 2 migration notes

Working record of decisions and discoveries that happened while executing the
Phase 2 mechanical migration. The plan lives in (the conversation's plan file);
this file is for things future maintainers should know that the plan did not
fully anticipate.

## Migration source

- Source repo:    `tt-metal`
- Source branch:  `alnah005/tt_symbiote_v2`
- Source SHA:     `43d758d972e7f3f3610236723ea22dfdbcecbc85`
- Source path:    `models/experimental/tt_symbiote/`
- Migration date: 2026-05-27
- Target branch:  `aroberge/bootstrap` (new repo, no shared history)

The same SHA is recorded as a "Vendored from" comment in every vendored file
(`src/tt_symbiote/core/{ccl.py, arch.py}` and
`src/tt_symbiote/integrations/tt_cnn/{builder, executor, pipeline}.py`).

## Dependency pinning policy

The branch `aroberge/bootstrap` targets `transformers==5.9.0`. The
`[project.dependencies]` block in `pyproject.toml` is structured so that
**every dependency other than `transformers`** mirrors the specifier used by
HF transformers v5.9.0's own
[`setup.py`](https://github.com/huggingface/transformers/blob/v5.9.0/setup.py)
verbatim:

- `transformers==5.9.0` — strict equality (our pin).
- `torch>=2.4`, `accelerate>=1.1.0` — promoted from HF's optional `torch` extra
  into our hard deps since tt_symbiote always imports torch.
- `huggingface-hub>=1.5.0,<2.0`, `numpy>=1.17`, `packaging>=20.0`,
  `pyyaml>=5.1`, `regex>=2025.10.22`, `tokenizers>=0.22.0,<=0.23.0`, `typer`,
  `safetensors>=0.4.3`, `tqdm>=4.27` — all from HF's `install_requires`,
  same specifiers verbatim.
- `loguru>=0.6.0` — tt_symbiote-specific (vendored arch helpers).

HF themselves do not use exact pins on any of these (they use `>=` lower
bounds and capped ranges), because exact pins would conflict with the
platform-specific wheel sets of torch / numpy / tokenizers. We follow their
posture. Reproducibility across machines, if/when it becomes a requirement,
should be solved with a separate `requirements.lock` / `uv.lock` file rather
than by tightening these specifiers.

Bump procedure when moving to a new transformers release:

1. Cut a new tt_symbiote branch (e.g. `transformers-v5.10.0`).
2. Open https://github.com/huggingface/transformers/blob/v5.10.0/setup.py
3. Copy every entry of `_deps` that backs `install_requires` and
   `extras["torch"]` into the dependency block above, verbatim.
4. Re-run the migration codemod against the new transformers source if the
   model APIs changed.

## Tooling left in `scripts/` (committed for reproducibility)

| Script | Purpose |
|---|---|
| `scripts/codemod_imports.py` | One-shot import rewriter. Drops dispatcher imports, rewrites `models.experimental.tt_symbiote.*` → `tt_symbiote.*`, rewrites `models.tt_transformers.tt.ccl` → `tt_symbiote.core.ccl`, rewrites `models.tt_cnn.tt` → `tt_symbiote.integrations.tt_cnn`, and remaps the intra-package `modules/` → `integrations/` + `models/<model>/` moves. Handles multi-line `from X import (\n  a,\n  b,\n)` blocks correctly. Idempotent (model-name rules use negative lookaheads). |
| `scripts/merge_model_files.py` | Concatenates the multiple per-model source files into a single `modeling_<model>.py` per model. Deduplicates top-level imports, drops self-imports that would become circular after the merge, and inserts `# === content from ... ===` section markers so the origin of each block is traceable. |
| `scripts/_smoke_conftest.py` | Local-only stubs for `ttnn` and `tracy`. Used to verify the Phase 2 import graph on a host where the C extensions are not yet installed. Activated by passing `-p scripts._smoke_conftest` to `pytest`. |

These scripts can be deleted any time after Phase 5 makes the package
self-validating; they are kept now so that the migration is reproducible from
scratch if a problem is discovered.

## `TT_CCL` vendoring (planned)

`models/tt_transformers/tt/ccl.py` and three helpers
(`determine_device_name`, `is_blackhole`, `is_wormhole_b0`) were vendored into
`src/tt_symbiote/core/{ccl.py, arch.py}`. This was anticipated in the plan
(§2.2) because `tt_transformers` is not a pip-installable package.

## `tt_cnn` vendoring (extra to plan)

Discovered during the codemod that
`src/tt_symbiote/integrations/ttnn_conv.py` (originally `modules/conv.py`) and
`tests/models/{vit,resnet}/...` referenced
`models.tt_cnn.tt.builder.{Conv2dConfiguration, MaxPool2dConfiguration, TtConv2d, TtMaxPool2d}`.
`tt_cnn` is another tt-metal-only source tree (3 files, ~1858 LOC,
self-contained on `ttnn`/`torch`). Same situation as `TT_CCL`, so it was
vendored identically into `src/tt_symbiote/integrations/tt_cnn/` and the
codemod was extended with the rule
`models.tt_cnn.tt` → `tt_symbiote.integrations.tt_cnn`.

This is **OQ-5** in spirit: revisit when (or if) tt-metal ships a pip-installable
distribution; at that point we can stop vendoring.

## Two dispatcher modules carrying NameErrors (resolved in Phase 3)

`src/tt_symbiote/core/dispatchers/dispatcher_config.py` and
`tensor_operations_dispatcher.py` could not import after the Phase 2 codemod's
"drop dispatcher imports" rule. **Phase 3 deleted the entire dispatcher
subtree** (`core/dispatcher.py`, `core/torch_dispatcher.py`,
`core/dispatchers/`) per `PROJECT_PROPOSAL.md` §6, so this gap no longer
exists. See the Phase 3+4 section below for the full removal scope.

## Empty model directory carried no content

`_staging/models/qwen_omni/` existed in v2 but contained only `__pycache__`.
No content was migrated. The qwen_omni *test* (`test_qwen_omni.py`) is preserved
under `tests/models/qwen_omni/test_modeling_qwen_omni.py`; the actual modeling
code does not exist yet and is part of Phase 7's work.

## File / move counts (Phase 2)

| Category | Count |
|---|---|
| `_staging/` files inspected | 124 (66 `*.py`) |
| Import lines rewritten by the codemod | 382 |
| Dispatcher import lines dropped | 43 |
| Files copied to `src/tt_symbiote/integrations/` | 10 (generic) + 3 (`tt_cnn/`) = 13 |
| Files copied to `src/tt_symbiote/core/` | 6 + dispatchers/{__init__, 5} = 12 + 2 vendored (ccl, arch) |
| Files copied to `src/tt_symbiote/utils/` | 4 |
| Merged `modeling_<model>.py` files | 3 (bailing_moe_v2, gemma4, qwen3_moe; total 8 input files → 3 output files) |
| Demo / example files moved | 2 (`HF_chat.py`, `chat README.md`) |
| Asset moves | 1 (`ARCHITECTURE.svg`) |
| Tests moved into `tests/capabilities/` | 5 |
| Tests moved into `tests/models/<model>/` | 21 (across 18 model subdirs) |
| Tests dropped (per plan §2.6) | 3 (yunet, dots.ocr, training) |
| Cross-repo imports surviving on purpose | 2 (whisper test, listed in `SKIP_DURING_BOOTSTRAP.md`) |

## Phase 3 — dispatcher removal

Per `PROJECT_PROPOSAL.md` §6 the Phase 3 commit removes the entire dispatcher
subsystem and replaces per-op routing with a module-level try/except fallback
inside `TTNNModule.call`.

Deleted in full:

- `src/tt_symbiote/core/dispatcher.py`
- `src/tt_symbiote/core/torch_dispatcher.py`
- `src/tt_symbiote/core/dispatchers/` (entire directory: `__init__.py`,
  `dispatcher_config.py`, `default_dispatcher.py` (~2,320 lines of ATen
  handlers), `debug_dispatcher.py`, `cpu_dispatcher.py`,
  `tensor_operations_dispatcher.py`, `README.md`)
- `TT_SYMBIOTE_DISPATCHER` env var and the `register_dispatcher` /
  `set_dispatcher` / `get_active_dispatcher` / `list_available_dispatchers`
  registry API.

Refactored:

- `src/tt_symbiote/core/run_config.py`: every `torch_dispatch` static method
  on `NormalRun`/`LightweightRun`/`NormalRunWithFallback`/`SELRun`/`DPLRun`/
  `DPLRunNoErrorProp`/`CPU` was deleted, along with
  `DispatchManager.dispatch_to_ttnn_wrapper` and
  `DispatchManager.dispatch_to_torch_wrapper`. The `module_run` method on
  each class was rewritten to operate at **module** granularity: it asserts
  `self._device is not None` with a `set_device`-mentioning error,
  preprocesses / moves weights, then runs `forward` inside a `try/except`
  that falls back to `_fallback_torch_layer` on failure. `DPLRun`/`SELRun`
  now compare torch-vs-TTNN at the **module** boundary instead of per op.
  `TracedRun.module_run` keeps its complex three-phase lifecycle but with
  the improved device-not-set assertion.
- `src/tt_symbiote/core/tensor.py`: `TorchTTNNTensor.__torch_dispatch__` no
  longer routes ops to TTNN — it unwraps to `elem` and runs the op via plain
  torch, emitting a one-shot per-op warning so model authors see when they
  leave the TTNN execution path.
- `src/tt_symbiote/utils/groot_utils.py`: `DPLRunExtended.torch_dispatch`,
  `_dispatch_to_torch_wrapper_gr00t`, and the
  `DispatchManager.dispatch_to_torch_wrapper` monkey-patch were deleted.
  The im2col / aten::view shape fixups previously performed there must be
  re-implemented inside each affected TTNNModule.forward when GR00T is
  re-ported in Phase 7. `set_device_gr00t` and the module-replacement /
  device-management patches are retained.

Run modes retained: `NORMAL` (default), `LIGHTWEIGHT`,
`NORMAL_WITH_FALLBACK`, `SEL`, `DPL`, `DPL_NO_ERROR_PROP`, `CPU`, `TRACED`.
`TT_SYMBIOTE_RUN_MODE` selects between them. The semantics changed: each run
mode operates at module granularity, not op granularity.

Blast radius: any existing TTNN `forward` that relied on transparent
`__torch_dispatch__` routing (`x.reshape(...)`, `x.permute(...)`, etc. on a
`TorchTTNNTensor` carrying a TTNN buffer) now degrades to torch fallback at
the module level. Affected modules are re-ported during Phases 5+. The
fallback emits a `warnings.warn`, so this is visible at runtime.

## Phase 4 — public API

Per `PROJECT_PROPOSAL.md` §4 the Phase 4 commit lands the user-facing HF API.

### Rename: `register_module_replacement_dict` → `register_modules`

- `src/tt_symbiote/utils/module_replacement.py` now exports `register_modules`
  as the canonical name. `register_module_replacement_dict` is retained as a
  thin wrapper that emits `DeprecationWarning` and delegates.
- Existing call sites in `tests/` and `examples/` were **not** rewritten;
  they keep working through the alias and will be migrated incrementally
  during model-port phases.

### `@run_on_devices` decorator: introspectable allow list

The decorator in `src/tt_symbiote/core/module.py` now stamps
`wrapper.__tt_allowed_archs__ = frozenset(allowed_archs)` on the wrapped
function in addition to the existing call-time check. `set_device` reads
this attribute to make the proactive fallback-swap decision.

### Model recipe registry

- `src/tt_symbiote/auto/auto_mappings.py` defines a `Recipe` protocol,
  a `TT_MODEL_REGISTRY: dict[str, Recipe]` keyed by HF model class name,
  and the `@register_recipe(hf_class_name="…")` class decorator that
  instantiates the recipe and inserts it into the registry.
- The registry has **no** runtime dependency on `transformers`; recipes
  can be registered before any HF import.

### Auto* factory + 43 classes

- `src/tt_symbiote/auto/auto_factory.py` defines `_BaseAutoModelClass` and
  `_BaseAutoBackboneClass`. `from_pretrained` delegates to the corresponding
  HF Auto class, looks the loaded model up in `TT_MODEL_REGISTRY`, runs
  `register_modules` with the recipe-provided dict, calls `post_register`,
  and sets `model._tt_symbiote_has_recipe = True` so `set_device` knows to
  enforce the contract on this model.
- `src/tt_symbiote/auto/modeling_auto.py` hand-lists every `AutoModel*`
  from `transformers.models.auto.modeling_auto` v5.9.0 (43 classes). Each
  is a 3-line subclass setting `_HF_AUTO_CLASS = transformers.AutoModelForX`.
- Processor / config Autos (`AutoConfig`, `AutoTokenizer`,
  `AutoImageProcessor`, `AutoFeatureExtractor`, `AutoProcessor`,
  `AutoVideoProcessor`) are thin re-exports from `transformers`.
- `src/tt_symbiote/auto/__init__.py` and the top-level
  `src/tt_symbiote/__init__.py` re-export the public surface (the 43
  `Auto*` classes plus `set_device`, `register_modules`, `register_recipe`,
  `TT_MODEL_REGISTRY`, `DispatchManager`, `TracedRun`).

### `set_device` contract

`src/tt_symbiote/utils/device_management.py` was rewritten to implement
`PROJECT_PROPOSAL.md` §4.4:

1. Walks the model graph (existing recursive walker).
2. For every `TTNNModule`, reads `forward.__tt_allowed_archs__`. Resolves
   the active arch from `MESH_DEVICE`. If the active arch is **not** in the
   allowed set, the module is **swapped in place** in its parent container
   (`nn.Module._modules`, `__dict__`, dict / list / tuple slot) with its
   `_fallback_torch_layer`, and `warnings.warn` records the swap.
3. For each remaining `TTNNModule`, calls `to_device(device)` and (for
   multi-device meshes) `set_device_state(...)`.
4. After the walk, calls `preprocess_weights()` followed by
   `move_weights_to_device()` on every visited TTNN module. This subsumes
   the per-test boilerplate loop that previously followed every
   `set_device(model, device)` call (resolves OQ-3 in favor of subsume).
5. Sets `model._tt_symbiote_device_set = True` on the root and on every
   visited TTNN module.

The hard-error contract is now enforced two ways: at module-execution time
(`module_run` asserts `_device is not None` with a `set_device`-mentioning
message) and proactively for `@run_on_devices`-decorated forwards.

### New tests

`tests/auto/` holds:

- `test_imports.py` — every `Auto*` class plus `set_device`,
  `register_modules`, `register_recipe`, `TT_MODEL_REGISTRY` import from the
  top-level `tt_symbiote` namespace.
- `test_registry.py` — `@register_recipe` round-trip; re-registration
  emits a warning.
- `test_module_replacement.py` — `register_modules` swaps a tiny
  `nn.Linear` correctly; `register_module_replacement_dict` triggers
  `DeprecationWarning`.
- `test_set_device.py` — `set_device` sets `_device`, runs preprocess /
  move-weights, swaps `@run_on_devices`-decorated modules to fallback on
  unsupported arch with a warning, and pre-`set_device` forward raises a
  message that mentions `set_device`.

These run under `scripts/_smoke_conftest.py` with stubbed `ttnn` and
`tracy`; they do not require real hardware.

---

## Phase 5 — Ling-mini-2.0 reference port

Phase 5 ports `inclusionAI/Ling-mini-2.0` (HF `BailingMoeV2ForCausalLM`)
through the new public API and locks in the **single-dict, single-pass
`Recipe.build_module_dict`** contract (Option 1) for every model that
follows. It also resolves PROJECT_PROPOSAL.md open question Q9 on KV-cache
ownership.

### OQ-2 — single-dict module-replacement (Option 1)

Two designs were considered for `Recipe.build_module_dict`:

| Option | Shape | Pros | Cons |
| --- | --- | --- | --- |
| **1 — single dict** | `{torch_class: ttnn_class}` applied in one pass by `register_modules` | Each TTNN wrapper class owns its own subtree conversion (`from_torch` builds the children). One pass, one diagnostic surface. Mirrors the way HF model classes are written — each module is responsible for its own children. | Wrappers that previously assumed their children would be swapped by a follow-up pass must now do that swap themselves in `from_torch`. |
| 2 — list of dicts | `[dict, dict, …]` applied left-to-right | Lets you "stage" replacements (decoder shells first, then linears, then outer wrapper). Matches the pre-Phase-5 test pattern of three sequential `register_module_replacement_dict` calls. | The replacement *order* becomes part of the public contract — fragile. Every model has to think about which pass it belongs to. The wrapper still has to read children to mutate them, so the encapsulation argument is weak. |

We picked **Option 1**. Multi-pass prototyping is still possible inside
`Recipe.post_register` (call `register_modules` directly there for the
unusual case), but the public, tested contract is single-dict.

The Ling port demonstrates the collapse:

- Before Phase 5, the test ran `register_module_replacement_dict` three
  times — first to swap the decoder layer / norm / embedding / rotary,
  then to swap any leftover `nn.Linear` / `nn.SiLU` (effectively only
  `lm_head`), then to swap the outer `BailingMoeV2Model` wrapper.
- After Phase 5, `BailingMoEV2Recipe.build_module_dict` returns one flat
  dict with two entries: `{BailingMoeV2Model: TTNNBailingMoeV2Model, nn.Linear: TTNNLinearIColShardedWRowSharded}`.
- `TTNNBailingMoeV2Model.from_torch` itself calls `register_modules` on
  its own subtree to swap the decoder layers, final norm, `nn.Embedding`,
  and rotary embedding. The wrapper owns the conversion of everything
  *inside* the HF `BailingMoeV2Model`; the recipe only describes the
  *outer* swaps.

See [`src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py`](../src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py)
for the reference layout.

### OQ-9 — KV-cache provisioning

Resolved by extending the `Recipe` protocol with an **optional**
`make_kv_cache(model, device, **kwargs)` hook (documented in the docstring
of `auto/auto_mappings.py`; intentionally not part of the `Protocol` body
so the runtime `isinstance(_, Recipe)` check stays permissive for
non-cached recipes).

The closest reference architecture in `tt-metal` is `tt_transformers/tt/model.py::Transformer.__init__`,
which takes `paged_attention_config=` directly so the model owns its KV
cache. That pattern doesn't translate verbatim because tt_symbiote loads
the model through HuggingFace (CPU) and only knows the device later in
`set_device`. We split the constructor in two:

1. The recipe declares **how** to make the cache (`make_kv_cache`).
2. `set_device` actually calls it (after device binding and weight
   preprocessing) and attaches the result as `model._tt_kv_cache`.

The user surface stays a single line of HF-style code:

```python
set_device(model, mesh_device)
out = model.generate(..., past_key_values=model._tt_kv_cache)
```

`make_kv_cache` failures are warnings, not hard errors — non-cached
recipes (most non-LM-style models) just leave `model._tt_kv_cache` unset.

### Top-level registration (HF-style side effect)

`src/tt_symbiote/__init__.py` now imports `tt_symbiote.models`, which in
turn imports every recipe-bearing model subpackage (currently just
`bailing_moe_v2`). The `@register_recipe` decorator inside each
`modeling_<model>.py` runs and populates `TT_MODEL_REGISTRY` at top-level
import time — so by the time a user calls `AutoModelForCausalLM.from_pretrained`
the registry is already populated. This mirrors how
`transformers/models/__init__.py` registers Auto-class mappings.

Each per-model import is wrapped in `try/except` so a broken model file
warns loudly but does not poison the whole `tt_symbiote` import.

### Structural fix: `next_power_of_2` moved out of the bailing modeling file

The Phase 2 mechanical merge dropped `_next_power_of_2` inside
`models/bailing_moe_v2/modeling_bailing_moe_v2.py`, but the generic
`integrations/ttnn_embedding.py` was reaching back into that model file
to import it — a circular import that only surfaced once the top-level
side-effect import chain in `src/tt_symbiote/__init__.py` started
exercising the bailing package eagerly. Phase 5 moves the helper to
[`src/tt_symbiote/utils/math_utils.py`](../src/tt_symbiote/utils/math_utils.py)
(public name `next_power_of_2`) and keeps `_next_power_of_2` as a
back-compat alias in the modeling file.

### Hardware acceptance — green on T3K

End-to-end smoke test passes on a Tenstorrent T3K (1×8 mesh). Loading
`inclusionAI/Ling-mini-2.0` through `tt_symbiote.AutoModelForCausalLM.from_pretrained`,
binding via `set_device`, and running `model.generate(...)` returns
coherent text (`"As an AI, I don't have personal preferences or taste
buds, …"`). The single-pass `register_modules` call dispatched from
`BailingMoEV2Recipe.build_module_dict` bound every layer (0..19),
the final `TTNNDistributedRMSNorm`, and the `lm_head`
(`TTNNLinearIColShardedWRowSharded`, picked up by the recipe's
`nn.Linear` entry) onto the mesh without manual intervention.

Two pre-existing blockers had to be resolved before the run was green
and are described in the next section.

### `_hf_compat`: shim layer for `transformers` API drift

Hub modeling files loaded via `trust_remote_code=True` are pinned to the
`transformers` release the *model author* used at upload time. When
`tt_symbiote` pins a newer release (5.9.0, per `PROJECT_PROPOSAL.md` §10),
those Hub files can import symbols that have since been removed or moved
upstream. [`src/tt_symbiote/_hf_compat.py`](../src/tt_symbiote/_hf_compat.py)
holds a small, idempotent shim catalog that
[`_BaseAutoModelClass.from_pretrained`](../src/tt_symbiote/auto/auto_factory.py)
installs once before invoking the HF auto factory, restoring the legacy
API surface those files were written against. Today's catalog:

1. **`transformers.utils.import_utils.is_torch_fx_available`** —
   removed between 4.x and 5.x. The Ling-mini-2.0 Hub file imports it as
   a feature gate before calling `torch.fx.wrap`. The shim re-installs
   it as `lambda: hasattr(torch, "fx")` (always `True` on modern PyTorch).

2. **`transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"]`** —
   the legacy unscaled-RoPE entry was dropped from the dict between 4.x
   and 5.x. The Hub file falls back to `self.rope_type = "default"`
   whenever `config.rope_scaling is None` and then looks the key up,
   raising `KeyError: 'default'`. The shim re-injects the canonical
   formula
   `inv_freq = 1 / base ** (arange(0, dim, 2) / dim)`,
   reading `rope_theta` / `head_dim` / `partial_rotary_factor` via
   `getattr` so it works with the *legacy* config shape (avoiding
   `config.standardize_rope_params()` which the legacy configs do not
   satisfy).

Both shims are guarded by `if not hasattr(...) / if key not in dict` so
they're no-ops on transformers releases that still ship the originals.
The whole installer is gated by a module-level `_INSTALLED` flag for
idempotency.

This pattern is the recommended way to extend `tt_symbiote` to new
remote-code models that ride on older `transformers` releases: add a new
guarded entry to `install_transformers_shims`.

### New tests

`tests/auto/test_ling_recipe.py` exercises the recipe shape without
hardware:

- `test_recipe_registered`: importing `tt_symbiote.models.bailing_moe_v2`
  populates `TT_MODEL_REGISTRY["BailingMoeV2ForCausalLM"]`.
- `test_build_module_dict_shape` / `test_build_module_dict_covers_outer_model_and_lm_head`:
  return value is a single flat dict (Option 1), keys / values are
  classes, the outer `BailingMoeV2Model` and `nn.Linear` are both
  present.
- `test_post_register_patches_device`: `model.device` is rewritten to a
  property returning CPU after `post_register` runs.
- `test_make_kv_cache_signature` / `test_make_kv_cache_reads_config`: the
  hook accepts `(model, device, **kwargs)` and reads the right HF config
  fields when constructing the paged cache.

These run under the same stubbed-`ttnn` conftest as the Phase 4 tests.
