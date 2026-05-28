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

### Standalone install path verified

Hardware acceptance was originally completed inside the tt-metal
Python env (with `pip install -e .` on top of tt-metal's locally-built
`ttnn`). It has since been re-verified end-to-end on T3K in a **fully
standalone** venv with zero tt-metal involvement: a fresh `python -m
venv`, `pip install ttnn==0.68.0` (whose embedded `sfpi-version` is
`7.35.3` — the version pre-installed system-wide at
`/opt/tenstorrent/sfpi/`), plus `pip install -e tt_symbiote`. Ling
loads, binds across the 1×8 mesh, and generates the same coherent
text.

That recipe is now codified by
[`scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh), which
reads the `(ttnn, sfpi)` pin from
[`scripts/ttnn-pin.txt`](../scripts/ttnn-pin.txt), probes the system
sfpi for a match, and bails with a remediation hint if not. PROJECT_PROPOSAL.md
OQ-1 ("revisit when tt-metal ships PyPI wheels") is partly closed by
this: ttnn wheels are on PyPI, the bootstrap consumes them, and the
only remaining system-level prerequisite is the matching sfpi
RISC-V toolchain (Tenstorrent's apt-installable package, the same
role CUDA plays for GPU users).

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

---

## Phase 6 — ResNet vision reference port

Phase 6 was originally scoped to GLM + Gemma4 LLM ports; we redirected
it at user request to a **vision** reference port instead, because the
Phase 5 single-dict recipe contract had only been exercised against an
LLM and we wanted a second axis (NHWC convs, no KV cache, image input)
before committing to it. The port covers all five canonical Microsoft
ResNet variants (`microsoft/resnet-{18,34,50,101,152}`) and is hardware-
verified end-to-end on `resnet-50`.

### Pivot rationale: vision first, GLM/Gemma4 deferred

Two pieces of confidence we wanted to earn before doing another LLM:

1. **Recipe shape works for non-LM tasks.** Vision models have no
   `make_kv_cache`, no `lm_head`, no causal mask. The
   `@register_recipe` no-op installer for `make_kv_cache` (added in
   Phase 5) is the part of the API contract that handles "this
   architecture doesn't need a KV cache" — Phase 6 is its first real
   user.
2. **NHWC convs through `set_device`.** The Phase 5 reference exercised
   tile-layout linear / attention. Convs use the NHWC integration
   (`integrations/ttnn_conv.py`) and have their own quirks (sliding-
   window L1 small-region scratch, fused BN). We wanted to confirm
   that the same `from_pretrained` → `set_device` → `forward(...)`
   surface composes with that integration stack.

GLM and Gemma4 land in a future phase. They're both LLM-shaped so they'll
reuse Phase 5 plumbing directly.

### `configuration_<model>.py` precedent

HuggingFace splits each model into `configuration_<model>.py` (the
`PretrainedConfig` subclass + per-model defaults) and
`modeling_<model>.py` (the `PreTrainedModel` subclass). We started
Phase 5 with everything in one file because Ling's TTNN-side knobs are
all attention / KV-cache shapes that the recipe pulls from
`model.config` directly. Phase 6 introduces the first model where the
TTNN side needs *its own* per-variant knobs that are **not** HF
hyperparameters — the `l1_small_size` value for the heavy 7x7 stem conv
is determined by the hardware footprint of the conv, not by the model.

Rather than smuggle those knobs into `ResNetConfig` (where they don't
belong — HF would refuse them on a round-trip through
`config.to_json_string()`) we kept them in a separate
[`configuration_resnet.py`](../src/tt_symbiote/models/resnet/configuration_resnet.py)
that re-exports the HF `ResNetConfig` *and* holds a
`RESNET_TTNN_TUNING` lookup table keyed by checkpoint id, plus a
shape-based fallback (`(tuple(depths), layer_type) → canonical id`) for
community fine-tunes. The recipe's `post_register` resolves the entry
and stashes it as `model._tt_runtime_config`.

This is the recommended layout for every future model that needs
TTNN-only tuning: a sibling `configuration_<model>.py` next to
`modeling_<model>.py`, both inside `src/tt_symbiote/models/<model>/`.

### `TTNNResNetBottleNeckLayer` lives in `modeling_resnet.py`, not in `integrations/`

There is already a `TTNNBottleneck` in
[`integrations/ttnn_conv.py`](../src/tt_symbiote/integrations/ttnn_conv.py)
that walks torchvision's flat `conv1/bn1/conv2/bn2/conv3/bn3` shape.
HF's `ResNetBottleNeckLayer` is structurally identical (same three
convs, same residual add) but its weight tree is nested:
`layer.layer[0].convolution`, `layer.layer[0].normalization`, ... — the
torchvision-shaped wrapper can't be reused.

Two ways to handle this:

1. **Add a second wrapper next to the model** (what we did). The
   HF-shaped `TTNNResNetBottleNeckLayer` lives in `modeling_resnet.py`
   alongside the recipe. Same precedent as
   `TTNNBailingMoEDecoderLayer` from Phase 5 (block-shape wrappers live
   next to their model).
2. **Generalize the existing `TTNNBottleneck`** to accept both shapes.
   Tempting but pollutes the integration with model-specific tree
   walking. We chose against it; the legacy torchvision-shaped class
   stays in `integrations/` for any future torchvision-shaped caller
   without changes.

Same call applied to `TTNNResNetBasicLayer` (resnet-18/34, 2-conv) and
`TTNNResNetAdaptiveAvgPool2dNHWC` (the NHWC-aware pooler; see next
section): these are model-specific wrappers, kept in
`modeling_resnet.py`.

### NHWC pooling: hardware bring-up surprise

The flow inside the TTNN ResNet is NCHW only at the boundaries — HF
hands us NCHW pixel values, `TTNNResNetEmbeddings` permutes to NHWC,
and the entire encoder runs NHWC. The exit point is HF's
`ResNetModel.pooler` (`nn.AdaptiveAvgPool2d(output_size=(1, 1))`)
followed by `nn.Sequential(nn.Flatten(), nn.Linear(...))`.

The naive plan was to leave the pooler on host. That broke during
bring-up: the pooler is `AdaptiveAvgPool2d`, which expects **NCHW**
input and reduces over axes [2, 3]. Feeding it an NHWC tensor
`(B, 7, 7, 2048)` made it pool the wrong axes, producing
`(B, 7, 1, 1)` instead of `(B, 2048, 1, 1)`. Flatten then gave a
7-element vector, which fed a 2048-input Linear, and the underlying
TTNN matmul rejected the shape with `width=7 height=2048`.

Fix: a small wrapper `TTNNResNetAdaptiveAvgPool2dNHWC` reduces over
the NHWC spatial axes `[1, 2]` with `keepdim=True` and then permutes
back to NCHW so the downstream Flatten + Linear see exactly what they
expect. Added as a seventh entry to the recipe's module-replacement
dict.

Open question implied by this: are there other places where HF expects
NCHW after our NHWC encoder? The answer for ResNet is "no — the pooler
is the only one". Future vision models (ViT, ConvNeXt) may need a
similar bridge; the pattern is documented here and the wrapper can be
ported nearly verbatim.

### `l1_small_size` is a hardware budget knob the user must pass

The 7x7 stride-2 stem conv on a 224x224 input needs the L1 *small*
region for sliding-window halo metadata. The default
`ttnn.open_mesh_device(...)` ships with `l1_small_size=0`, which
produces an OOM at first forward (`Out of Memory: Not enough space to
allocate 1792 B L1_SMALL buffer across 56 banks`).

We considered three remedies:

1. Make `set_device` introspect `model._tt_runtime_config` and re-open
   the mesh device with the right budget. Rejected — `set_device`
   should not own device lifecycle.
2. Document the user-visible setting on the reproducer. Chosen. The
   value (`245760`) is now in `RESNET_TTNN_TUNING`, in
   `examples/e2e/run_resnet50.py`'s `open_mesh_device` call, and in
   this note.
3. Add a `device_params` helper that returns the budget for a given
   recipe. Deferred (it's a thin convenience over (2)).

### Hardware acceptance — green on N150 (and T3K with single chip)

End-to-end smoke pass on a single Tenstorrent chip. Loading
`microsoft/resnet-50` through `tt_symbiote.AutoModelForImageClassification.from_pretrained`,
binding via `set_device(model, mesh_device)` (with
`l1_small_size=245760` on `open_mesh_device`), and running
`model(pixel_values=...)` returns logits whose top-1 prediction is
`'tiger cat'` on the canonical COCO val cat image
(`images.cocodataset.org/val2017/000000039769.jpg`). Top-2 is
`'tabby, tabby cat'`. Zero TTNN-forward-fallback warnings; every conv,
shortcut, residual add, pool, and the classifier head executes on
device.

### Variant status

| Checkpoint | Recipe / `set_device` | Forward through TTNN | Hardware verified |
|---|---|---|---|
| `microsoft/resnet-18` | ✅ | ✅ (path covered) | ⏳ (basic-layer needs hw smoke) |
| `microsoft/resnet-34` | ✅ | ✅ (path covered) | ⏳ (basic-layer needs hw smoke) |
| `microsoft/resnet-50` | ✅ | ✅ | ✅ (N150, T3K 1×1) |
| `microsoft/resnet-101` | ✅ | ✅ (path covered) | ⏳ (depth tuning may need a larger trace region) |
| `microsoft/resnet-152` | ✅ | ✅ (path covered) | ⏳ (depth tuning may need a larger trace region) |

The recipe is variant-agnostic; verifying the other four is a one-line
swap of the model id in `examples/e2e/run_resnet50.py` plus flipping
the `hw_verified` flag in `RESNET_TTNN_TUNING`. Tracked in the open
todo as a follow-up rather than gating Phase 6.

### New tests

`tests/auto/test_resnet_recipe.py` exercises the recipe shape without
hardware (89/89 unit tests green under the same stubbed-`ttnn`
conftest used by Phase 4 / Phase 5):

- `test_recipe_registered` — importing `tt_symbiote.models.resnet`
  populates `TT_MODEL_REGISTRY["ResNetForImageClassification"]`.
- `test_build_module_dict_shape` / `test_build_module_dict_covers_required_swaps`
  — return value is a single flat dict (Option 1), keys / values are
  classes, all seven required HF building blocks are present
  (`ResNetConvLayer`, `ResNetShortCut`, `ResNetBasicLayer`,
  `ResNetBottleNeckLayer`, `ResNetEmbeddings`, `nn.AdaptiveAvgPool2d`,
  `nn.Linear`).
- `test_post_register_patches_device` /
  `test_post_register_attaches_runtime_config` — both expected side
  effects happen on a freshly loaded model.
- `test_make_kv_cache_is_noop` — confirms the `@register_recipe`
  no-op installer covers vision recipes that don't need a cache.
- `test_lookup_ttnn_tuning_fallbacks` — the checkpoint → shape →
  default ladder behaves as documented.

The hardware-bound smoke at
`tests/models/resnet/test_modeling_resnet.py` was rewritten from its
old torchvision shape to the new `AutoModelForImageClassification +
set_device + forward` flow, parametrized over a `mesh_device` fixture
just like the Phase 5 Ling smoke.
