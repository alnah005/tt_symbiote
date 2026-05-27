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

## Two dispatcher modules carrying NameErrors

`src/tt_symbiote/core/dispatchers/dispatcher_config.py` and
`tensor_operations_dispatcher.py` cannot import after the codemod's
"drop dispatcher imports" rule. This is **expected** — the codemod drops the
defining imports (`default_dispatcher`, handle_*) so the rest of the tree
compiles, but those two files now reference names that no longer exist.
The entire dispatcher subsystem is removed in Phase 3, at which point this
naturally goes away. Tracked in
[`tests/SKIP_DURING_BOOTSTRAP.md`](../tests/SKIP_DURING_BOOTSTRAP.md).

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
