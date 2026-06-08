# Experimental / partial-TTNN tests

This is the live half of the two-tree per-model test taxonomy:

- **`tests/experimental/<name>/`** (this tree) — partial-TTNN bring-up
  references. All models currently live here.
- **`tests/models/<name>/`** — an empty rich-tree placeholder, the promotion
  target for models whose end-to-end traced correctness is proven.
- **`tests/shared/`** — shared helpers (`pcc_utils.py`, `shared_configs.py`,
  `conftest.py`) plus the shared capability tests.

The former per-model capability tree has been **removed**; its helpers and
shared capability tests now live in `tests/shared/`.

## Minimal floor

Every dir under `tests/experimental/<name>/` must contain at least an
`__init__.py` and a schema-valid `test_config.json` with the 5 keys:
`tt_metal_commit` (str; `""` until validated), `device_arch` (str; default
`"T3K"`), `pcc_threshold` (number; default `0.99`), `hf_model_id` (str;
`"<unknown>"` if unresolved), `hf_revision` (str; default `"main"`). Some models
exceed this floor with full Tier1–4 + sweep/profiling artifacts (e.g. dots_ocr).
The machine check for this floor (and the RICH floor) is
[`tests/auto/test_tier_structure.py`](../auto/test_tier_structure.py) (a thin
wrapper over `scripts/check_tier_structure.py`).

## Naming rule

Each per-model dir name MUST equal a `src/tt_symbiote/models/<name>`
(transformers-canonical) token (canonical `owlvit`, `speecht5`,
`qwen3_omni_moe`). Banned bare variant suffixes as dir names: `_4_7`, `_5`,
`_flash`, `_coder_next` — variant test files use the `_variant_<qualifier>` form
and are not floor-required. The banned-name set lives in
[`tests/auto/test_tier_structure.py`](../auto/test_tier_structure.py).

## Status

These tests are **excluded from default pytest collection** (`testpaths =
["tests/auto", "tests/shared"]`; `tests/experimental/` excluded via `addopts =
"... --ignore=tests/experimental"` in [`pyproject.toml`](../../pyproject.toml)),
**excluded from the published sdist / wheel**, and **not part of any CI gate**.
Many target models with cross-repo imports, optional deps, or shapes that no
longer exist — treat this folder as a parking lot of porting references.

## Promotion: experimental → models

To promote, BOTH must hold: (1) **proven e2e traced correctness** — all
RICH-tier PCC green in TRACED mode, traced-mode PCC matches NORMAL, semantic
validation passes; (2) **upgrade to the RICH floor** — `shapes.json`,
`op_map.json`, the Tier1–4 test files, and a populated `test_config.json` with a
pinned `tt_metal_commit`. Use `git mv tests/experimental/<name>/
tests/models/<name>/` to preserve history, then register `<name>` in
`_RECIPE_BEARING_SUBPACKAGES` where applicable.

## Current dirs

Run `ls tests/experimental/` for the current set. Load-bearing path notes:
`glm4_moe` → `tt_symbiote.modules.ttnn_moe.Glm4MoeConfig`; `vit` →
`TTNNViTEmbeddings` from `modules/ttnn_conv.py`; `whisper` has surviving
cross-repo imports from `models.demos.whisper.tt.*`.
