# Experimental / partial-TTNN tests

This directory is one half of the two-tree per-model test taxonomy:

- **`tests/experimental/<name>/`** (this tree) — **MINIMAL** floor. Partial
  TTNN bring-up: the model is either not registered in
  [`src/tt_symbiote/models/__init__.py::_RECIPE_BEARING_SUBPACKAGES`](../../src/tt_symbiote/models/__init__.py)
  or its end-to-end traced correctness has not yet been proven. These are
  bring-up references, not a validated suite.
- **`tests/models/<name>/`** (sibling tree) — **RICH** floor. Per-model dirs
  whose end-to-end traced correctness is proven (all rich-tier PCC green in
  TRACED mode + traced-mode PCC matches NORMAL + semantic validation).
- **`tests/shared/`** — shared helpers (`pcc_utils.py`, `shared_configs.py`,
  `conftest.py`) plus the shared capability tests (`test_attention.py`,
  `test_conv.py`, `test_moe.py`, `test_rope.py`, `test_dpl.py`).

The former per-model capability tree has been **removed**; its contents moved
to `tests/shared/` (helpers + shared tests) and `tests/experimental/dots_ocr/`.

## MINIMAL floor

Every dir under `tests/experimental/<name>/` must contain at least:

- `__init__.py`
- a schema-valid `test_config.json` with the 5 keys:
  `tt_metal_commit` (str; `""` until validated against a pinned tt-metal
  checkout), `device_arch` (str; default `"T3K"`), `pcc_threshold` (number;
  default `0.99`), `hf_model_id` (str; `"<unknown>"` if unresolved),
  `hf_revision` (str; default `"main"`).

The machine check for this floor (and the RICH floor) is
[`tests/auto/test_tier_structure.py`](../auto/test_tier_structure.py) (a thin wrapper over `scripts/check_tier_structure.py`).

## Naming rule

Each per-model dir name MUST equal a `src/tt_symbiote/models/<name>`
(transformers-canonical) token. Non-canonical aliases are forbidden as dir
names — use the canonical `owlvit`, `speecht5`, and `qwen3_omni_moe` forms
(and the deprecated NVIDIA humanoid-robotics model dir is forbidden entirely).
Banned bare variant suffixes as dir names: `_4_7`, `_5`, `_flash`,
`_coder_next`. Variant test files use the `_variant_<qualifier>` form (e.g.
`test_modeling_glm4_moe_variant_4_7.py`) and are not floor-required. The
machine-enforced banned-name set lives in
[`tests/auto/test_tier_structure.py`](../auto/test_tier_structure.py) (a thin wrapper over `scripts/check_tier_structure.py`).

## Status

These tests are:

- **Excluded from default pytest collection.** The default
  `testpaths` is now `["tests/auto", "tests/shared"]`;
  `tests/experimental/` stays excluded via `addopts = "... --ignore=tests/experimental"`
  in [`pyproject.toml`](../../pyproject.toml) `[tool.pytest.ini_options]`.
- **Excluded from the published sdist / wheel** — not guaranteed to import
  cleanly even with the runtime installed.
- **Not part of any CI gate.** The lint workflow does not touch them; the
  release workflow does not touch them.
- **Not guaranteed to run.** Many target models with cross-repo imports from
  `tt-metal/models/demos/`, optional deps (`decord`, `av`,
  `torchvision`-extras), or hardware shapes that no longer exist.

Treat this folder as a parking lot of porting references, not as a test suite.

## Promotion: experimental → models

To promote a model from `tests/experimental/<name>/` to `tests/models/<name>/`,
BOTH of the following must hold:

1. **Proven e2e traced correctness**: all RICH-tier PCC green in TRACED mode
   (0.99 default / 0.999 bring-up threshold) AND traced-mode PCC matches NORMAL
   mode AND semantic validation passes.
2. **Upgrade to the RICH floor**: add `shapes.json`, `op_map.json`,
   `test_ops_<name>.py`, `test_composites_<name>.py`, `test_decoder_<name>.py`,
   `test_modeling_<name>.py`, `test_traced_<name>.py`, and a populated
   `test_config.json` with a pinned `tt_metal_commit`.

Use `git mv tests/experimental/<name>/ tests/models/<name>/` to preserve history.
Then add a recipe-level test under `tests/auto/test_<name>_recipe.py` and
register `<name>` in `_RECIPE_BEARING_SUBPACKAGES` where applicable.

## Inventory

| Subdir | Notes |
|---|---|
| `glm4_moe/` | base + `_variant_4_7` / `_variant_5` / `_variant_flash`; references `tt_symbiote.integrations.ttnn_moe.Glm4MoeConfig` which IS in the runtime |
| `gpt_oss/` | GPT-OSS MoE |
| `hunyuan_video/` | Hunyuan video diffusion |
| `llama/` | Llama text-only |
| `molmo2/` | AllenAI Molmo |
| `olmo3/` | AllenAI OLMo |
| `openvla/` | OpenVLA |
| `owlvit/` | OwlViT zero-shot detection |
| `qwen3_moe/` | base + `_variant_coder_next` |
| `qwen3_omni_moe/` | Qwen3 Omni MoE |
| `speecht5/` | SpeechT5 |
| `vit/` | ViT — references `TTNNViTEmbeddings` from `integrations/ttnn_conv.py` which IS in the runtime |
| `whisper/` | Has surviving cross-repo imports from `models.demos.whisper.tt.*` |
