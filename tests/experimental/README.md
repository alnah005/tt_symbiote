# Experimental / quarantined tests

This directory holds capability-level tests that survive from the Phase 2
tt-metal migration but whose corresponding `src/tt_symbiote/models/<name>/`
package was **not** ported to the recipe API. Concretely, each subdirectory
here tests TTNN module replacement against an HF model that is **not**
registered in
[`src/tt_symbiote/models/__init__.py::_RECIPE_BEARING_SUBPACKAGES`](../../src/tt_symbiote/models/__init__.py).

## Status

These tests are:

- **Excluded from default pytest collection** (see `addopts` in
  [`pyproject.toml`](../../pyproject.toml) `[tool.pytest.ini_options]`).
- **Excluded from the published sdist / wheel** — tests are not packaged
  in `tt_symbiote` regardless, but this directory in particular is not
  guaranteed to import cleanly even with the runtime installed.
- **Not part of any CI gate.** The lint workflow does not touch them; the
  release workflow does not touch them.
- **Not guaranteed to run.** Many target models with cross-repo imports
  from `tt-metal/models/demos/`, optional deps (`decord`, `av`,
  `torchvision`-extras), or hardware shapes that no longer exist.

Treat this folder as a parking lot of porting references, not as a test
suite.

## Resurrecting a port

To bring one of these models back into the supported set:

1. Create `src/tt_symbiote/models/<name>/` with the standard layout
   (`__init__.py`, `configuration_<name>.py`, `modeling_<name>.py`)
   following the pattern in `src/tt_symbiote/models/bailing_moe_v2/` or
   `src/tt_symbiote/models/resnet/`.
2. Add `<name>` to `_RECIPE_BEARING_SUBPACKAGES` in
   [`src/tt_symbiote/models/__init__.py`](../../src/tt_symbiote/models/__init__.py)
   so the `@register_recipe` decorator fires on package import.
3. `git mv tests/experimental/<name>/ tests/models/<name>/` to bring the
   capability tests back into default collection.
4. Add a recipe-level test under `tests/auto/test_<name>_recipe.py`
   mirroring `tests/auto/test_resnet_recipe.py`.
5. Add an end-to-end example under `examples/e2e/<name>/`.
6. Update `docs/supported_models.md`.

## Inventory (at the time of quarantine)

| Subdir | Notes |
|---|---|
| `glm4_moe/` | 4 variant files; references `tt_symbiote.integrations.ttnn_moe.Glm4MoeConfig` which IS in the runtime |
| `gpt_oss/` | GPT-OSS MoE |
| `gr00t/` | NVIDIA GR00T |
| `hunyuan_video/` | Hunyuan video diffusion |
| `llama/` | Llama text-only |
| `molmo2/` | AllenAI Molmo |
| `olmo3/` | AllenAI OLMo |
| `openvla/` | OpenVLA |
| `owl_vit/` | OwlViT zero-shot detection |
| `qwen_omni/` | Qwen Omni |
| `speech_t5/` | SpeechT5 |
| `vit/` | ViT — references `TTNNViTEmbeddings` from `integrations/ttnn_conv.py` which IS in the runtime |
| `whisper/` | Has surviving cross-repo imports from `models.demos.whisper.tt.*` |
