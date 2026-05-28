# Supported models

Catalog of models that ship with `tt_symbiote` recipes. A row in this
table means the model can be loaded through
`tt_symbiote.AutoModel*.from_pretrained`, bound to a Tenstorrent device
via `tt_symbiote.set_device(model, device)`, and called through the
standard HuggingFace forward / generate API.

Hardware verification is split into three states:

- ✅ **verified** — end-to-end smoke pass on the listed hardware target,
  with zero TTNN-forward-fallback warnings, and (for vision)
  semantically correct top-1 on a known image / (for LLM) coherent
  EOS-terminated output.
- ⏳ **structurally supported** — recipe + module replacement land,
  unit tests pass. Hardware forward has not been exercised yet.
- ⚠ **partial** — runs but with documented caveats. See the linked
  walkthrough for what works and what doesn't.

When a row is at ✅ there is a matching script in
[`examples/e2e/`](../examples/e2e/) that reproduces the run on the
listed hardware.

## Causal LM

| Checkpoint | HF class | Status | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|
| `inclusionAI/Ling-mini-2.0` | `BailingMoeV2ForCausalLM` | ✅ verified | T3K (1×8) | [`run_ling_mini_2_0.py`](../examples/e2e/run_ling_mini_2_0.py) | [`ling_mini_2_0_guide.md`](ling_mini_2_0_guide.md) |

## Image classification

| Checkpoint | HF class | Status | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|
| `microsoft/resnet-18` | `ResNetForImageClassification` | ⏳ structurally supported | N150 (1×1) | (see resnet-50) | — |
| `microsoft/resnet-34` | `ResNetForImageClassification` | ⏳ structurally supported | N150 (1×1) | (see resnet-50) | — |
| `microsoft/resnet-50` | `ResNetForImageClassification` | ✅ verified | N150 (1×1), T3K (1×1) | [`run_resnet50.py`](../examples/e2e/run_resnet50.py) | — |
| `microsoft/resnet-101` | `ResNetForImageClassification` | ⏳ structurally supported | N150 (1×1) | (see resnet-50) | — |
| `microsoft/resnet-152` | `ResNetForImageClassification` | ⏳ structurally supported | N150 (1×1) | (see resnet-50) | — |

The four non-50 ResNet variants share the same recipe; flipping each to
✅ is a one-line model-id swap in `run_resnet50.py` plus updating the
`hw_verified` flag in
[`RESNET_TTNN_TUNING`](../src/tt_symbiote/models/resnet/configuration_resnet.py).

## How to add a model

1. Add the recipe under `src/tt_symbiote/models/<name>/` following the
   `bailing_moe_v2` (LLM) or `resnet` (vision) layout: a
   `configuration_<name>.py` (re-export HF config + TTNN tuning table
   if needed), a `modeling_<name>.py` (TTNN wrappers + a class
   decorated with `@register_recipe(hf_class_name=...)`), and an
   `__init__.py` that imports them eagerly so the registry is
   populated at top-level import.
2. List the recipe-bearing subpackage in
   [`src/tt_symbiote/models/__init__.py`](../src/tt_symbiote/models/__init__.py)
   so it gets imported when `tt_symbiote` does.
3. Add a hardware-free unit test under `tests/auto/test_<name>_recipe.py`
   covering the recipe shape and `post_register` side-effects.
4. Add a hardware smoke under `tests/models/<name>/`.
5. Add an end-to-end reproducer under
   [`examples/e2e/`](../examples/e2e/) and a row to its
   [tracking table](../examples/e2e/README.md).
6. Add a row here, in the appropriate task section.

See [`docs/migration_notes.md`](migration_notes.md) for the design
rationale behind each Phase.
