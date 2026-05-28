# Supported models

Catalog of models that ship with `tt_symbiote` recipes. A row in this
table means the model can be loaded through
`tt_symbiote.AutoModel*.from_pretrained`, bound to a Tenstorrent device
via `tt_symbiote.set_device(model, device)`, and called through the
standard HuggingFace forward / generate API.

Hardware verification is split into three states:

- ✅ **verified** — end-to-end smoke pass on the listed hardware target,
  with zero *unexpected* TTNN-forward-fallback warnings, and (for vision)
  semantically correct top-1 on a known image / (for LLM) coherent
  EOS-terminated output / (for VLM) semantically correct answer to the
  documented prompt. CPU-only ports count as verified iff every CPU
  module is explicitly listed in the recipe's `cpu_fallback` (so
  `tt_symbiote.compatibility.report(model)["runtime_observed"]["unexpected"]`
  is empty).
- ⏳ **structurally supported** — recipe + module replacement land,
  unit tests pass. Hardware forward has not been exercised yet.
- ⚠ **partial** — runs but with documented caveats. See the linked
  walkthrough for what works and what doesn't.

When a row is at ✅ there is a matching script in
[`examples/e2e/`](../examples/e2e/) that reproduces the run on the
listed hardware.

## TT-implemented vs. CPU coverage

Every recipe declares three class-name lists that drive
[`tt_symbiote.compatibility.report(model)`](../src/tt_symbiote/utils/compatibility.py):

- **`tt_implemented`** — HF classes for which the recipe ships a TTNN
  wrapper (i.e. it is in `build_module_dict`). These run on
  Tenstorrent silicon.
- **`cpu_fallback`** — HF classes that are *exercised* by the
  documented demos but stay as PyTorch in this recipe. These run on
  the host CPU. Entries here are intentional — they're the queue of
  modules awaiting a TTNN port.
- **`out_of_scope`** — HF classes that exist in the upstream
  modeling file but aren't touched by the documented demos (e.g. the
  Gemma-4 audio tower under the image-text path, or output
  dataclasses that aren't modules at all).

Per-model the "TT impl / CPU fallback / out of scope" counts in the
tables below are read directly from the recipes; the runtime ledger
in `compatibility.report(...)["runtime_observed"]` flags any
fallback that fired during the demo *and* was not declared (a sign
the recipe drifted relative to the HF source).

## Causal LM

| Checkpoint | HF class | Status | TT / CPU / OOS | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|---|
| `inclusionAI/Ling-mini-2.0` | `BailingMoeV2ForCausalLM` | ✅ verified | full TTNN | T3K (1×8) | [`run_ling_mini_2_0.py`](../examples/e2e/run_ling_mini_2_0.py) | [`ling_mini_2_0_guide.md`](ling_mini_2_0_guide.md) |

## Image classification

| Checkpoint | HF class | Status | TT / CPU / OOS | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|---|
| `microsoft/resnet-18` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | (see resnet-50) | — |
| `microsoft/resnet-34` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | (see resnet-50) | — |
| `microsoft/resnet-50` | `ResNetForImageClassification` | ✅ verified | full TTNN | N150 (1×1), T3K (1×1) | [`run_resnet50.py`](../examples/e2e/run_resnet50.py) | — |
| `microsoft/resnet-101` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | (see resnet-50) | — |
| `microsoft/resnet-152` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | (see resnet-50) | — |

The four non-50 ResNet variants share the same recipe; flipping each to
✅ is a one-line model-id swap in `run_resnet50.py` plus updating the
`hw_verified` flag in
[`RESNET_TTNN_TUNING`](../src/tt_symbiote/models/resnet/configuration_resnet.py).

## Image-text-to-text (VLM)

| Checkpoint | HF class | Status | TT / CPU / OOS | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | `Gemma4ForConditionalGeneration` | ✅ verified (CPU-first) | 0 / 21 / 14 | N150 (1×1) | [`run_gemma4_e2b.py`](../examples/e2e/run_gemma4_e2b.py) | — |
| `google/gemma-4-E4B-it` | `Gemma4ForConditionalGeneration` | ⏳ structurally supported | 0 / 21 / 14 | N150 (1×1) | (see E2B) | — |
| `google/gemma-4-31B-it` | `Gemma4ForConditionalGeneration` | ⏳ structurally supported | 0 / 21 / 14 | T3K (1×8) | [`run_gemma4_31b.py`](../examples/e2e/run_gemma4_31b.py) | — |
| `google/gemma-4-26B-A4B-it` | `Gemma4ForConditionalGeneration` | ⏳ structurally supported | 0 / 21 / 14 | T3K (1×8) | (see 31B) | — |

All four Gemma-4 variants share the same recipe
([`Gemma4Recipe`](../src/tt_symbiote/models/gemma4/modeling_gemma4.py)),
which is a **Phase 7 CPU-first port**: every Gemma-4 submodule
(vision tower, multimodal projection, text decoder) runs on the host
PyTorch backend. The recipe still drives the documented
`AutoModelForImageTextToText.from_pretrained` →
`set_device(model, device)` → `model.generate(...)` flow, and TTNN
wrappers will swap submodules from `cpu_fallback` to `tt_implemented`
incrementally in subsequent commits. Confirm with
`tt_symbiote.compatibility.report(model)` after any demo run — the
``runtime_observed.unexpected`` field must stay empty.

Verified semantic answer for E2B (verbatim):

> *"The animal in the photo is a **dog**. It appears to be a
> light-colored, fluffy breed, possibly a Golden Retriever puppy or
> a similar breed."*

against [`tests/images/test-dog.png`](../tests/images/test-dog.png)
and the prompt `"What is this animal in the photo?"`.

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
