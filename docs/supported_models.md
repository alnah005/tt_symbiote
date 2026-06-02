# Supported models

Catalog of models that ship with `tt_symbiote` recipes. A row in this
table means the model can be loaded through
`tt_symbiote.AutoModel*.from_pretrained`, bound to a Tenstorrent device
via `tt_symbiote.set_device(model, device)`, and called through the
standard HuggingFace forward / generate API.

Hardware verification is split into three states:

- ✅ **verified** — end-to-end smoke pass on the listed hardware target,
  with zero unexpected TTNN-forward-fallback warnings, and (for vision)
  semantically correct top-1 on a known image / (for LLM) coherent
  EOS-terminated output / (for VLM) semantically correct answer to the
  documented prompt. CPU-only ports count as verified iff every CPU
  module is explicitly listed in the recipe's `cpu_fallback` (so
  `tt_symbiote.compatibility.report(model)["regressions"]` is empty).
- ⏳ **structurally supported** — recipe + module replacement land,
  unit tests pass. Hardware forward has not been exercised yet.
- ⚠ **partial** — runs but with documented caveats. See the linked
  walkthrough for what works and what doesn't.

When a row is at ✅ there is a matching script in
[`examples/e2e/`](../examples/e2e/) that reproduces the run on the
listed hardware.

Reproducing any ✅ row requires a working `(ttnn, sfpi)` install on
the host — see [`install_prerequisites.md`](install_prerequisites.md)
for what to install and how to verify it.

## TT-implemented vs. CPU coverage

Every recipe declares four class-name lists that drive
[`tt_symbiote.compatibility.report(model)`](../src/tt_symbiote/utils/compatibility.py):

- **`tt_implemented`** — HF classes for which the recipe ships a TTNN
  wrapper (i.e. it is in `build_module_dict`). These run on
  Tenstorrent silicon.
- **`cpu_fallback`** — HF classes that are *exercised* by the
  documented demos but stay as PyTorch in this recipe *for now*. These
  run on the host CPU. Entries here are the actionable backlog — they
  represent ops with FLOPs to move onto the device.
- **`host_glue`** — HF classes that are intentionally CPU-only by
  policy (orchestration, mask building, masked-scatter fusion, output
  dataclasses, index walks). These have no FLOPs to accelerate;
  flagging them separately from `cpu_fallback` keeps the backlog
  focused on compute, not glue. Added in Phase 8 alongside the first
  Gemma-4 TTNN swaps.
- **`out_of_scope`** — HF classes that exist in the upstream
  modeling file but aren't touched by the documented demos (e.g. the
  Gemma-4 audio tower under the image-text path, or output
  dataclasses that aren't modules at all).

Per-model the "TT / CPU / host_glue / OOS" counts in the tables below
are read directly from the recipes. At runtime,
`compatibility.report(model)["regressions"]` flags any fallback class
the recipe did *not* declare under `cpu_fallback` — a clean run leaves
that list empty.

For the *full* CPU-vs-device split per demo (which exact classes run
where, and on what hardware), see
[`docs/development/cpu_vs_device_coverage.md`](development/cpu_vs_device_coverage.md). Each
demo script also writes a `<script>_coverage.json` next to itself
after every run. **Phase 8.5** made these JSONs pure runtime
observation artefacts — gitignored, regenerated on every run, read
locally to confirm the run was clean. The recipe declarations stay in
source code; the JSON serializes only what actually happened.

## Causal LM

| Checkpoint | HF class | Status | TT / CPU / OOS | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|---|
| `inclusionAI/Ling-mini-2.0` | `BailingMoeV2ForCausalLM` | ✅ verified | full TTNN | T3K (1×8) | [`run_ling_mini_2_0.py`](../examples/e2e/run_ling_mini_2_0.py) | [`ling_mini_2_0_guide.md`](ling_mini_2_0_guide.md) |

## Image classification

| Checkpoint | HF class | Status | TT / CPU / OOS | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|---|
| `microsoft/resnet-18` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | [`resnet/run_resnet18.py`](../examples/e2e/resnet/run_resnet18.py) | — |
| `microsoft/resnet-34` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | [`resnet/run_resnet34.py`](../examples/e2e/resnet/run_resnet34.py) | — |
| `microsoft/resnet-50` | `ResNetForImageClassification` | ✅ verified | full TTNN | N150 (1×1), T3K (1×1) | [`resnet/run_resnet50.py`](../examples/e2e/resnet/run_resnet50.py) | — |
| `microsoft/resnet-101` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | [`resnet/run_resnet101.py`](../examples/e2e/resnet/run_resnet101.py) | — |
| `microsoft/resnet-152` | `ResNetForImageClassification` | ⏳ structurally supported | full TTNN | N150 (1×1) | [`resnet/run_resnet152.py`](../examples/e2e/resnet/run_resnet152.py) | — |

The four non-50 ResNet variants share the same recipe; flipping each to
✅ requires running the per-variant script and updating the
`hw_verified` flag in
[`RESNET_TTNN_TUNING`](../src/tt_symbiote/models/resnet/configuration_resnet.py).

## Image-text-to-text (VLM)

| Checkpoint | HF class | Status | TT / CPU / glue / OOS | Hardware target | Reproducer | Walkthrough |
|---|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | `Gemma4ForConditionalGeneration` | ✅ verified (Phase 8 Wave A) | 5 / 14 / 2 / 14 | N150 (1×1) | [`gemma4/run_gemma4_e2b.py`](../examples/e2e/gemma4/run_gemma4_e2b.py) | — |
| `google/gemma-4-E4B-it` | `Gemma4ForConditionalGeneration` | ✅ verified (Phase 8 Wave A) | 5 / 14 / 2 / 14 | N150 (1×1) | [`gemma4/run_gemma4_e4b.py`](../examples/e2e/gemma4/run_gemma4_e4b.py) | — |
| `google/gemma-4-31B-it` | `Gemma4ForConditionalGeneration` | ✅ verified (CPU-only via budget gate) | 0 / 19 / 2 / 14 | T3K (1×8) | [`gemma4/run_gemma4_31b.py`](../examples/e2e/gemma4/run_gemma4_31b.py) | — |
| `google/gemma-4-26B-A4B-it` | `Gemma4ForConditionalGeneration` | ✅ verified (CPU-only via MoE gate) | 0 / 19 / 2 / 14 | T3K (1×8) | [`gemma4/run_gemma4_26b_a4b.py`](../examples/e2e/gemma4/run_gemma4_26b_a4b.py) | — |
| `Qwen/Qwen3-VL-2B-Instruct` | `Qwen3VLForConditionalGeneration` | ✅ verified (Phase 8 Wave B) | 4 / 9 / 3 / 3 | N150 (1×1) | [`qwen3_vl/run_qwen3_vl_2b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_2b.py) | — |
| `Qwen/Qwen3-VL-4B-Instruct` | `Qwen3VLForConditionalGeneration` | ⏳ structurally supported | 4 / 9 / 3 / 3 | N150 (1×1) | [`qwen3_vl/run_qwen3_vl_4b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_4b.py) | — |
| `Qwen/Qwen3-VL-8B-Instruct` | `Qwen3VLForConditionalGeneration` | ⏳ structurally supported | 4 / 9 / 3 / 3 | N150 (1×1) | [`qwen3_vl/run_qwen3_vl_8b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_8b.py) | — |
| `Qwen/Qwen3-VL-32B-Instruct` | `Qwen3VLForConditionalGeneration` | ⏳ structurally supported | 4 / 9 / 3 / 3 | T3K (1×8) | [`qwen3_vl/run_qwen3_vl_32b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_32b.py) | — |

All four Gemma-4 variants share the same recipe
([`Gemma4Recipe`](../src/tt_symbiote/models/gemma4/modeling_gemma4.py)).
**Phase 8 Wave A** turned the structurally simple compute (`Gemma4RMSNorm`,
`Gemma4TextScaledWordEmbedding`, `Gemma4TextMLP`, `Gemma4VisionMLP`,
`Gemma4MultimodalEmbedder`) into on-device wrappers via the existing
TTNN integrations. The bespoke text-side compute (KV-shared attention,
dual RoPE tables, Per-Layer Embeddings, dual sliding/full mask) and
the bespoke vision-side compute (2-D RoPE, position-aware pooler,
non-causal SDPA) stay declared `cpu_fallback` and are the Wave A+1
backlog. Top-level fusion (`Gemma4Model`,
`Gemma4ForConditionalGeneration`) is declared `host_glue` — it owns
the `masked_scatter` of vision tokens into the text embedding stream,
the sliding/full causal mask construction, and the optional final
logit softcap, none of which have FLOPs worth moving. Confirm with
`tt_symbiote.compatibility.report(model)` after any demo run — the
``regressions`` field must stay empty.

### Budget + MoE gating for the larger Gemma-4 variants

`Gemma4Recipe.build_module_dict` runs a `_ttnn_swap_is_safe(model)`
check before returning the Wave A swap map. Two conditions short-
circuit it to an empty dict (no swaps, full CPU execution) and emit
a `UserWarning`:

1. **Replicated weight footprint > 9 GB per chip.** The current
   `TTNNLinear` / `TTNNEmbedding` integrations replicate weights
   across every chip in the mesh; tensor-parallel sharding is a
   Wave A+2 follow-up. The 31B-it variant's Wave A swaps alone
   would replicate ~43.5 GB across each T3K chip (8× over the
   12 GB DRAM budget). The gate forces 31B-it onto CPU until the
   sharding work lands.
2. **MoE layout (`text_config.enable_moe_block == True`).** The
   26B-A4B-it variant ships `Gemma4TextExperts` + `Gemma4TextRouter`
   plus per-head RMSNorms whose `dim` (32 / 96) is incompatible with
   the current TTNN RMSNorm tile geometry. A partial swap produces
   hundreds of shape-validation fallbacks at runtime. The gate
   forces the MoE variant onto CPU until those wrappers are
   bespoke-ported.

When a variant is gated, `model._tt_runtime_config` carries
diagnostic flags (`ttnn_swap_skipped`, `ttnn_replicated_footprint_bytes`,
`ttnn_swap_skipped_reason`) so the user can inspect *why* the swap
was skipped. The runtime JSON reflects this cleanly:
`ttnn_swap_skipped == true`, `modules_swapped == {by_class: {},
by_module: {}}`, `regressions == []`, and the model produces the same
semantically correct answer as the smaller variants. See
[`docs/development/cpu_vs_device_coverage.md`](development/cpu_vs_device_coverage.md)
"Budget and MoE gating" for the data behind the 9 GB threshold and
the per-variant headline numbers.

Verified semantic answer for E2B (verbatim, Phase 8 Wave A run):

> *"The animal in the photo is a **dog**. It appears to be a young
> Golden Retriever or a similar light-colored breed."*

against [`tests/images/test-dog.png`](../tests/images/test-dog.png)
and the prompt `"What is this animal in the photo?"`.

All four dense Qwen3-VL variants share the same recipe
([`Qwen3VLRecipe`](../src/tt_symbiote/models/qwen3_vl/modeling_qwen3_vl.py)).
**Phase 8 Wave B** turned the structurally simple compute
(`Qwen3VLTextRMSNorm`, `Qwen3VLTextMLP`, `Qwen3VLVisionMLP`,
`Qwen3VLVisionPatchMerger`) into on-device wrappers via the existing
TTNN integrations. Bespoke compute (M-RoPE, Q/K head-norm-aware text
attention, varlen-packed vision SDPA, DeepStack injection at sparse
layers) stays declared `cpu_fallback` and is the Wave B+1 backlog.
Top-level fusion (`Qwen3VLPreTrainedModel`, `Qwen3VLModel`,
`Qwen3VLForConditionalGeneration`) is declared `host_glue`. The first
model in this family was landed via the
[`port-hf-model-to-tt-symbiote`](../.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md)
Cursor skill; Wave B's TTNN swaps were added by hand on top of that
foundation. The two MoE variants (`Qwen/Qwen3-VL-30B-A3B`,
`Qwen/Qwen3-VL-235B-A22B`) route to a distinct top-level HF class
(`Qwen3VLMoeForConditionalGeneration`) and will be ported under a
separate `qwen3_vl_moe` recipe in a follow-up commit.

Verified semantic answer for Qwen3-VL-2B-Instruct (verbatim, truncated
at 64 tokens):

> *"Based on the visual characteristics in the photo, the animal is a
> **puppy**. More specifically, it appears to be a **Golden Retriever
> puppy**. This is indicated by several key features: - Coat Color: The
> puppy has a light, golden-brown coat, which is the typical color"*

The model identifies the animal as a Golden Retriever puppy, which is
unambiguously a dog. The demo's semantic assertion accepts any of
``{"dog", "puppy", "retriever", "labrador", "poodle", "terrier",
"spaniel", "shepherd", "husky", "bulldog"}`` rather than a strict
``"dog"`` substring — see
[reference-vlm.md](../.cursor/skills/port-hf-model-to-tt-symbiote/reference-vlm.md)
"Semantic check" for the rationale.

## How to add a model

**Prefer the automated skill.** The Cursor skill at
[`.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md`](../.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md)
encodes the full workflow used for Qwen3-VL-2B and reproduces every
artefact below from a small set of placeholders. Trigger it by asking
your Cursor agent to *"port `<HF model id>` to `tt_symbiote`"*.

If you'd rather do it by hand, the manual checklist is:

1. Add the recipe under `src/tt_symbiote/models/<name>/` following the
   `bailing_moe_v2` (LLM), `resnet` (vision), `gemma4`/`qwen3_vl` (VLM)
   layout: a `configuration_<name>.py` (re-export HF config + TTNN
   tuning table if needed), a `modeling_<name>.py` (TTNN wrappers + a
   class decorated with `@register_recipe(hf_class_name=...)`), and an
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

See [`docs/development/migration_notes.md`](development/migration_notes.md) for the design
rationale behind each Phase.
