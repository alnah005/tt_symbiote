---
name: port-hf-model-to-tt-symbiote
description: >-
  Ports a HuggingFace transformers model into the tt_symbiote repo as a
  CPU-first recipe, end-to-end: file tree, recipe registration, unit
  tests, hardware-verified e2e demo, and the supported_models / migration
  notes updates. Use when the user asks to port, add, integrate, convert,
  or onboard a HuggingFace model (causal LM, image classification, or
  vision-language model) into tt_symbiote, especially when they reference
  a `transformers/src/transformers/models/<name>/` folder.
---

# port-hf-model-to-tt-symbiote

The repeatable workflow encoded by this skill ships a CPU-first port: the
HF reference modeling stays as PyTorch on the host, while the recipe
registers under the correct `Auto*` class, declares which submodules
**would** be TTNN-accelerated (`tt_implemented`), which run on CPU
(`cpu_fallback`), and which are deliberately not exercised
(`out_of_scope`). The result is a working `from_pretrained` -> `set_device`
-> `generate` / forward flow with the [compatibility report](../../src/tt_symbiote/utils/compatibility.py)
clean on every run. Subsequent commits port submodules from
`cpu_fallback` to `tt_implemented` without ever changing the public API.

## Mental model

```
HF transformers source -->  Recipe (CPU-first)  -->  tt_symbiote Auto* loader
        |                          |                            |
   class enumeration         build_module_dict={}        set_device(model, dev)
   variant table            post_register patches device       compatibility.report
   auto_class lookup        coverage lists for tracking
```

Three ports landed before this skill — they are the load-bearing
references. Cite them; do not re-design.

| Task | Reference port | Recipe shape |
|---|---|---|
| Causal LM | `inclusionAI/Ling-mini-2.0` | [`bailing_moe_v2`](../../src/tt_symbiote/models/bailing_moe_v2/) (full TTNN) |
| Image classification | `microsoft/resnet-50` | [`resnet`](../../src/tt_symbiote/models/resnet/) (full TTNN) |
| Vision-language (VLM) | `google/gemma-4-E2B-it` | [`gemma4`](../../src/tt_symbiote/models/gemma4/) (CPU-first; the canonical CPU-first template) |

## Phase A — Discovery (read-only)

Goal: gather every parameter the templates need without writing anything.

Run these checks in parallel:

1. **HF source enumeration.** Grep `^class\s+\w+` against `transformers/src/transformers/models/<hf_name>/modeling_<hf_name>.py` to get every class. Split them into three buckets:
   - **Exercised on demo path** (text decoder + vision tower + multimodal projection, if applicable) -> `cpu_fallback`.
   - **Not exercised** (audio tower if user asked image+text only; alternative top-level heads like `*ForCausalLM` when targeting the conditional-generation head; output dataclasses) -> `out_of_scope`.
   - **Shared utilities** (`*RMSNorm`, `*MLP` re-used across towers) -> `cpu_fallback`.

2. **Auto class mapping.** Grep the HF model id against `transformers/src/transformers/models/auto/modeling_auto.py`. The first match in `MODEL_FOR_*_MAPPING_NAMES` decides the `tt_symbiote.AutoModel*` class:
   - `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` -> `AutoModelForCausalLM`
   - `MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES` -> `AutoModelForImageClassification`
   - `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` -> `AutoModelForImageTextToText` (VLM)
   - `MODEL_FOR_MULTIMODAL_TO_TEXT_MAPPING_NAMES` -> `AutoModelForMultimodalLM`

3. **Variant table.** Search the model's HF Hub page or `QwenLM/<name>` README for canonical instruction-tuned checkpoints. Record `mesh_shape` per variant:
   - Dense models that fit a single chip (~ ≤ 12 GB BF16) -> `(1, 1)` (N150).
   - Dense models requiring tensor parallelism -> `(1, 8)` (T3K).
   - MoE: same as their dense-equivalent memory footprint.

4. **Task type.** This drives the rest of the workflow:
   - LLM -> read [reference-llm.md](reference-llm.md), use `templates/e2e_demo_llm_template.py`.
   - Vision -> read [reference-vision.md](reference-vision.md), use `templates/e2e_demo_vision_template.py`.
   - VLM -> read [reference-vlm.md](reference-vlm.md), use `templates/e2e_demo_vlm_template.py`. **This is the CPU-first default.**

## Phase B — Open questions (AskQuestion)

Ask exactly two questions before scaffolding:

1. **Model variant** — list the variants discovered in Phase A, mark the smallest-that-fits-target as `RECOMMENDED`.
2. **TTNN coverage scope** — `cpu_first` (RECOMMENDED for the first commit of any model) / `partial_ttnn` / `full_ttnn`. The skill below only encodes `cpu_first`; the other two require model-specific TTNN wrappers and are out of scope for the first commit.

Do not start scaffolding until both questions are answered.

## Phase C — Scaffold

Drop the file tree using the templates in [templates/](templates/). Substitute:

| Placeholder | Source |
|---|---|
| `<NAME>` | HF model directory name, e.g. `qwen3_vl`. Used in file paths and recipe class names. |
| `<HF_CLASS>` | Top-level conditional-generation / causal-LM / classification class. Decides `@register_recipe(hf_class_name=...)`. |
| `<AUTO_CLASS>` | `tt_symbiote.AutoModel*` class from Phase A step 2. |
| `<MODEL_ID>` | HF Hub id of the chosen variant (e.g. `Qwen/Qwen3-VL-2B-Instruct`). |
| `<MESH_SHAPE>` | `(1, 1)` for single chip, `(1, 8)` for T3K. |
| `<CLASS_NAME_PASCAL>` | The recipe's class name, e.g. `Qwen3VLRecipe`. |
| `<CPU_FALLBACK_LIST>` | Newline-separated class names from Phase A step 1. |
| `<OUT_OF_SCOPE_LIST>` | Same. |
| `<VARIANT_TABLE>` | Per-variant tuning rows for `configuration_<name>.py`. |

Files to produce, in order:

```
src/tt_symbiote/models/<NAME>/
├── __init__.py                       <- from templates/init_template.py
├── configuration_<NAME>.py           <- from templates/configuration_template.py
└── modeling_<NAME>.py                <- from templates/modeling_template.py

src/tt_symbiote/models/__init__.py    <- APPEND "<NAME>" to _RECIPE_BEARING_SUBPACKAGES

tests/auto/test_<NAME>_recipe.py      <- from templates/recipe_test_template.py
tests/models/<NAME>/test_modeling_<NAME>.py
                                      <- from templates/smoke_test_template.py

examples/e2e/run_<MODEL>.py           <- from templates/e2e_demo_<TASK>_template.py
                                         (TASK in {llm, vision, vlm})
```

**Do not edit the templates** — they are the source of truth. Substitute placeholders, write the substituted files into the target paths, then run the post-scaffold checks below.

Post-scaffold sanity:

```bash
cd /home/aroberge/tt_symbiote && source .venv/bin/activate
python -c "
import tt_symbiote
print(sorted(tt_symbiote.TT_MODEL_REGISTRY.keys()))
recipe = tt_symbiote.TT_MODEL_REGISTRY['<HF_CLASS>']
print('cpu_fallback count:', len(recipe.cpu_fallback))
print('out_of_scope count:', len(recipe.out_of_scope))
print('build_module_dict:', recipe.build_module_dict(None))
"
python -m pytest tests/auto/test_<NAME>_recipe.py tests/models/<NAME>/ -v
python -m pytest tests/auto/ -q     # full HW-free regression check
```

All three commands must succeed before moving to Phase D.

## Phase D — Hardware verification

Run the e2e demo on the listed hardware target:

```bash
python examples/e2e/run_<MODEL>.py
```

Expect:
1. Exit code 0.
2. Task-specific semantic check passes — the demo script enforces this with an `assert` so a wrong answer fails the run:
   - LLM: coherent EOS-terminated text (no assertion, manual eyeball).
   - Vision: top-1 ImageNet label is plausible.
   - **VLM: decoded answer contains the expected token. For the canonical [`tests/images/test-dog.png`](../../tests/images/test-dog.png) + `"What is this animal in the photo?"` prompt, the answer must contain `"dog"`.**
3. The `compatibility.report(model)` printed at the end shows `runtime_observed.unexpected == []` (every CPU fallback was declared in the recipe).

If any of the above fails, do **not** edit the recipe to mask the failure. Iterate:

- `GatedRepoError` -> ask user to accept the HF license on the model card.
- `Missing key` errors from `from_pretrained` -> check whether the HF source uses `trust_remote_code=True` (some models do) and add it to the demo.
- `KeyError`/`AttributeError` deep inside `transformers` -> probably needs a [_hf_compat.py](../../src/tt_symbiote/_hf_compat.py) shim; add one and document the reason.
- Image decode failure on AVIF inputs -> install `pillow-heif` (`pip install pillow-heif`).
- Unexpected runtime fallback -> add the class to the recipe's `cpu_fallback` (or move from `out_of_scope` if applicable), commit-blocking until empty.

## Phase E — Docs + commit

1. **`docs/supported_models.md`** — append a row under the right task section. Use the same `TT / CPU / OOS` count column the Gemma-4 row uses. If the model has multiple variants, list all of them with `⏳ structurally supported` and mark the verified one `✅ verified`.

2. **`docs/migration_notes.md`** — append a section titled `## Phase 7 — <Model> port via porting skill`. Mirror the Gemma-4 section's structure: pivot rationale, what shipped, follow-ups. **Cite the skill** at `.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md`.

3. **`examples/e2e/README.md`** — append a row with the script path, task type, and verified hardware target.

4. **Commit** (no push):
   ```bash
   git add <every file touched>
   git commit -m "feat: phase 7 - <NAME> port (CPU-first, <variant id> verified on <hw>)"
   ```
   Conventional message body should mention: model id, hardware target,
   semantic check that passed, count of `cpu_fallback` classes, compatibility
   report cleanliness, and explicit follow-ups (TTNN wrappers per
   submodule).

**Never** `git push` from this skill. The user pushes manually.

## Acceptance gates (commit blockers)

All must pass; do not commit otherwise:

```
[ ] ReadLints on every file the skill touched -> 0 errors.
[ ] python -m pytest tests/auto/ -q              -> full HW-free suite green
[ ] python -m pytest tests/models/<NAME>/ -v     -> new model's HW-free tests green
[ ] python examples/e2e/run_<MODEL>.py            -> exit 0 + semantic assert ok
[ ] compatibility.report(model)["runtime_observed"]["unexpected"] == []
[ ] docs/supported_models.md row exists with the right status flag
[ ] docs/migration_notes.md Phase 7 section exists and cites this skill
[ ] git status (after staging) lists every artefact; no surprise files
```

## Trigger phrases

The agent should invoke this skill when the user says things like:

- *"Port `<model>` to tt_symbiote"*
- *"Add `Qwen3-VL` support"*
- *"Convert `transformers/src/transformers/models/<x>` into a tt_symbiote recipe"*
- *"Onboard `<HF model id>` to the new repo"*
- *"Phase 7 model port"*

## Reference material

- [`reference-llm.md`](reference-llm.md) — causal LM specifics (Ling-mini-2.0 walkthrough).
- [`reference-vision.md`](reference-vision.md) — image classification specifics (ResNet walkthrough).
- [`reference-vlm.md`](reference-vlm.md) — VLM specifics (Gemma-4 walkthrough + Qwen3-VL applied example).
- [`templates/`](templates/) — file templates with `<PLACEHOLDER>` markers.

## What this skill deliberately does *not* do

- Decide whether a model needs TTNN wrappers. That's a follow-up commit per submodule.
- Optimise sharding strategy or KV-cache sizing. The CPU-first port uses HF `DynamicCache` and runs entirely on the host.
- Touch `set_device` / `register_modules` / the `Auto*` factory. Those are stable Phase 4-5 APIs.
- Push commits to remote. User pushes manually.
- Re-run the model graph visualization (`dump_visualization=False` in every demo to avoid noise on multi-tower models).
