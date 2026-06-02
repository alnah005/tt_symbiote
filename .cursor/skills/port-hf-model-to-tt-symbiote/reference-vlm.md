# VLM (image-text-to-text) port specifics

Reference material for `port-hf-model-to-tt-symbiote` when the target HF
model maps to `AutoModelForImageTextToText` — i.e. a vision-language
model that takes image + text in and produces text out. Read this in
addition to [`SKILL.md`](SKILL.md).

The canonical CPU-first VLM port is
[`google/gemma-4-E2B-it`](../../src/tt_symbiote/models/gemma4/) (committed
in Phase 7; see [`docs/internal/migration_notes.md`](../../docs/internal/migration_notes.md)).
Copy its shape exactly; only the deltas listed below need attention per
model. Phase 8 Wave A added 5 on-device TTNN swaps on top of the
CPU-first scaffolding (RMSNorm, scaled word embedding, text MLP, vision
MLP, multimodal embedder) plus a budget/MoE gate; those swaps are a
follow-up *after* this skill runs, not part of the skill's output.

## Architecture shape

Every VLM in `transformers` follows the same skeleton:

```
<Model>ForConditionalGeneration
├── model: <Model>Model
│   ├── vision_tower (e.g. Gemma4VisionModel, Qwen3VLVisionModel)
│   ├── language_model / text_model (the decoder)
│   ├── (optional) audio_tower
│   └── multimodal embedder / projection (vision -> text-space)
└── lm_head: nn.Linear
```

The recipe registers under the **`*ForConditionalGeneration`** class, never
under the inner `*Model`. `AutoModelForImageTextToText` resolves this for
you — confirm via `transformers/src/transformers/models/auto/modeling_auto.py`
that the model is in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`.

## Class enumeration: what goes where

Use this matrix when populating the recipe's three lists:

| HF class category | List |
|---|---|
| `<Model>VisionPatchEmbed`, `<Model>VisionRotaryEmbedding`, `<Model>VisionAttention`, `<Model>VisionMLP`, `<Model>VisionBlock` / `EncoderLayer`, `<Model>VisionEncoder`, `<Model>VisionPooler`, `<Model>VisionModel`, `<Model>VisionPatchMerger` | `cpu_fallback` |
| `<Model>TextRotaryEmbedding`, `<Model>TextRMSNorm`, `<Model>TextAttention`, `<Model>TextMLP`, `<Model>TextDecoderLayer`, `<Model>TextModel`, `<Model>TextScaledWordEmbedding` | `cpu_fallback` |
| `<Model>MultimodalEmbedder` / `*MultimodalProjector` / `*VisionPatchMerger` | `cpu_fallback` |
| `<Model>Model`, `<Model>ForConditionalGeneration` | `host_glue` (top-level composites that own orchestration with no FLOPs to accelerate — `masked_scatter`, mask building, logit softcap, etc.) |
| `<Model>RMSNorm`, `<Model>ClippableLinear`, etc. — shared utilities | `cpu_fallback` |
| `<Model>AudioModel` / any audio class | `out_of_scope` (image-text demo doesn't exercise audio) |
| `<Model>ForCausalLM` | `out_of_scope` (alternative text-only head, not the VLM path) |
| Output dataclasses: `*ModelOutputWithPast`, `*CausalLMOutputWithPast`, etc. | `out_of_scope` (not torch modules) |
| MoE pair `<Model>TextExperts` + `<Model>TextRouter` (when present) | `cpu_fallback` (the dense recipe is shared with MoE variants) |

`host_glue` was added in Phase 8 alongside the first Gemma-4 TTNN
swaps. The intuition: classes whose forward is `masked_scatter` + mask
construction + (optional) logit softcap have effectively zero FLOPs;
flagging them separately from `cpu_fallback` keeps the actionable
backlog focused on compute, not glue. Recipes that don't declare
`host_glue` still work — `compatibility.report` defaults to an empty
list.

The disjointness invariant — checked by the recipe test — is that no
class name appears in two lists.

## Processor surface (this is where most VLM bugs come from)

Every supported VLM ships a `<Model>Processor` accessible via `AutoProcessor`.
The chat template is structured, not a string:

```python
from PIL import Image
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID)
image = Image.open("tests/images/test-dog.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is this animal in the photo?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
```

Pitfalls observed across Gemma-4 and Qwen3-VL:

- **`return_dict=True`** *and* **`tokenize=True`** are required to get a kwargs dict that `model.generate(**inputs, ...)` accepts. Forgetting either yields a tokenized list or a string.
- The image goes inside `content` as `{"type": "image", "image": <PIL>}`. Passing it as `images=` kwarg works on some models but not others; the structured form is universal.
- `inputs` may include `pixel_values` AND `pixel_values_videos` (and `input_features` for audio models). The `model.generate` call is fine with extras — don't try to drop them manually.
- After `apply_chat_template`, **explicitly `.to("cpu")`** every tensor in the dict. The CPU-first port keeps weights on CPU; HF can occasionally route tensors to a non-existent default device.

```python
inputs = {k: (v.to("cpu") if hasattr(v, "to") else v) for k, v in inputs.items()}
```

## Generation call

Use deterministic generation so the semantic assertion is reproducible:

```python
out = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    use_cache=True,
)
```

64 tokens is plenty — typical "what is this?" answers are well under 30.
The model's EOS terminates earlier.

Decoding skips the prompt:

```python
prompt_len = inputs["input_ids"].shape[-1]
answer = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()
```

## Semantic check

For the canonical Phase 7 image
[`tests/images/test-dog.png`](../../tests/images/test-dog.png) (an AVIF
of a fluffy light-coloured puppy) and prompt
`"What is this animal in the photo?"`:

```python
_DOG_EQUIVALENTS = (
    "dog", "puppy", "retriever", "labrador", "poodle",
    "terrier", "spaniel", "shepherd", "husky", "bulldog",
)
_lower = answer.lower()
assert any(term in _lower for term in _DOG_EQUIVALENTS), (
    f"Expected the answer to mention a dog or a dog breed. Got: {answer!r}."
)
```

**Do not** use a strict `"dog" in answer.lower()` check. Some VLMs
(observed on Qwen3-VL-2B-Instruct) skip the generic word "dog" entirely
and jump straight to a breed: *"the animal is a Golden Retriever
puppy"*. Calling that a failure would mask the fact that the model is
working correctly — it's giving a *more specific* answer than asked.
The breed list above covers the breeds Gemma-4 / Qwen3-VL have
volunteered for this image so far; extend it if a future model
identifies the same dog as something not in the list.

Verified answers from existing ports for cross-reference:
- Gemma-4 E2B-it: `"The animal in the photo is a **dog**. It appears to be a light-colored, fluffy breed, possibly a Golden Retriever puppy or a similar breed."`
- Qwen3-VL-2B-Instruct: `"Based on the visual characteristics in the photo, the animal is a **puppy**. More specifically, it appears to be a **Golden Retriever puppy**."`

If a new VLM gives a confidently wrong answer (e.g. "cat", "rabbit") the
problem is almost always image-token splicing — verify
`processor.apply_chat_template` was called with `tokenize=True` and that
the input tensor dict contains `pixel_values` non-None.

## Per-variant tuning table

The configuration file's `<NAME>_TTNN_TUNING` dict shapes a row per
variant. Fields are:

| Field | Meaning | Typical |
|---|---|---|
| `mesh_shape` | Tuple passed to `ttnn.open_mesh_device` | `(1, 1)` for ≤ 12 GB BF16 single-chip, `(1, 8)` for T3K |
| `l1_small_size` | Reserve for sliding-window halo / small ops | `245760` (matches ResNet); harmless for CPU-first |
| `dtype` | `bfloat16` for everything |
| `hw_verified` | Flip to `True` only when the demo passes |
| `notes` | Free-text shown in docs |

Resolver `lookup_ttnn_tuning(model)` does the same checkpoint -> shape ->
default ladder ResNet does. The shape key for VLM is
`(text_config.num_hidden_layers, has_vision, has_audio)`.

## Mesh device lifecycle

Even on the CPU-first path the demo opens a TTNN mesh:

```python
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)   # single-chip
# or ttnn.FabricConfig.FABRIC_1D_RING for T3K

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(*<MESH_SHAPE>),
    trace_region_size=200_000_000,
    num_command_queues=1,
    l1_small_size=245760,
)
```

Why? Three reasons:

1. The recipe's `post_register` reads `mesh_shape` from
   `_tt_runtime_config`, which downstream TTNN-port commits will
   consume immediately when they swap their first wrapper in.
2. The mesh handle is the only way `set_device` can attach the
   distributed config that later TTNN modules need.
3. Demonstrates the **documented user-facing API**. A demo that
   skipped the mesh open would set a bad example for users.

Always pair `open_mesh_device` with `ttnn.close_mesh_device(mesh_device)`
plus `ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)` on T3K paths.

## set_device call

```python
set_device(model, mesh_device, dump_visualization=False)
assert hasattr(model, "_tt_runtime_config"), (
    "<Class>Recipe.post_register should have attached _tt_runtime_config"
)
```

`dump_visualization=False` — VLMs have 30-60 layer text decoders plus a
30-layer vision tower; the default graph PNG is enormous and useless.

## Trust-remote-code

Most stock transformers VLMs (Gemma-4, Qwen3-VL) live inside the
transformers package proper — `trust_remote_code` is **not** required
and should not be set. Older or community VLMs (e.g. some
`internlm/internvl-*`) need `trust_remote_code=True`; that's also when
[`_hf_compat.py`](../../src/tt_symbiote/_hf_compat.py) shims become
relevant.

## Worked example: Qwen3-VL

Variables extracted in Phase A, fed into the templates in Phase C:

| Variable | Value |
|---|---|
| `<NAME>` | `qwen3_vl` |
| `<HF_CLASS>` | `Qwen3VLForConditionalGeneration` |
| `<AUTO_CLASS>` | `AutoModelForImageTextToText` |
| `<CLASS_NAME_PASCAL>` | `Qwen3VLRecipe` |
| `<MODEL_ID>` (chosen) | `Qwen/Qwen3-VL-2B-Instruct` |
| `<MESH_SHAPE>` | `(1, 1)` |
| Vision classes | `Qwen3VLVisionMLP`, `Qwen3VLVisionPatchEmbed`, `Qwen3VLVisionRotaryEmbedding`, `Qwen3VLVisionPatchMerger`, `Qwen3VLVisionAttention`, `Qwen3VLVisionBlock`, `Qwen3VLVisionModel` |
| Text classes | `Qwen3VLTextRotaryEmbedding`, `Qwen3VLTextRMSNorm`, `Qwen3VLTextAttention`, `Qwen3VLTextMLP`, `Qwen3VLTextDecoderLayer`, `Qwen3VLTextModel` |
| Top-level | `Qwen3VLModel`, `Qwen3VLForConditionalGeneration` |
| Variant table | 2B / 4B / 8B / 32B (dense), 30B-A3B / 235B-A22B (MoE) |
| OOS | `BaseModelOutputWithDeepstackFeatures`, `Qwen3VLModelOutputWithPast`, `Qwen3VLCausalLMOutputWithPast` (output dataclasses); no audio classes exist |

Note Qwen3-VL has no audio tower (unlike Gemma-4 E2B), so the
`out_of_scope` list is shorter — just the output dataclasses.

## Follow-on Phase 8 wrappers (post-skill)

Once the skill produces the CPU-first scaffolding, the next commit
follows the Wave A / Wave B pattern: pick the **structurally simple,
FLOP-intensive** classes (RMSNorm with scale, scaled embedding, text
MLP / vision MLP / patch merger / multimodal embedder) and wrap them
using existing TTNN integrations. The harder bespoke pieces (text
attention with KV sharing / per-head Q/K norms / M-RoPE,
position-aware vision pooling, varlen-packed vision SDPA) stay in
`cpu_fallback` until model-specific TTNN kernels are bespoke-engineered.

For models whose Wave A swap map exceeds the per-chip DRAM budget
(observed at ~9 GB for the current `TTNNLinear` replicated-weight
path), copy Gemma-4's `_ttnn_swap_is_safe` gate verbatim. For MoE
variants whose dense-only Wave A swaps would produce many runtime
fallbacks, copy the MoE branch of the same gate.
