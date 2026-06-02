# tt_symbiote — Project Proposal

**Status:** Approved + executed. Phases 1–8 (Wave B) complete on
`aroberge/bootstrap`. This document is preserved verbatim as the
original (frozen) design intent — for live status, see
[`README.md`](README.md) (top-level), the per-phase rationale in
[`docs/migration_notes.md`](docs/migration_notes.md), and the
per-variant verification state in
[`docs/supported_models.md`](docs/supported_models.md) +
[`docs/cpu_vs_device_coverage.md`](docs/cpu_vs_device_coverage.md).
Open questions Q1, Q2, Q3, Q4, Q5, Q9 are resolved in
`docs/migration_notes.md` Phase 2–8 sections; Q6 (CI runner topology),
Q7 (auto-sync of vendored ccl.py), Q8 (recursive recipes), and Q10
(multi-version in one checkout) remain deferred.
**Owner:** Suhail Alnahari / Adam Roberge
**Target first release:** `tt_symbiote 0.1.0` pinned to `transformers v5.9.0`
**Migration source:** the directory `models/experimental/tt_symbiote/` on the `alnah005/tt_symbiote_v2` branch of the `tt-metal` repo. This is *source material to copy*, not a branch to fork — see §13 for the exact workflow.
**Build target:** the new standalone repo at `github.com/alnah005/tt_symbiote` (currently empty). All new work happens here.

---

## 1. Vision

`tt_symbiote` becomes the **Hugging Face of Tenstorrent**: a standalone, pip-installable Python library whose user-facing API is **identical** to `transformers`, and whose folder layout **mirrors** `transformers/src/transformers/` so that any developer already familiar with HF can navigate it on day one.

A user runs:

```python
from tt_symbiote import AutoModelForCausalLM, set_device
import ttnn

model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0", trust_remote_code=True, dtype="auto"
)
set_device(model, ttnn.open_mesh_device(...))   # mandatory final step

out = model.generate(**inputs, max_new_tokens=128, past_key_values=kv_cache, use_cache=True)
```

The only difference from HF is the import path (`tt_symbiote` instead of `transformers`) and the mandatory `set_device(...)` call after construction.

---

## 2. Guiding principles (locked decisions)

| # | Decision | Rationale |
|---|---|---|
| P1 | **Standalone pip package**, not a `tt-metal` submodule. | Lets us pin our own `transformers` version, branch per HF release, decouple from `tt-transformers`. |
| P2 | **Folder + filenames mirror `transformers/src/transformers/` exactly.** | Lowers the learning curve for HF users; we want people to look at HF docs and find the same files here. |
| P3 | **HF user-facing API is identical.** Full `Auto*` coverage (placeholders allowed for non-production). | Drop-in replacement story; we never want a user to learn new ergonomics. |
| P4 | **One `transformers` version per `tt_symbiote` git branch / release.** | Tight coupling is acceptable because we always pin; deprecations between versions are fine. |
| P5 | **`set_device(model, device)` is mandatory and final.** Missing it must hard-error. | Eliminates a class of silent bugs and forces the explicit boundary between PyTorch and TTNN. |
| P6 | **No dispatcher abstraction.** Fallback for unsupported modules is "run on CPU with a warning." | The dispatcher layer is unreliable, untraceable, and the team explicitly wants it removed. |
| P7 | **Generic TTNN building blocks live in `integrations/`; model-specific subclasses live in `models/<model>/`.** | Mirrors HF's split between `integrations/` (third-party backends like flash-attn, bitsandbytes) and `models/<model>/modeling_<model>.py`. We re-purpose `integrations/` to hold TTNN as a "backend integration." |
| P8 | **Run modes (`NORMAL`, `DPL`, `CPU`, `TRACED`) stay.** Same env-var contract as today. | They are user-facing debugging primitives, unrelated to the dispatcher object hierarchy. |
| P9 | **Each model is a separate `modeling_<model>.py` file**; tests are split into capability-grouped tests + a per-model smoke test (mirrors HF's `tests/models/<model>/test_modeling_<model>.py`). | HF parity for both source and tests. |
| P10 | **Examples (demos, chat, generation scripts) live outside the package** in `examples/`. | Pip users get a lean install; demos are inspiration code, not part of the API surface. |

---

## 3. Target repository layout

The layout below is a 1:1 structural mirror of `transformers/src/transformers/`, plus a TTNN-specific `core/` subpackage for internals.

```text
tt_symbiote/                           # repo root
├── pyproject.toml                     # name="tt_symbiote", version="0.1.0"
├── setup.py                           # if needed for editable installs
├── README.md
├── PROJECT_PROPOSAL.md                # this file
├── LICENSE                            # Apache-2.0 (matches both tt-metal and HF)
├── .github/workflows/                 # CI: lint, smoke tests on N150/T3K via self-hosted runners
│
├── src/tt_symbiote/
│   ├── __init__.py                    # re-exports: AutoModelForCausalLM, AutoModel, set_device, register_modules, …
│   │
│   ├── auto/                          # mirrors transformers/models/auto
│   │   ├── __init__.py
│   │   ├── auto_factory.py            # _BaseAutoModelClass equivalent: HF.from_pretrained → register_modules → return
│   │   ├── auto_mappings.py           # registry of HF class name → tt_symbiote modeling recipe
│   │   ├── configuration_auto.py
│   │   ├── modeling_auto.py           # AutoModel, AutoModelForCausalLM, … all 43 HF Auto* classes (placeholders ok)
│   │   ├── tokenization_auto.py       # thin re-exports from HF
│   │   ├── image_processing_auto.py   # ditto
│   │   ├── feature_extraction_auto.py
│   │   ├── processing_auto.py
│   │   └── video_processing_auto.py
│   │
│   ├── integrations/                  # mirrors transformers/integrations — *generic*, model-agnostic TTNN blocks
│   │   ├── __init__.py
│   │   ├── ttnn_linear.py             # TTNNLinear, TTNNLinearLLama, TTNNLinearIColShardedWRowSharded, TTNNLinearSilu, TTNNLinearGelu
│   │   ├── ttnn_normalization.py      # TTNNLayerNorm, TTNNRMSNorm, TTNNDistributedRMSNorm
│   │   ├── ttnn_activation.py         # TTNNSilu, TTNNReLU, TTNNGelu
│   │   ├── ttnn_embedding.py          # TTNNEmbedding, TTNNPaddedEmbedding
│   │   ├── ttnn_rope.py               # TTNNRotaryEmbedding (+ helpers)
│   │   ├── ttnn_attention.py          # generic TTNNSDPAAttention, TTNNFusedQKVSelfAttention, TTNNSelfAttention, PagedAttentionConfig, TTNNPagedAttentionKVCache
│   │   ├── ttnn_moe.py                # generic TTNNMoE base (Glm4MoeConfig stays; shared MoE machinery lives here)
│   │   ├── ttnn_conv.py               # TTNNConv2dNHWC, TTNNConv2dBNNHWC, TTNNConv2dBNActivationNHWC, TTNNBottleneck, TTNNMaxPool2dNHWC, TTNNUpsampleNHWC, TTNNPatchEmbedding
│   │   └── ttnn_tensor.py             # TTNNPermute, TTNNReshape, TTNNAdd
│   │
│   ├── models/                        # mirrors transformers/models — *model-specific* code
│   │   ├── __init__.py
│   │   ├── bailing_moe_v2/            # Ling-mini-2.0
│   │   │   ├── __init__.py
│   │   │   ├── modeling_bailing_moe_v2.py    # TTNNBailingMoEDecoderLayerPadded, TTNNBailingMoeV2Model, register dict
│   │   │   └── configuration_bailing_moe_v2.py  # (only if we need to override HF config)
│   │   ├── gemma4/
│   │   │   ├── __init__.py
│   │   │   ├── modeling_gemma4.py     # TTNNGemma4DecoderLayer, TTNNGemma4ScaledEmbedding, TTNNGemma4TextModel, TTNNGemma4PagedAttentionKVCache, register dict
│   │   │   └── configuration_gemma4.py
│   │   ├── glm4_moe/                  # GLM-4.5-Air, GLM-4-7, GLM-5, GLM-flash (if they really share modeling)
│   │   ├── qwen3_moe/                 # Qwen3 35B-A3B, Qwen3 Coder Next
│   │   ├── gpt_oss/
│   │   ├── olmo3/
│   │   ├── llama/
│   │   ├── vit/
│   │   ├── owl_vit/
│   │   ├── resnet/
│   │   ├── whisper/
│   │   ├── speech_t5/
│   │   ├── hunyuan_video/
│   │   ├── openvla/
│   │   ├── gr00t/
│   │   ├── qwen_omni/
│   │   ├── molmo2/
│   │   └── …                          # rest of HF model list, populated lazily
│   │
│   ├── core/                          # INTERNAL — runtime + base classes. Not a HF concept.
│   │   ├── __init__.py
│   │   ├── module.py                  # TTNNModule, @run_on_devices, DeviceArch, deallocate_weights_after
│   │   ├── tensor.py                  # TorchTTNNTensor
│   │   ├── run_config.py              # DispatchManager, DistributedConfig, TracedRun, run modes
│   │   ├── ccl.py                     # vendored TT_CCL (see §5)
│   │   ├── model_config.py            # vendored determine_device_name + minimal helpers used by ccl.py
│   │   └── utils.py                   # tree_map, dtype helpers
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── module_replacement.py      # register_modules (renamed from register_module_replacement_dict for HF feel)
│   │   ├── device_management.py       # set_device, DeviceInit
│   │   └── graph_visualization.py     # draw_model_graph (debug only)
│   │
│   └── generation/                    # placeholder if we ever need to override HF generate()
│       └── __init__.py
│
├── tests/                             # mirrors transformers/tests/
│   ├── __init__.py
│   ├── conftest.py                    # pytest fixtures: mesh_device, device_params (moved from current tests/conftest.py)
│   │
│   ├── capabilities/                  # grouped tests by capability (NEW; replaces today's flat tests/)
│   │   ├── test_attention.py          # exercises every TTNNSDPA/FusedQKV variant on a small synthetic input
│   │   ├── test_moe.py
│   │   ├── test_linear.py
│   │   ├── test_normalization.py
│   │   ├── test_rope.py
│   │   ├── test_conv.py
│   │   ├── test_embedding.py
│   │   └── test_dpl.py                # DPL run-mode contract test
│   │
│   └── capabilities/                   # per-model tests (mirrors transformers/tests/models/ naming)
│       ├── bailing_moe_v2/test_modeling_bailing_moe_v2.py    # smoke test: load → register → set_device → generate(max_new_tokens=8)
│       ├── gemma4/test_modeling_gemma4.py
│       ├── glm4_moe/test_modeling_glm4_moe.py
│       └── …                          # one smoke test per model
│
├── examples/                          # OUTSIDE the package. Pip users still get them in the repo, but they aren't installed.
│   ├── README.md
│   ├── chat/HF_chat.py                # ported from current demos/HF_chat.py
│   └── generation/ling_mini.py
│
└── docs/
    ├── architecture.md
    ├── porting_a_new_model.md         # Claude-skill workflow doc
    ├── run_modes.md
    └── images/ARCHITECTURE.svg        # current ARCHITECTURE.svg copied here
```

### Naming convention reconciliation

The current package uses prefix `TTNN` (`TTNNLinear`, `TTNNRMSNorm`, …). We keep that prefix — it visually distinguishes TT classes from HF originals and makes mixed code readable. **TBD during implementation:** whether per-file filenames should drop the `ttnn_` prefix (e.g. `integrations/linear.py` vs `integrations/ttnn_linear.py`); leaning toward `ttnn_*` so the import line `from tt_symbiote.modules.ttnn_linear import TTNNLinear` is unambiguous.

---

## 4. Public API surface (v0.1.0)

### 4.1 Top-level re-exports (`tt_symbiote/__init__.py`)

```python
# Mirrors transformers.__init__ exports — same names, same call signatures.
from .auto.modeling_auto import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForImageTextToText,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoModelForAudioClassification,
    AutoModelForTextToSpectrogram,
    AutoModelForTextToWaveform,
    AutoModelForVideoClassification,
    AutoModelForVision2Seq,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForMultipleChoice,
    AutoBackbone,
    # ... all 43 Auto* classes from transformers.models.auto.modeling_auto
)
from .auto.tokenization_auto import AutoTokenizer
from .auto.image_processing_auto import AutoImageProcessor
from .auto.processing_auto import AutoProcessor
from .auto.feature_extraction_auto import AutoFeatureExtractor

# TT-specific public API (the *only* additions over HF).
from .utils.device_management import set_device
from .utils.module_replacement import register_modules
from .core.run_config import DispatchManager, TracedRun
```

### 4.2 How `Auto*` classes work

Every `tt_symbiote.Auto*` class is a thin wrapper over the HF `Auto*` class that, after loading, applies the registered tt_symbiote modeling recipe for that HF class. Pseudocode for `auto/auto_factory.py`:

```python
class _BaseAutoModelClass:
    _HF_AUTO_CLASS = None        # e.g. transformers.AutoModelForCausalLM
    _MAPPING_NAME = None         # e.g. "MODEL_FOR_CAUSAL_LM"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        model = cls._HF_AUTO_CLASS.from_pretrained(pretrained_name_or_path, *args, **kwargs)
        recipe = TT_MODEL_REGISTRY.lookup(model.__class__.__name__)
        if recipe is None:
            warnings.warn(
                f"No tt_symbiote recipe for {model.__class__.__name__}; "
                f"returning unmodified HF model. set_device() will be a no-op."
            )
            return model
        module_dict = recipe.build_module_dict(model)
        register_modules(model, module_dict)
        recipe.post_register(model)         # optional model-specific hooks (e.g. KV-cache setup, lm_head patches)
        return model
```

The registry is populated by per-model modules at import time via a decorator:

```python
# src/tt_symbiote/models/bailing_moe_v2/modeling_bailing_moe_v2.py
from tt_symbiote.models.auto.auto_mappings import register_recipe

@register_recipe(hf_class_name="BailingMoeV2ForCausalLM")
class BailingMoEV2Recipe:
    def build_module_dict(self, model):
        return {
            model.model.layers[0].__class__: TTNNBailingMoEDecoderLayerPadded,
            model.model.norm.__class__: TTNNDistributedRMSNorm,
            nn.Embedding: TTNNBailingPaddedEmbedding,
            model.model.rotary_emb.__class__: TTNNBailingRotaryEmbedding,
            nn.Linear: TTNNLinearIColShardedWRowSharded,
            nn.SiLU: TTNNSilu,
            model.model.__class__: TTNNBailingMoeV2Model,
        }

    def post_register(self, model):
        type(model).device = property(lambda self: torch.device("cpu"))
```

**TBD during implementation:** the recipe object likely needs to express the "three-pass replacement" pattern (decoder/norm/embed → linear/silu → model wrapper). Open question whether to (a) keep it as a flat dict and run `register_modules` in a fixed pass order internally, or (b) make `build_module_dict` return a list of dicts. Decision deferred until we port the second model.

### 4.3 Full `Auto*` coverage

We will create **all 43 HF Auto* classes**, even for tasks we have no model recipe for. Classes with no recipes raise no error — they just fall back to "load HF, warn, return unmodified." This guarantees a developer copying HF code never hits an `AttributeError: module 'tt_symbiote' has no attribute 'AutoModelForXxx'`.

The list (extracted from `transformers.models.auto.modeling_auto`):

`AutoModel`, `AutoModelForPreTraining`, `AutoModelForCausalLM`, `AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`, `AutoModelForMaskGeneration`, `AutoModelForKeypointDetection`, `AutoModelForKeypointMatching`, `AutoModelForTextEncoding`, `AutoModelForImageToImage`, `AutoModelForSequenceClassification`, `AutoModelForQuestionAnswering`, `AutoModelForTableQuestionAnswering`, `AutoModelForVisualQuestionAnswering`, `AutoModelForDocumentQuestionAnswering`, `AutoModelForTokenClassification`, `AutoModelForMultipleChoice`, `AutoModelForNextSentencePrediction`, `AutoModelForImageClassification`, `AutoModelForZeroShotImageClassification`, `AutoModelForImageSegmentation`, `AutoModelForSemanticSegmentation`, `AutoModelForUniversalSegmentation`, `AutoModelForInstanceSegmentation`, `AutoModelForObjectDetection`, `AutoModelForZeroShotObjectDetection`, `AutoModelForDepthEstimation`, `AutoModelForTextRecognition`, `AutoModelForTableRecognition`, `AutoModelForVideoClassification`, `AutoModelForImageTextToText`, `AutoModelForMultimodalLM`, `AutoModelForAudioClassification`, `AutoModelForCTC`, `AutoModelForTDT`, `AutoModelForSpeechSeq2Seq`, `AutoModelForAudioFrameClassification`, `AutoModelForAudioXVector`, `AutoModelForTextToSpectrogram`, `AutoModelForTextToWaveform`, `AutoModelForTimeSeriesPrediction`, `AutoModelForMaskedImageModeling`, `AutoModelForAudioTokenization`, `AutoBackbone`.

Plus the processor-side Autos: `AutoTokenizer`, `AutoImageProcessor`, `AutoFeatureExtractor`, `AutoProcessor`, `AutoVideoProcessor`.

### 4.4 The `set_device(model, device)` contract

This is the only meaningful API addition over HF. Its precise contract:

1. Walks the model graph and calls `to_device(device)` on every `TTNNModule`.
2. For each `TTNNModule`, reads the `@run_on_devices(...)` declaration on its `forward`/`call` method.
3. If the active device's `DeviceArch` is **not** in the declared set, the module is **swapped back to its `_fallback_torch_layer`** and a warning is logged: `Running <module_name> on CPU; not supported on <device_arch>`.
4. After the walk, calls `preprocess_weights()` and `move_weights_to_device()` on every remaining `TTNNModule`. **TBD:** today the test scripts do this loop explicitly (`for v in modules: v.preprocess_weights(); v.move_weights_to_device()`); decide whether `set_device` should subsume that loop or expose a `preprocess=True` kwarg. Leaning toward subsume so the user only ever writes one line.
5. Setting `_set_device_called = True` on the root model. Any forward call on a model where this flag is False **raises**: `RuntimeError: set_device() must be called before model invocation.`

---

## 5. Dependency on `TT_CCL` — Decision

The only `tt-metal` Python dependency in the current code is:

```python
# tt_symbiote/core/run_config.py:22
from models.tt_transformers.tt.ccl import TT_CCL

# tt_symbiote/utils/groot_utils.py:1440 — same import, optional
```

`TT_CCL` is a 495-line CCL (collective communications) helper in `tt-metal/models/tt_transformers/tt/ccl.py`. Its only non-`ttnn` dependency is `determine_device_name` from `tt_transformers.tt.model_config`.

**Options considered:**

| Option | Verdict |
|---|---|
| (a) Drop `TT_CCL` from `tt_symbiote`. | Not viable: it's used in `DistributedConfig.__post_init__` to construct the multi-device CCL manager. Multi-device support (T3K, TG, P150x4) is a hard requirement. |
| (b) `from tt_transformers... import TT_CCL` at runtime. | Not viable: `tt-metal/models/tt_transformers/` is **not** a pip-installable package. It's a path inside the `tt-metal` source tree, which `pip install tt_symbiote` will not have. |
| (c) **Vendor `TT_CCL` into `tt_symbiote/core/ccl.py`** (with `determine_device_name` vendored alongside). | **Chosen.** 495 lines, low churn, self-contained, breaks the cross-repo coupling. |

**Action:** at migration time, copy `tt-metal/models/tt_transformers/tt/ccl.py` → `src/tt_symbiote/core/ccl.py` and the device-name helper to `src/tt_symbiote/core/model_config.py`. Add a one-line note in each: `# Vendored from tt-metal commit <sha> at <date>. Re-sync manually if upstream changes materially.`

**TBD during implementation:** whether to keep a tiny sync script (`scripts/sync_ttnn_helpers.py`) that re-pulls `ccl.py` from a pinned `tt-metal` commit on demand. Probably yes but deferred until the second time we have to sync.

---

## 6. Dispatcher removal

What gets deleted:

- `core/dispatchers/` (entire folder: `default_dispatcher.py`, `debug_dispatcher.py`, `cpu_dispatcher.py`, `dispatcher_config.py`)
- `core/dispatcher.py` and `core/torch_dispatcher.py` indirection where used purely for op routing
- The `TT_SYMBIOTE_DISPATCHER` env var
- The `dispatcher_config.py` plumbing currently consulted from `run_config.py` (lines 199, 220, 532, 656, 707, 749, 799, 1367)

What replaces it:

- A simple `try/except` in `TTNNModule.call` that, on TTNN failure, calls the stored `_fallback_torch_layer` and emits a warning.
- The `@run_on_devices(DeviceArch.X)` declaration is the **proactive** version of the same fallback (decide at `set_device` time rather than first-call time).

What is retained:

- The `run modes` env var `TT_SYMBIOTE_RUN_MODE` and its four modes: `NORMAL`, `DPL`, `CPU`, `TRACED`. These are user-facing debugging primitives, not a dispatch hierarchy.

**Affected files (must be edited during Phase 3):** `core/run_config.py`, `core/module.py`, and any test that explicitly sets `TT_SYMBIOTE_DISPATCHER`.

**TBD during implementation:** confirm no model recipe relies on the *debug* dispatcher's verbose op logging. If one does, replace with a logger configured by `TT_SYMBIOTE_LOG_LEVEL=DEBUG`.

---

## 7. Model porting plan

### 7.1 Priority (locked from the team call)

| Tier | Model | Source today | Status | Notes |
|---|---|---|---|---|
| P0 | Ling-mini-2.0 (`bailing_moe_v2`) | `tests/test_ling_mini_2_0.py`, `modules/decoder_layer.py`, `models/bailing_moe_v2.py` | ✅ Phase 5 (full TTNN, T3K) | Reference port. Defines the recipe pattern. |
| P1 | GLM-4 family (`glm4_moe`) | `tests/test_glm.py`, `test_glm_4_7.py`, `test_glm_5.py`, `test_glm_flash.py`, `modules/moe.py::Glm4MoeConfig` | ⏳ deferred | Shares decoder shape with Ling. Verify shared-module factorization is correct. |
| P1 | Gemma4 | `tests/test_gemma4.py`, `models/gemma4_text.py`, `modules/gemma4_*` | ✅ Phase 7 (CPU-first) + Phase 8 Wave A (5 on-device wrappers, 4 variants verified — see [`docs/cpu_vs_device_coverage.md`](docs/cpu_vs_device_coverage.md)) | Recipe + budget/MoE gate landed; full text decoder still on CPU pending Wave A+1. |
| P2 | Qwen3-VL (dense) | `transformers/src/transformers/models/qwen3_vl/` | ✅ Phase 7 follow-up (skill-driven CPU-first port, 2B verified) + Phase 8 Wave B (4 on-device wrappers) | Landed via the [`port-hf-model-to-tt-symbiote`](.cursor/skills/port-hf-model-to-tt-symbiote/SKILL.md) skill. |
| P2 | Qwen3 family (`qwen3_moe`, `qwen3_coder_next`) | `tests/test_qwen3_5_35b_a3b.py`, `test_qwen3_coder_next.py`, `modules/qwen_*` | ⏳ deferred | |
| P2 | GPT-OSS, Olmo3, LLaMA, Molmo2, Qwen-Omni | corresponding `tests/test_*.py` | ⏳ deferred | |
| P3 | ResNet | `tests/test_resnet.py` | ✅ Phase 6 (full TTNN on N150, 5 variants under one recipe; resnet-50 hardware-verified) | First vision reference port. |
| P3 | Vision/audio: ViT, OWL-ViT, Whisper3, SpeechT5, HunyuanVideo, OpenVLA, Gr00t | corresponding `tests/test_*.py` | ⏳ deferred | Wire up the right `Auto*` parent (`AutoModelForImageClassification`, `AutoModelForSpeechSeq2Seq`, etc.) |
| Drop | YuNet | `tests/test_yunet.py` | dropped Phase 2.6 | Not on HF Hub. |
| Drop | `dots.ocr` | `tests/test_deepseek_ocr.py` | dropped Phase 2.6 | Hit HW limits per team. |
| Drop | `tests/test_training.py` | n/a | dropped Phase 2.6 | Training is out of scope for v0.1. |

### 7.2 Per-model port checklist

1. Create `src/tt_symbiote/models/<model>/__init__.py` and `modeling_<model>.py`.
2. Move all model-specific TTNN classes from today's `modules/<model>_*.py` and `models/<model>.py` into `modeling_<model>.py`.
3. Register the recipe via `@register_recipe(hf_class_name="…")`.
4. If the model's HF class name maps to multiple Auto* tasks (e.g. Gemma4 is `Gemma4ForConditionalGeneration` and routed through `AutoModelForImageTextToText`), add the appropriate entry in `auto/auto_mappings.py`.
5. Create `tests/capabilities/<model>/test_modeling_<model>.py` with one smoke test: load → `set_device` → `generate(max_new_tokens=8)`.
6. Add the model to `docs/supported_models.md`.
7. CI must pass on the target device. If the test cannot be made to pass within the porting effort, the model is **dropped from this release branch** (per the call: "whatever can run, add it; if not, remove it").

### 7.3 The "Claude skill" porting workflow

Create `.cursor/skills/port-hf-model/SKILL.md` inside the new repo. The skill's contract:

- **Input:** HF model repo id (e.g. `inclusionAI/Ling-mini-3.0`), target `DeviceArch` (e.g. `T3K`), optional reference: an already-ported sibling model in `tt_symbiote/models/`.
- **Stages:**
  1. Load the HF model in a sandbox, dump the type tree, identify the decoder/attention/MoE/embedding/norm classes.
  2. Generate a skeleton `modeling_<model>.py` next to its HF analog, importing generic blocks from `tt_symbiote.modules.*` and only overriding what's model-specific.
  3. Generate the `@register_recipe` decorator with the module dict (analog of the three-pass dict in today's `test_gemma4.py` / `test_ling_mini_2_0.py`).
  4. Generate a thin smoke test in `tests/capabilities/<model>/`.
  5. Run the smoke test; iterate on the recipe until it passes or the model is dropped.
- **Out of scope for the skill (human required):** custom TTNN kernels, paged-attention shape derivations, novel sharding configs. The skill must explicitly bail and report when one of these is needed.

**TBD during implementation:** the exact prompt and tool-use sequence in `SKILL.md`. We will draft this after Ling-mini-2.0 is ported and we have a known-good reference port to point the skill at.

---

## 8. Test strategy

Mirrors `transformers/tests/`:

- `tests/capabilities/` — exercises each `integrations/` building block in isolation against a synthetic input. Catches regressions in TTNN ops without needing a real model checkpoint. Today's flat `tests/test_attention.py`, `test_moe.py`, `test_conv.py`, `test_rope.py`, `test_dpl.py` become these.
- `tests/capabilities/<model>/test_modeling_<model>.py` — per-model smoke test. Load + register + `set_device` + `generate(max_new_tokens=8)`. Asserts output is non-empty. **Not** a numerical equivalence test — numerical PCC checks live in DPL run mode.
- `tests/conftest.py` — pytest fixtures for `mesh_device`, `device_params`. Carried over verbatim from today's `tests/conftest.py`.

CI matrix (placeholder, **TBD**): we will run capability tests on every push, but smoke tests only on a nightly run, gated by which device is connected to the runner. The exact runner topology (which models tested on which device) will be defined during Phase 9 as runners come online.

---

## 9. Phased delivery

Phases are unit-of-work boundaries, not release boundaries. **No tags are created until Phase 5 succeeds** — the first tag is `v0.0.0` after Ling-mini-2.0 passes its smoke test on the new repo (see §13). Subsequent tags follow real, user-visible milestones, not internal checkpoints.

### Phase 0 — Decisions locked (this document)

Done when this proposal is approved.

### Phase 1 — Repo skeleton + tooling

- Create the directory tree from §3 with empty/placeholder files.
- Write `pyproject.toml` (deps: `transformers==5.9.0`, `torch`, `ttnn` — sourcing of `ttnn` is **TBD**: editable install pointing at a `tt-metal` build, or wait for a `ttnn` wheel).
- Add `Makefile` targets: `make install`, `make test`, `make smoke MODEL=ling_mini_2_0`.
- Set up `pre-commit` (black, isort, ruff) matching the rest of the TT codebase.
- Create initial CI workflow that at least lints + runs `pytest tests/capabilities -k synthetic`.

### Phase 2 — Mechanical migration

- Copy current `tt-metal/models/experimental/tt_symbiote/{core,modules,models,utils,tests,demos,docs}` into the new layout.
- Run a scripted codemod to rewrite imports `models.experimental.tt_symbiote.X` → `tt_symbiote.X`. Search hits today: 80+ files.
- Vendor `TT_CCL` and `determine_device_name` per §5.
- Split `modules/*` into `integrations/*` (generic) vs `models/<model>/modeling_<model>.py` (model-specific). Concretely:
  - `modules/gemma4_attention.py`, `modules/gemma4_mlp.py`, `modules/gemma4_modules.py`, `models/gemma4_text.py` → `models/gemma4/modeling_gemma4.py`
  - `modules/qwen_attention.py`, `modules/qwen_moe.py` → `models/qwen3_moe/modeling_qwen3_moe.py`
  - `models/bailing_moe_v2.py`, `modules/decoder_layer.py` → `models/bailing_moe_v2/modeling_bailing_moe_v2.py`
  - Generic remainder of `modules/{linear,normalization,activation,embedding,rope,attention,moe,conv,tensor}.py` → `integrations/ttnn_*.py`
- Move `demos/` → `examples/`.

### Phase 3 — Remove dispatcher

Per §6.

### Phase 4 — Public API

- Implement `auto/auto_factory.py` + the 43 `Auto*` classes.
- Implement `register_recipe` and `TT_MODEL_REGISTRY`.
- Implement the new `set_device` contract (including the hard-error and `@run_on_devices` enforcement).
- Implement `register_modules` (rename of `register_module_replacement_dict`; keep the old name as a deprecation alias for one release).

### Phase 5 — Ling-mini-2.0 reference port

The first end-to-end model port. Validates the recipe shape, the `Auto*` plumbing, and the test layout. **When the Ling-mini-2.0 smoke test passes, tag `v0.0.0`** — this is the first tag on the new repo and marks the point at which one real model demonstrably works through the new API.

### Phase 6 — ResNet vision reference port (replaced "GLM + Gemma4" per redirect)

Originally scoped to "GLM + Gemma4 LLM ports"; redirected mid-flight
to a vision reference port to exercise the recipe contract along a
second axis (NHWC convs, no KV cache, image input). Five Microsoft
ResNet variants share a single recipe; resnet-50 is hardware-verified
end-to-end on N150 (full TTNN). See
[`docs/migration_notes.md`](docs/migration_notes.md) Phase 6.

### Phase 7 — Gemma-4 VLM (CPU-first) + porting skill + Qwen3-VL (CPU-first)

Phase 7 turned into three landings:

1. **Gemma-4 VLM port** — CPU-first commit that exercises the full
   image-text-to-text demo on N150 against `gemma-4-E2B-it` and ships
   the four-class compatibility tracker (`tt_implemented` /
   `cpu_fallback` / `out_of_scope`; `host_glue` added later in
   Phase 8) that downstream TTNN commits plug into.
2. **`port-hf-model-to-tt-symbiote` Cursor skill** at
   `.cursor/skills/port-hf-model-to-tt-symbiote/` — encodes the
   templated workflow used by Gemma-4 and exercises it by porting a
   new model (Qwen3-VL).
3. **Qwen3-VL-2B-Instruct port** — produced end-to-end by the skill,
   first model landed with zero design decisions during execution.

### Phase 8 — Gemma-4 + Qwen3-VL TTNN push (high-value-first)

The first wave of on-device wrappers for the two VLMs, focused on
"structurally simple high-value compute" via existing TTNN
integrations. Wave A (Gemma-4) ships 5 swaps (`Gemma4RMSNorm`,
`Gemma4TextScaledWordEmbedding`, `Gemma4TextMLP`, `Gemma4VisionMLP`,
`Gemma4MultimodalEmbedder`) plus a budget + MoE gate that forces
oversize or MoE Gemma-4 variants to whole-model CPU execution
with a clean compatibility report. Wave B (Qwen3-VL) ships 4 swaps
(`Qwen3VLTextRMSNorm`, `Qwen3VLTextMLP`, `Qwen3VLVisionMLP`,
`Qwen3VLVisionPatchMerger`). All four Gemma-4 variants verified
end-to-end (two via Wave A swaps, two via the gate). See
[`docs/migration_notes.md`](docs/migration_notes.md) Phase 8 sections.

### Phase 9 — Documentation + first release (was Phase 8 in original draft)

- Fill in `docs/architecture.md`, `docs/porting_a_new_model.md`, `docs/run_modes.md`.
- Update `README.md` with the HF-style quick start (done).
- Tag `v0.1.0` — first *external* release. The branch becomes `transformers-5.9` (the long-lived branch for that HF version per §10). `v0.0.0` (Ling-mini-2.0 working) and any intermediate tags from Phases 6–8 are internal milestones; `v0.1.0` is the first version we'd point an outside user at.

### Phase 10 — CI + release automation (rolling; was Phase 9 in original draft)

- Wire up self-hosted runners for N150, T3K, P150 nightly smoke runs.
- Add a `release-please`-style automation for version bumps.

---

## 10. Versioning & branching strategy

- `main` always tracks the latest stable `transformers` release we support.
- For each supported `transformers` version, a long-lived branch: `transformers-5.9`, `transformers-5.10`, etc.
- Bumping `transformers` is a **branch event**, not a PR into `main`. The bump branch may drop models that don't port; those models can be re-added later via the Claude porting skill.
- `tt_symbiote` semver: major bumps reserved for breaking API changes (rare, given we mirror HF); minor bumps for new model coverage; patch bumps for bugfixes.

---

## 11. Open questions (to resolve during implementation, not now)

| # | Question | When it must be answered |
|---|---|---|
| Q1 | Should `integrations/` filenames carry the `ttnn_` prefix? | Phase 2 |
| Q2 | Should the recipe expose a single dict (one-pass) or a list of dicts (multi-pass replacement)? | Phase 5 (Ling port) |
| Q3 | Should `set_device` subsume `preprocess_weights` + `move_weights_to_device`, or stay strictly orthogonal? | Phase 4 |
| Q4 | How is `ttnn` sourced in `pyproject.toml` — editable path to `tt-metal`, internal wheel index, or PyPI? | Phase 1 |
| Q5 | Do we keep the verbose-logging behavior previously provided by `debug_dispatcher`, and if so via `TT_SYMBIOTE_LOG_LEVEL`? | Phase 3 |
| Q6 | CI runner topology: which models run on which device, on what cadence? | Phase 9 |
| Q7 | Auto-sync script for vendored `ccl.py` from upstream `tt-metal`? | After first re-sync need |
| Q8 | Does `register_modules` need to support nested/recursive recipes (a model whose recipe is a composition of two sub-recipes)? | First time a model shares >50% of its decoder with another |
| Q9 | How does `kv_cache` get created? Today it's per-test boilerplate (`TTNNPagedAttentionKVCache(...)`). Should each `Recipe` expose `make_kv_cache(model)`? | Phase 5 |
| Q10 | Multi-version support in *one* checkout: do we need a `tt_symbiote[transformers-5.9]` extras-style mechanism, or strictly one version per branch? | First time a user asks for it |

---

## 12. Glossary

- **TTNN** — Tenstorrent Neural Network library (Python bindings to tt-metal kernels).
- **CCL** — Collective communications (all-reduce, all-gather, etc.) over the mesh fabric.
- **DPL** — Debug Per Layer; a run mode that runs both TTNN and PyTorch and compares with PCC.
- **PCC** — Pearson correlation coefficient; the standard numerical-equivalence metric in TT.
- **DeviceArch** — Enum tag (`N150`, `T3K`, `P150`, …) used by `@run_on_devices` to declare module support.
- **Recipe** — A `tt_symbiote` object that, given a loaded HF model instance, returns the PyTorch→TTNN module dict for `register_modules`.

---

## 13. Concrete starting workflow

All new development happens in the new `tt_symbiote` repo. The `tt-metal` repo is read-only source material.

```bash
# 1. In tt-metal: check out the agreed-upon source version (read-only).
#    Source we are migrating from = models/experimental/tt_symbiote/
#    on the `alnah005/tt_symbiote_v2` branch (NOT main, which differs
#    significantly).
cd ~/tt-metal
git fetch origin alnah005/tt_symbiote_v2
git checkout alnah005/tt_symbiote_v2

# 2. In the new tt_symbiote repo: all work happens on aroberge/bootstrap.
#    This branch carries Phases 1-5: skeleton → migration → dispatcher
#    removal → public API → Ling-mini-2.0 reference port.
cd /home/aroberge/tt_symbiote
git checkout aroberge/bootstrap

# 3. Phase 1 lands the repo skeleton (§3) as the first commit on this branch.
#    Phase 2 copies files from tt-metal/models/experimental/tt_symbiote/
#    into src/tt_symbiote/, rewriting imports along the way.
#    No git history is carried across — clean repo, fresh commits.
#
#    NO TAGS are created during Phases 1-4. A tag at the skeleton stage
#    would imply something runs, which is not true until a model ports
#    successfully.

# 4. The FIRST tag is v0.0.0, created only after Phase 5 completes —
#    i.e. once Ling-mini-2.0's smoke test passes end-to-end on the new
#    repo (load → register_modules → set_device → generate). At that
#    point the repo is genuinely "v0" in the sense that one real model
#    works through the new API.
git tag v0.0.0    # ONLY after Ling-mini-2.0 passes
```

Once this proposal is approved, Phase 1 starts on `aroberge/bootstrap`.
