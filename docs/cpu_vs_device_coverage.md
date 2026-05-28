# CPU vs. device coverage per demo

For every verified or structurally-supported `examples/e2e/` script,
this page documents *where every model submodule runs* — on the
Tenstorrent device (N150 / N300 / T3K) or on the host CPU. The data is
sourced from the per-script `*_coverage.json` artefacts that each demo
writes next to itself; this document just aggregates them into a
human-readable table.

## Pipeline

```mermaid
graph LR
    Recipe[Recipe lists in<br/>tt_symbiote/models/&lt;name&gt;/]
    Recipe -->|design-time intent| Report["compatibility.report(model)"]
    Runtime[run_config TTNN forward<br/>fallback hook]
    Runtime -->|runtime observation| Report
    Report --> JSON["examples/e2e/&lt;family&gt;/<br/>run_&lt;variant&gt;_coverage.json"]
    JSON --> Doc[This page]
```

Each recipe in [`tt_symbiote.models.*`](../src/tt_symbiote/models/)
declares four class-name lists (Phase 8 added `host_glue` alongside
the original three):

- `tt_implemented` — HF classes the recipe ships a TTNN wrapper for
  (in `build_module_dict`). These run on Tenstorrent silicon.
- `cpu_fallback` — HF classes exercised by the demo that stay as
  PyTorch *for now* (no TTNN equivalent yet). These run on the host
  CPU and are the actionable backlog.
- `host_glue` — HF classes that are intentionally host-only by policy
  (orchestration, mask building, scatter fusion, output dataclasses,
  index walks). These have no FLOPs to accelerate — flagging them
  separately from `cpu_fallback` keeps the backlog focused on
  compute, not glue.
- `out_of_scope` — HF classes that exist in the upstream modeling
  file but aren't touched by the documented demo path (audio towers
  on image-only demos, alternative heads).

Where the recipe also enables it (Phase 7+), a runtime hook in
[`tt_symbiote.core.run_config`](../src/tt_symbiote/core/run_config.py)
records every `TTNNModule.forward` exception that fell back to PyTorch.
That ledger is surfaced under `runtime_observed.unexpected` — *any
non-empty value there is a regression*: it means a TTNN wrapper failed
mid-run and the recipe didn't declare it as a known fallback.

## How to regenerate this page

Re-run the demo whose row you want refreshed. The JSON next to the
script updates atomically; copy the relevant counts and class lists
into the table below. The JSON files are checked in so the table never
drifts from reality.

## Coverage by model

### Causal LM

#### Ling-mini-2.0 — T3K (1×8), full TTNN

Source: [`examples/e2e/run_ling_mini_2_0.py`](../examples/e2e/run_ling_mini_2_0.py)
+ [`examples/e2e/run_ling_mini_2_0_coverage.json`](../examples/e2e/run_ling_mini_2_0_coverage.json)

The Ling recipe predates the Phase 7 design-time list convention, so
its `tt_implemented` / `cpu_fallback` / `out_of_scope` lists are not
yet populated; the runtime ledger (when re-run on T3K) is the
authoritative source. From the Phase 5 acceptance: every text decoder
submodule (RMSNorm, attention, MoE experts + router, decoder layer,
top-level model) runs on **T3K device**; tokenizer + the `generate`
scaffolding run on CPU.

Follow-up: backfill the recipe's three design-time lists, mirroring
the Gemma-4 / Qwen3-VL pattern.

### Image classification

#### ResNet-50 — N150 (1×1), full TTNN

Source: [`examples/e2e/resnet/run_resnet50.py`](../examples/e2e/resnet/run_resnet50.py)
+ [`examples/e2e/resnet/run_resnet50_coverage.json`](../examples/e2e/resnet/run_resnet50_coverage.json)

Same caveat as Ling — the ResNet recipe predates the design-time list
convention; counts in the JSON are zero. Runtime: zero unexpected
fallbacks on the verified run.

From the recipe source ([`modeling_resnet.py`](../src/tt_symbiote/models/resnet/modeling_resnet.py)):

| Module | Where it runs | Notes |
|---|---|---|
| `TTNNResNetEmbeddings` (stem) | **N150** | 7×7 stride-2 conv, NCHW → NHWC permute |
| `TTNNResNetConvLayer` | **N150** | fused conv + BN + activation |
| `TTNNResNetShortCut` | **N150** | identity / 1×1 projection |
| `TTNNResNetBasicLayer` (resnet-18/34) | **N150** | two-conv block |
| `TTNNResNetBottleNeckLayer` (resnet-50/101/152) | **N150** | three-conv block |
| `ResNetEncoder` | **N150** (walks children) | unchanged HF wrapper |
| `nn.AdaptiveAvgPool2d` + `nn.Linear` (classifier) | **CPU** | post NHWC → NCHW boundary |

The four sibling variants (`resnet-18`, `-34`, `-101`, `-152`) share
the same recipe; their coverage JSONs are stubs until each runs on
hardware.

### Image-text-to-text (VLM)

Phase 8 Wave A moved the structurally simple Gemma-4 modules
(RMSNorm, scaled word embedding, text MLP, vision MLP, multimodal
embedder) from `cpu_fallback` to `tt_implemented`; the bespoke pieces
(KV-shared text attention, dual RoPE, PLE, 2-D vision RoPE,
position-aware pooler) stay declared `cpu_fallback`. Top-level fusion
(`Gemma4Model`, `Gemma4ForConditionalGeneration`) is `host_glue`. The
four Qwen3-VL variants remain CPU-first until Wave B lands.

#### Gemma-4 E2B-it — N150 (1×1), Wave A (✅ verified)

Source: [`examples/e2e/gemma4/run_gemma4_e2b.py`](../examples/e2e/gemma4/run_gemma4_e2b.py)
+ [`examples/e2e/gemma4/run_gemma4_e2b_coverage.json`](../examples/e2e/gemma4/run_gemma4_e2b_coverage.json)

Counts (design + runtime): **5 TT / 14 CPU / 2 host_glue / 14 OOS**.
`runtime_observed.unexpected == []`.

| Stage | Component class | Where | Hardware |
|---|---|---|---|
| TTNN runtime | `ttnn.set_fabric_config` + `open_mesh_device` | N150 (mgmt only) | device init |
| Loader | `AutoProcessor.from_pretrained` | CPU | — |
| Loader | `AutoModelForImageTextToText.from_pretrained` | CPU (host RAM) | — |
| Walker | `tt_symbiote.set_device` | CPU + N150 handle | swaps 5 classes |
| Tokenisation | `processor.apply_chat_template` | CPU | — |
| Forward (Wave A TTNN) | `Gemma4RMSNorm`, `Gemma4TextScaledWordEmbedding`, `Gemma4TextMLP`, `Gemma4VisionMLP`, `Gemma4MultimodalEmbedder` | **N150** | matmuls + GELU |
| Forward (shared) | `Gemma4ClippableLinear` | CPU (thin clamp) | inner Linear may be on-device when its parent MLP is wrapped |
| Forward (vision) | `Gemma4VisionPatchEmbedder`, `Gemma4VisionRotaryEmbedding` (2-D RoPE), `Gemma4VisionAttention`, `Gemma4VisionEncoderLayer`, `Gemma4VisionEncoder`, `Gemma4VisionPooler`, `Gemma4VisionModel` | CPU | bespoke 2-D RoPE not yet ported |
| Forward (text) | `Gemma4TextRotaryEmbedding` (dual table), `Gemma4TextAttention` (KV-share), `Gemma4TextExperts`*, `Gemma4TextRouter`*, `Gemma4TextDecoderLayer`, `Gemma4TextModel` (PLE) | CPU | * MoE pair unused on E2B |
| Forward (host_glue) | `Gemma4Model`, `Gemma4ForConditionalGeneration` | CPU (policy) | masked_scatter + mask build + softcap |
| De-tokenisation | `processor.batch_decode` | CPU | — |
| Out of scope | 9× `Gemma4Audio*`, `Gemma4ForCausalLM`, 4× output dataclass | — | — |

#### Gemma-4 E4B-it — N150 (1×1), Wave A (⏳ structurally supported)

Source: [`run_gemma4_e4b.py`](../examples/e2e/gemma4/run_gemma4_e4b.py)
+ [`coverage stub`](../examples/e2e/gemma4/run_gemma4_e4b_coverage.json).
Same class layout as E2B (same recipe). Design-time counts: 5 TT /
14 CPU / 2 host_glue / 14 OOS.

#### Gemma-4 31B-it — T3K (1×8), Wave A (⏳ structurally supported)

Source: [`run_gemma4_31b.py`](../examples/e2e/gemma4/run_gemma4_31b.py)
+ [`coverage stub`](../examples/e2e/gemma4/run_gemma4_31b_coverage.json).
Same class layout as E2B; mesh shape `(1, 8)`.

#### Gemma-4 26B-A4B-it — T3K (1×8), MoE, Wave A (⏳ structurally supported)

Source: [`run_gemma4_26b_a4b.py`](../examples/e2e/gemma4/run_gemma4_26b_a4b.py)
+ [`coverage stub`](../examples/e2e/gemma4/run_gemma4_26b_a4b_coverage.json).
Same class layout; the `Gemma4TextExperts` / `Gemma4TextRouter` MoE
pair *is* exercised here (unlike on E2B).

#### Qwen3-VL-2B-Instruct — N150 (1×1), Wave B (✅ verified)

Source: [`examples/e2e/qwen3_vl/run_qwen3_vl_2b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_2b.py)
+ [`examples/e2e/qwen3_vl/run_qwen3_vl_2b_coverage.json`](../examples/e2e/qwen3_vl/run_qwen3_vl_2b_coverage.json)

Counts: **4 TT / 9 CPU / 3 host_glue / 3 OOS**.
`runtime_observed.unexpected == []`.

| Stage | Component class | Where | Hardware |
|---|---|---|---|
| TTNN runtime | `set_fabric_config(DISABLED)` + `open_mesh_device((1,1))` | N150 (mgmt) | init/alloc |
| Loader | `AutoProcessor.from_pretrained` (→ `Qwen3VLProcessor`) | CPU | — |
| Loader | `AutoModelForImageTextToText.from_pretrained` | CPU | — |
| Walker | `tt_symbiote.set_device` | CPU + N150 handle | swaps 4 classes |
| Tokenisation | `processor.apply_chat_template` | CPU | — |
| Forward (Wave B TTNN) | `Qwen3VLTextRMSNorm`, `Qwen3VLTextMLP`, `Qwen3VLVisionMLP`, `Qwen3VLVisionPatchMerger` | **N150** | matmuls + SiLU/GELU |
| Forward (vision) | `Qwen3VLVisionPatchEmbed` (Conv3d), `Qwen3VLVisionRotaryEmbedding` (2-D), `Qwen3VLVisionAttention` (varlen-packed), `Qwen3VLVisionBlock`, `Qwen3VLVisionModel` | CPU | bespoke 2-D RoPE / varlen SDPA not yet ported |
| Forward (text) | `Qwen3VLTextRotaryEmbedding` (M-RoPE), `Qwen3VLTextAttention` (Q/K head-norms), `Qwen3VLTextDecoderLayer`, `Qwen3VLTextModel` (DeepStack injection) | CPU | bespoke M-RoPE + Q/K head-norms not yet ported |
| Forward (host_glue) | `Qwen3VLPreTrainedModel`, `Qwen3VLModel`, `Qwen3VLForConditionalGeneration` | CPU (policy) | masked_scatter + cu_seqlens build + DeepStack dispatch |
| De-tokenisation | `processor.batch_decode` | CPU | — |
| Out of scope | `BaseModelOutputWithDeepstackFeatures`, `Qwen3VLModelOutputWithPast`, `Qwen3VLCausalLMOutputWithPast` | — | — |

Qwen3-VL has no audio tower, which is why the OOS count is 3 (just
the output dataclasses) vs. Gemma-4's 14.

#### Qwen3-VL-4B / 8B / 32B Instruct — Wave B (⏳ structurally supported)

Same class layout as 2B (same recipe). Per-variant sources:
[`run_qwen3_vl_4b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_4b.py)
([stub](../examples/e2e/qwen3_vl/run_qwen3_vl_4b_coverage.json)),
[`run_qwen3_vl_8b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_8b.py)
([stub](../examples/e2e/qwen3_vl/run_qwen3_vl_8b_coverage.json)),
[`run_qwen3_vl_32b.py`](../examples/e2e/qwen3_vl/run_qwen3_vl_32b.py)
([stub](../examples/e2e/qwen3_vl/run_qwen3_vl_32b_coverage.json)).

## Summary table

Aggregating the JSON files into a single overview:

| Demo | Hardware | Status | TT classes | CPU classes | host_glue | OOS classes | Runtime unexpected |
|---|---|---|---|---|---|---|---|
| `run_ling_mini_2_0.py` | T3K (1×8) | ✅ verified (Phase 5) | not yet declared | not yet declared | not yet declared | not yet declared | clean at Phase 5 acceptance |
| `resnet/run_resnet18.py` | N150 (1×1) | ⏳ stub | 0 | 0 | 0 | 0 | n/a |
| `resnet/run_resnet34.py` | N150 (1×1) | ⏳ stub | 0 | 0 | 0 | 0 | n/a |
| `resnet/run_resnet50.py` | N150 (1×1) | ✅ verified (re-run in reorg) | 0 (lists not declared) | 0 (lists not declared) | 0 (lists not declared) | 0 (lists not declared) | 0 |
| `resnet/run_resnet101.py` | N150 (1×1) | ⏳ stub | 0 | 0 | 0 | 0 | n/a |
| `resnet/run_resnet152.py` | N150 (1×1) | ⏳ stub | 0 | 0 | 0 | 0 | n/a |
| `gemma4/run_gemma4_e2b.py` | N150 (1×1) | ✅ verified (Phase 8 Wave A) | **5** | 14 | 2 | 14 | 0 |
| `gemma4/run_gemma4_e4b.py` | N150 (1×1) | ⏳ stub | 5 | 14 | 2 | 14 | n/a |
| `gemma4/run_gemma4_31b.py` | T3K (1×8) | ⏳ stub | 5 | 14 | 2 | 14 | n/a |
| `gemma4/run_gemma4_26b_a4b.py` | T3K (1×8) | ⏳ stub | 5 | 14 | 2 | 14 | n/a |
| `qwen3_vl/run_qwen3_vl_2b.py` | N150 (1×1) | ✅ verified (Phase 8 Wave B) | **4** | 9 | 3 | 3 | 0 |
| `qwen3_vl/run_qwen3_vl_4b.py` | N150 (1×1) | ⏳ stub | 4 | 9 | 3 | 3 | n/a |
| `qwen3_vl/run_qwen3_vl_8b.py` | N150 (1×1) | ⏳ stub | 4 | 9 | 3 | 3 | n/a |
| `qwen3_vl/run_qwen3_vl_32b.py` | T3K (1×8) | ⏳ stub | 4 | 9 | 3 | 3 | n/a |

**Net Tenstorrent coverage today:** ResNet-50 (full TTNN on N150) and
Ling-mini-2.0 (full TTNN on T3K) remain the only models with the
attention/MoE/decoder loop on device. Gemma-4 E2B and Qwen3-VL-2B
now have *partial* on-device execution (5 and 4 simple compute classes
respectively; their attention + decoder loops still run on host
pending the bespoke text-attention / RoPE wrappers).

## Open follow-ups

- Backfill `tt_implemented` / `cpu_fallback` / `host_glue` /
  `out_of_scope` lists in the **Ling** and **ResNet** recipes so
  their JSON artefacts have populated design-time sections (currently
  empty for both because those recipes predate the list convention).
- Land Wave A+1 for Gemma-4: the bespoke text attention (KV-sharing
  + dual RoPE + per-head norms) and PLE. This is what unblocks moving
  the decoder loop to device.
- Land Wave B+1 for Qwen3-VL: the bespoke M-RoPE precompute,
  Q/K head-norm-aware attention, varlen-packed vision SDPA, and
  DeepStack-aware decoder loop.
- CI gate that fails if any committed `*_coverage.json` has
  `runtime_observed.unexpected != []`. Trivial follow-up; the file
  format is already JSON-friendly.
