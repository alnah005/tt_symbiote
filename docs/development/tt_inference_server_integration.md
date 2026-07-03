<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serving tt_symbiote models through tt-inference-server (vLLM backend)

This document is the design for making `tt_symbiote` models servable through
[`tt-inference-server`](https://github.com/tenstorrent/tt-inference-server) on
the vLLM backend, **the same way `tt-metal`'s `tt_transformers` models are
served today**, with each model pinned to its own tt-metal commit SHA.

It is a design/plan document — no code has been written yet. It is the output of
an end-to-end investigation of `tt-inference-server`, `tt-metal`, the
Tenstorrent `vllm` fork, and `tt_symbiote` itself.

---

## 1. Goal & non-goals

**Goal.** `pip install tt_symbiote` inside a tt-inference-server Docker image and
serve a `tt_symbiote` model (starting with `dots_ocr`) through the vLLM
OpenAI-compatible API, exactly like a `tt_transformers` model — with the model's
tt-metal commit (already recorded in `RUNTIME_PINS`) baked into that image.

**Non-goals (for the first milestone).**

- Continuous batching / multi-request block-table scheduling for every model.
  We adopt vLLM's scheduler, but the first model (`dots_ocr`) keeps its existing
  data-parallel batch semantics behind the adapter.
- Porting `gemma4` / `qwen3_vl` attention off-host (they are partial ports and
  not servable yet).
- Changing `tt_symbiote`'s public HuggingFace-shaped API
  (`from_pretrained` → `set_device` → `generate`). The vLLM adapter is
  **additive**.

---

## 2. How tt-inference-server serves `tt_transformers` today (the pattern to mirror)

### 2.1 Runtime path

```
run.py --docker-server
  └─ docker pull/run  model_spec.docker_image      (commit baked into image)
       └─ run_vllm_api_server.py  (in container)
            ├─ sets HF_MODEL / TT_CACHE_PATH / MESH_DEVICE
            ├─ register_tt_models(impl_id)          (TT{Arch} → tt-metal class)
            └─ runpy: vllm.entrypoints.openai.api_server
                 └─ vLLM TT plugin (vllm_tt_plugin)
                      TTPlatform → TTWorker → TTModelRunner → TTModelLoader
                        └─ Generator.prefill_forward / decode_forward
                             └─ ttnn ops on MeshDevice
```

Key files:

| Role | Path |
|---|---|
| Server entry (in container) | `tt-inference-server/vllm-tt-metal/src/run_vllm_api_server.py` |
| Model spec schema + loader | `tt-inference-server/workflows/model_spec.py` |
| Model catalog (per-model data) | `tt-inference-server/workflows/model_specs/{prod,dev}/*.yaml` |
| Parameterized build | `tt-inference-server/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile` |
| Build orchestration | `tt-inference-server/scripts/build_docker_images.py` |
| vLLM Generator adapter (tt-metal) | `tt-metal/models/tt_transformers/tt/generator_vllm.py` |
| Generator base | `tt-metal/models/tt_transformers/tt/generator.py` |
| vLLM TT plugin | `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/` |

### 2.2 The model contract vLLM expects

The Tenstorrent vLLM plugin **never runs vLLM's standard `model.forward()`**. It:

1. Prepends `"TT"` to `hf_config.architectures` in
   `TTPlatform.check_and_update_config()` (e.g. `LlamaForCausalLM` →
   `TTLlamaForCausalLM`).
2. Resolves that name via `ModelRegistry` to a tt-metal **`Generator` subclass**.
3. `TTModelLoader.load_model()` calls
   `model_class.initialize_vllm_model(hf_config, mesh_device, ...)` — bypassing
   HF weight loading.

So a model family must provide:

- A **`Generator` subclass** with:
  - `@classmethod initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel, optimizations)`
  - `prefill_forward(...)`
  - `decode_forward(...)`
  - `allocate_kv_cache(...)`
  - a `model_capabilities` dict (`supports_prefix_caching`, `supports_async_decode`, …)
- A **`TT{Arch}` → module:class** entry registered in the vLLM plugin
  (`register_tt_models()`), keyed by `"TT" + hf_config.architectures[0]`.
- A **model_spec catalog entry** (YAML) with `tt_metal_commit`, `vllm_commit`,
  device specs, and an `impl` id.

### 2.3 The per-model commit mechanism (and the correction)

There is **one parameterized Dockerfile**, but it builds **one image per unique
`(tt_metal_commit, vllm_commit)` pair**. Inside the builder stage:

```
git clone tt-metal && git checkout $TT_METAL_COMMIT_SHA_OR_TAG \
  && bash ./build_metal.sh && bash ./create_venv.sh
```

i.e. **ttnn is source-built from the pinned commit**. The commit is embedded in
the image tag (`…-src-release-…:{version}-{tt_metal[:12]}-{vllm[:12]}`). At
runtime `run_docker_server.py` only does `docker pull model_spec.docker_image` —
no checkout, no rebuild.

**Consequence (this is the crux for tt_symbiote):** "per-model tt-metal commit"
means **per-model Docker image**, *not* multiple commits in one process. This is
how tt-inference-server lives under the same "one ttnn per process" constraint
(per-model commits are metadata in `RUNTIME_PINS`, enforced by the runtime gate)
— it gives each commit its own container. `tt_symbiote`'s per-model
`RUNTIME_PINS[...]["tt_metal_commit"]`
maps **1:1** onto a model_spec `tt_metal_commit` → its own image.

---

## 3. The gap: `tt_symbiote` today vs. the Generator contract

`tt_symbiote` is a HuggingFace-mirror library, not a serving stack. There is **no
vLLM integration, no `Generator` class, and no server entry point**. Per-model
status:

| Model | Generation today | KV cache | prefill/decode split | Servable readiness |
|---|---|---|---|---|
| `dots_ocr` | bespoke `TTNNDotsOCRPipeline.generate()` | paged (pipeline-owned) | **yes** (explicit graphs) | **closest** — wrap the pipeline |
| `bailing_moe_v2` (Ling-mini) | HF `generate()` loop | `TTNNPagedAttentionKVCache` | yes (seq-len routing) | needs a Generator + traced decode (`reset_trace_state`) |
| `gemma4`, `qwen3_vl` | HF `generate()` | HF `DynamicCache` | no TTNN paged path | not servable (attention on host) |

What already lines up:

- `set_device(model, mesh)` accepts an **externally-opened mesh device** — exactly
  how `TTWorker.init_device()` hands a `ttnn.MeshDevice` to a model. No global
  singleton mesh inside `tt_symbiote`.
- Per-model `tt_metal_commit` is already first-class metadata in `RUNTIME_PINS`.

What is missing (the work):

- A vLLM-facing `Generator`-style adapter exposing the methods in §2.2.
- A `TT{Arch}` registry entry pointing at that adapter.
- A model_spec YAML entry with a `tt_symbiote` `impl`.
- A Docker build that installs `tt_symbiote` on top of the source-built ttnn.

---

## 4. Proposed architecture

```
vLLM Worker (TTWorker)                  tt_symbiote adapter (NEW)              tt_symbiote core (exists)
──────────────────────                  ─────────────────────────             ─────────────────────────
open_mesh_device()            ──────▶   initialize_vllm_model(...)    ──────▶  AutoModelForCausalLM.from_pretrained
                                           builds HF model + recipe             set_device(model, mesh)
get_kv_cache_spec / allocate  ──────▶   allocate_kv_cache(...)        ──────▶  recipe.make_kv_cache (pipeline owns paged KV)
schedule prefill batch        ──────▶   prefill_forward(...)          ──────▶  pipeline.prefill(...)
execute decode step           ──────▶   decode_forward(...)           ──────▶  pipeline.decode_step(...)
```

### 4.1 New `tt_symbiote` vLLM adapter package

A new **optional** subpackage, e.g. `src/tt_symbiote/serving/vllm/`, importable
only when vLLM + the plugin are present (guarded import; not a hard dependency).
For `dots_ocr` it wraps the existing `TTNNDotsOCRPipeline`:

```python
# src/tt_symbiote/serving/vllm/dots_ocr.py  (sketch)
class TTDotsOCRForConditionalGeneration(Generator):  # Generator from tt-metal
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                              max_seq_len, tt_data_parallel=1, optimizations=None):
        model = AutoModelForCausalLM.from_pretrained(hf_config._name_or_path)
        set_device(model, mesh_device)              # binds externally-opened mesh
        return cls(model._tt_pipeline, hf_config)   # pipeline built in make_kv_cache

    def allocate_kv_cache(self, *a, **k): ...        # delegate to pipeline's paged KV
    def prefill_forward(self, *a, **k):  ...         # delegate to pipeline.prefill
    def decode_forward(self, *a, **k):   ...         # delegate to pipeline.decode_step

    model_capabilities = {"supports_prefix_caching": False,
                          "supports_async_decode": False}
```

The adapter base class (`Generator`) comes from tt-metal at runtime
(`models.tt_transformers.tt.generator`), so the adapter is only importable inside
a tt-inference-server image — never a `tt_symbiote` install-time dependency.

### 4.2 vLLM plugin registration

Register `TTDotsOCRForCausalLM` → the adapter. Two options:

- **Preferred (no vLLM fork change):** add registration in
  `run_vllm_api_server.register_tt_models()` when `impl_id == "tt_symbiote"`,
  pointing at `tt_symbiote.serving.vllm.dots_ocr:TTDotsOCRForConditionalGeneration`.
- **Or:** add an entry to `vllm_tt_plugin.platform.register_tt_models()`.

`dots_ocr`'s HF `config.architectures` is `["DotsOCRForCausalLM"]`, so the TT
prefix yields `TTDotsOCRForCausalLM` — that is the registry key.

### 4.3 model_spec catalog entry

Add a `tt_symbiote` `impl` id (new `ImplSpec` in `workflows/model_spec.py`) and a
YAML entry under `workflows/model_specs/dev/` sourced from `RUNTIME_PINS`:

```yaml
- weights: [rednote-hilab/dots.ocr]      # actual HF repo id TBD
  impl: tt_symbiote
  inference_engine: VLLM
  tt_metal_commit: c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195   # = RUNTIME_PINS["DotsOCRForCausalLM"]
  vllm_commit: <pin>
  device_model_specs:
    - device: T3K
      default_impl: true
      max_concurrency: 8
      max_context: <model max>
      override_tt_config: { trace_region_size: 200000000 }
```

The `tt_metal_commit` is the **single source of truth shared with
`RUNTIME_PINS`** — see §5.

### 4.4 Dockerfile / build changes

Extend (or branch from) `vllm.tt-metal.src.dev.Dockerfile` so that, after
tt-metal + ttnn are source-built from `TT_METAL_COMMIT_SHA_OR_TAG`, the image
**installs `tt_symbiote` against that source-built ttnn**:

```dockerfile
# after create_venv.sh (ttnn built from the pinned commit is already importable)
RUN pip install tt_symbiote==<version>
```

As of `tt_symbiote` 0.1.5, `ttnn` is **not** a dependency of the package, so a
plain `pip install tt_symbiote` can no longer override the source-built ttnn —
served images get ttnn built from the model's own commit, not a PyPI wheel. (The
older `--no-deps` + "install deps except ttnn" workaround was only needed for
≤0.1.4, which still pinned `ttnn==0.68.0`.)

---

## 5. The `RELEASE_TTNN` vs. per-model-commit coupling (a decision)

`tt_symbiote` ships a **single `RELEASE_TTNN`** PyPI pin (developer-experience:
one importable ttnn for `pip install tt_symbiote`). tt-inference-server wants
**ttnn built per-model from each model's commit**.

These are compatible because each model gets its own image. The explicit design
position:

> `RELEASE_TTNN` is the *developer/PyPI* install pin only. **Served images build
> ttnn from `RUNTIME_PINS[arch]["tt_metal_commit"]`** and install `tt_symbiote`
> with `--no-deps`. The runtime compat gate (`check_ttnn_compat`) then sees a
> *matching* commit and stays silent (instead of the soft-warning it emits today
> against the PyPI wheel).

Proposed mechanism to keep the two in sync (avoid drift between `RUNTIME_PINS`
and the model_spec YAML): a small exporter in `tt_symbiote` (or a check in
`tt-inference-server` CI) that emits/validates the YAML `tt_metal_commit` from
`RUNTIME_PINS`. Analogous to the existing `scripts/sync_ttnn_extras.py`.

---

## 6. Phased rollout

1. **Milestone 1 — `dots_ocr` end-to-end.**
   - Adapter wrapping `TTNNDotsOCRPipeline` (it already has prefill/decode/paged
     KV/traced decode/on-device argmax).
   - Plugin registration via `run_vllm_api_server`.
   - `dev/` model_spec entry + Dockerfile `pip install tt_symbiote --no-deps`.
   - Validate against the verified commit `c09f09c3` → confirms correct OCR (the
     thing no PyPI ttnn wheel could do).
2. **Milestone 2 — Ling-mini-2.0.**
   - Implement `reset_trace_state` for paged attention (currently
     `NotImplementedError`) so decode can be traced.
   - A `Generator` adapter replacing the HF `generate()` loop.
3. **Milestone 3 — generalize.**
   - A model-agnostic `tt_symbiote` `Generator` base so new recipes plug in by
     declaring prefill/decode + capabilities, mirroring `tt_transformers`.

---

## 7. Open questions / decisions needed

1. **Where does the adapter live** — in `tt_symbiote` (`serving/vllm/`, guarded
   import) or in `tt-inference-server`/the vLLM plugin? Recommendation: in
   `tt_symbiote` so the model code and its serving adapter ship together.
2. **Source-of-truth sync** — auto-generate the model_spec `tt_metal_commit` from
   `RUNTIME_PINS`, or validate-only in CI? Recommendation: exporter + CI check.
3. **`vllm_commit` per model** — `tt_symbiote` does not track this yet. Add a
   `vllm_commit` field to `RUNTIME_PINS` entries, or pin it at the
   tt-inference-server layer?
4. **Continuous batching** — `dots_ocr`'s DP batch is replication, not vLLM
   request batching. First milestone serves it as-is; do we need true continuous
   batching before "production"?
5. **Weight loading** — `tt_transformers` reads `HF_MODEL`; `tt_symbiote` uses
   `from_pretrained`. Confirm the adapter honors the container's `HF_MODEL`
   symlink / `MODEL_WEIGHTS_DIR` layout.

---

## 8. File-by-file change inventory (Milestone 1)

| Repo | File | Change |
|---|---|---|
| `tt_symbiote` | `src/tt_symbiote/serving/vllm/__init__.py` (new) | guarded-import adapter package |
| `tt_symbiote` | `src/tt_symbiote/serving/vllm/dots_ocr.py` (new) | `Generator` adapter over `TTNNDotsOCRPipeline` |
| `tt_symbiote` | `src/tt_symbiote/models/_runtime_pins.py` | add optional `vllm_commit` to entries (decision 3) |
| `tt_symbiote` | `scripts/export_model_spec.py` (new) | emit/validate model_spec `tt_metal_commit` from `RUNTIME_PINS` |
| `tt-inference-server` | `workflows/model_spec.py` | new `ImplSpec` `tt_symbiote` |
| `tt-inference-server` | `workflows/model_specs/dev/*.yaml` | `dots_ocr` entry (commit from `RUNTIME_PINS`) |
| `tt-inference-server` | `vllm-tt-metal/src/run_vllm_api_server.py` | register `TTDotsOCRForCausalLM` when `impl_id == "tt_symbiote"` |
| `tt-inference-server` | `vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile` | `pip install tt_symbiote --no-deps` after ttnn source build |

---

## 9. Prefill/decode bridge design (scalable to every tt_symbiote model)

This is the core of the integration: how vLLM's per-step `prefill_forward` /
`decode_forward` calls drive a tt_symbiote model. dots.ocr is the first model,
but the design must let **every** tt_symbiote model plug in with O(1) per-model
work. The solution is a **capability-tiered bridge** driven by a declarative
serving registry, so a single generic adapter serves all models.

### 9.1 The two contracts (grounded in code)

**vLLM `Generator` (target), from `tt-metal/.../generator.py` + `generator_vllm.py`:**

| Method | vLLM passes | Returns |
|---|---|---|
| `allocate_kv_cache(kv_cache_shape, dtype, num_layers, ...)` | block-paged shape from vLLM's block manager | **paged TT KV tensors** (one set per submesh) |
| `prefill_forward(tokens, page_table, kv_cache, prompt_lens, empty_slots, sampling_params, ...)` | full prompt tokens + **dynamic `page_table`** (logical→physical block ids) | **logits** `[B,1,vocab]` (or device-sampled tokens) |
| `decode_forward(tokens, start_pos, page_table, kv_cache, sampling_params, ...)` | one token/step + `start_pos` + `page_table` | **logits** `[B,1,vocab]` |

vLLM owns scheduling, continuous batching, block allocation, and (host or
on-device) sampling. The KV tensors are TT-allocated but **indexed by vLLM's
page table** — this is the paged-attention integration.

**tt_symbiote archetypes (today):**

| Archetype | Example | KV cache | Output of a step | Page table |
|---|---|---|---|---|
| Generate-engine | dots.ocr `TTNNDotsOCRPipeline` | own paged cache | **token** (on-device argmax) | fixed, internal |
| Paged HF model | Ling-mini `BailingMoeV2` | `TTNNPagedAttentionKVCache` (real ttnn paged ops) | **logits** via `model.forward` | **fixed `torch.arange`** (`ttnn_attention.py:101`) |
| Partial port | gemma4 / qwen3_vl | HF `DynamicCache` (host) | logits (host attn) | n/a |

**The decisive seam:** Ling-mini's paged cache already calls the *same* ttnn ops
vLLM's TT path uses (`paged_fill_cache` / `paged_update_cache` /
`paged_scaled_dot_product_attention_decode`), each of which accepts a
`page_table` argument. The only gap is that the wrapper hardcodes
`self._tt_page_table = arange(...)` instead of accepting vLLM's dynamic block
ids per step. **Closing that one seam unlocks true paged serving for every
tt_symbiote text LLM at once.**

### 9.2 Three serving tiers

A model declares the **deepest tier it supports**; one generic
`TTSymbioteGenerator(Generator)` dispatches by tier.

#### Tier S2 — paged, logits, continuous batching (target for text LLMs)
Full vLLM integration: continuous batching, prefix caching, vLLM sampling.

- `allocate_kv_cache` → build the model's `TTNNPagedAttentionKVCache` sized to
  vLLM's `(num_blocks, block_size)`; hand its TT k/v tensors to vLLM.
- `prefill_forward` → **set the cache's page table to vLLM's `page_table`** for
  this request, then call the model's *normal HF* `forward(input_ids,
  past_key_values=cache)` and return logits.
- `decode_forward` → set page table + `current_pos = start_pos`, call
  `forward(next_token, past_key_values=cache)`, return logits.

**Requires one additive, HF-preserving tt_symbiote hook** (shared by all S2
models, written once): install an external (vLLM block-manager) page table on
`TTNNPagedAttentionKVCache`. Implemented as:

```python
def set_vllm_page_table(self, page_table: torch.Tensor): ...
    # page_table: int32 [batch, blocks_per_sequence] logical->physical block ids.
    # Replaces the default contiguous arange mapping; reallocates the device
    # page-table tensor (call outside a trace boundary).
```

Per-request cache positions are NOT part of the hook — the adapter passes them
through the model's normal HF `cache_position` argument, so the cache needs no
position-tracking change. Default behavior (internal `arange`) is unchanged, so
HF `generate()` is byte-for-byte identical — `tt_symbiote` stays HF-shaped. This
is the *only* tt_symbiote-side change the whole roadmap needs.

**Status (M2):** implemented in `tt_symbiote.modules.ttnn_attention` and
**hardware-validated on T3K** — `tests/experimental/glm4_moe/Tier4/
test_vllm_page_table_hook.py` installs a non-identity (reversed) table and
confirms both `paged_fill_on_device` (prefill) and `paged_sdpa_decode` (decode
read-back) stay correct (PCC ≥ 0.99) against a torch `DynamicCache` reference,
plus a device page-table round-trip. The S2 adapter dispatch
(`allocate_kv_cache` / `_prefill_s2` / `_decode_s2` in
`tt_symbiote_generators.py`) is code-complete on top of this hook; full
multi-user continuous-batching e2e awaits a registered S2 model.

#### Tier S1 — logits, model-managed KV, no cross-request paging (fallback)
Model returns logits but its cache isn't vLLM-paged. The adapter serves one
request at a time (`max_num_seqs=1`); vLLM still samples. Lower throughput, zero
model changes. A safe default for any newly-ported logits model before its cache
is made page-table-aware.

#### Tier S0 — generate-engine, greedy only (dots.ocr today)
Model emits **tokens** (internal argmax), owns its cache, may be multimodal.
Bridge via the **one-hot-logits trick**:

- `decode_forward` runs the model's `decode_step(prev_token)` to get the argmax
  token `t`, then returns a synthetic logits vector with `+LARGE` at index `t`.
  vLLM's greedy sampler then selects exactly `t` and feeds it back as the next
  decode input — keeping the pipeline's internal cache/positions in lockstep.
- `prefill_forward` runs `pipeline.prefill(input_ids, pixel_values,
  image_grid_thw)` → first token → one-hot logits.
- `allocate_kv_cache` → `None` (model-managed; vLLM page table ignored).
- Constraints: greedy only (no temperature/top-p), `max_num_seqs=1` initially
  (the pipeline's DP batch is same-prompt replication, not vLLM request
  batching). Multimodal image inputs flow vLLM-MM → adapter → `pipeline.prefill`.

This lets a token-emitting engine masquerade as a logits model **with zero
tt_symbiote changes** — exactly what dots.ocr needs for milestone 1.

### 9.3 The scalability mechanism: a declarative serving registry

One generic adapter + a table keyed by HF architecture (mirrors how
`RUNTIME_PINS` already scales to 100+ models):

```python
# tt-inference-server side (vllm-tt-metal/src/tt_symbiote_generators.py)
SERVING_RECIPES = {
    "DotsOCRForCausalLM":      ServingRecipe(tier=S0_GREEDY_ENGINE, multimodal=True),
    "BailingMoeV2ForCausalLM": ServingRecipe(tier=S2_PAGED),
    # ... one row per model; no new adapter class per model ...
}

class TTSymbioteGenerator(Generator):
    """Dispatches prefill/decode by the model's ServingRecipe.tier."""
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, ...): ...
    def prefill_forward(self, *a, **k):  # dispatch S0/S1/S2
    def decode_forward(self, *a, **k):   # dispatch S0/S1/S2
    def allocate_kv_cache(self, *a, **k):# dispatch by tier
```

Registration becomes a loop over `SERVING_RECIPES` in
`run_vllm_api_server.register_tt_models()` → every arch maps `TT{Arch}` to the
**same** `TTSymbioteGenerator`. Adding model #101 = one registry row (+ for S2,
nothing, because the page-table hook is shared).

**Source of truth.** The tier is a property of the model, so the cleanest place
is a tiny additive field in tt_symbiote's existing per-model metadata
(`RUNTIME_PINS[arch]["serving_tier"]`, metadata only — no behavior change),
exported alongside the tt-metal commit. The adapter reads it; tt-inference-server
holds no per-model logic. This keeps the model package authoritative and the
serving layer generic, matching the "pin = metadata" philosophy of
`RUNTIME_PINS`.

### 9.4 Multimodal (dots.ocr, future VLMs)

vLLM routes image inputs through its multimodal registry. The tt_transformers
VLM Generators (`Gemma3ForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`)
register a `MULTIMODAL_REGISTRY` processor and subclass `SupportsMultiModal`.
`TTSymbioteGenerator` needs the same wiring for `multimodal=True` recipes:
accept `pixel_values` / `image_grid_thw` from the vLLM request and pass them into
`prefill_forward`. This is per-modality (image/video/audio) glue written once and
reused by every VLM recipe.

### 9.5 Recommended milestone path

1. **M1 (dots.ocr, S0):** implement the one-hot-logits bridge + multimodal image
   wiring in `TTSymbioteGenerator`. **Zero tt_symbiote changes.** `max_num_seqs=1`,
   greedy. Validates the whole pipeline (image build, registration, serving) on
   `c09f09c3`.
2. **M2 (S2 seam):** add the additive `set_vllm_page_table` hook to
   `TTNNPagedAttentionKVCache` (the one shared tt_symbiote change) + S2 dispatch
   in the adapter. **Done + T3K-validated at the hook level** (see §9.2 Status).
   Unlocks continuous batching for the whole text-LLM family once an S2 model is
   registered. (Ling-mini/`bailing_moe_v2` is the intended first S2 model but
   currently fails import — its `TTNNBailingMoeV2Model` extends `TTNNModule`
   directly, which the class guard disallows; fixing that is the remaining step
   to take S2 e2e.)
3. **M3 (generalize):** `SERVING_RECIPES` + `serving_tier` metadata export; new
   models are one row. Port gemma4/qwen3_vl attention to reach S1/S2.

### 9.6 Open bridge decisions

1. **dots.ocr DP under vLLM** — keep `max_num_seqs=1` (simplest), or map vLLM's
   request batch onto the pipeline's 8 DP streams (higher throughput, more glue)?
2. **S2 KV ownership** — does vLLM allocate the paged tensors (and we wrap them
   as the model's cache), or does the model allocate and we expose them to vLLM?
   tt_transformers does the latter (`allocate_vllm_kv_cache`); recommend matching.
3. **`serving_tier` location** — in tt_symbiote `RUNTIME_PINS` (model-authoritative,
   tiny additive metadata) vs. a table in tt-inference-server (zero tt_symbiote
   touch). Leaning model-authoritative for scalability/consistency with pins.

## References

- `src/tt_symbiote/models/_runtime_pins.py` — per-model commit pinning, "pin vs.
  install", and the one-ttnn-per-process constraint this design works within
  (`RUNTIME_PINS`, `RELEASE_TTNN`); the README Installation section covers the
  source-built-ttnn policy.
- `tt-metal/models/tt_transformers/tt/generator.py` — the `Generator` base whose
  contract the adapter implements (`prefill_forward`, `decode_forward`).
- `tt-metal/models/tt_transformers/tt/generator_vllm.py` — `allocate_vllm_kv_cache`
  and per-model `initialize_vllm_model` / `allocate_kv_cache`.
- `tt_symbiote/src/tt_symbiote/modules/ttnn_attention.py` — `TTNNPagedAttentionKVCache`
  (the paged ops + the fixed `page_table` seam to make vLLM-aware for Tier S2).
- `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py` — `register_tt_models()`
  and the `TT{Arch}` naming rule.
