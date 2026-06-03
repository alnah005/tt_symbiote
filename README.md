# tt_symbiote

**The `transformers` API, accelerated on Tenstorrent silicon.**

`tt_symbiote` is a pip-installable Python library whose public surface mirrors
[Hugging Face transformers](https://github.com/huggingface/transformers)
and whose model implementations run on Tenstorrent Wormhole hardware
(N150 / N300 / T3K) via [TTNN](https://docs.tenstorrent.com/ttnn/latest/).

If you know `transformers`, you already know `tt_symbiote`. The only line you
add to a normal HF script is `set_device(model, mesh)`, which binds the loaded
model to a TTNN mesh device:

```python
import ttnn
from tt_symbiote import AutoModelForCausalLM, set_device

mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8))
model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0", trust_remote_code=True,
)
set_device(model, mesh)              # <-- the one TT-specific line

# Everything below is the same as `transformers`:
out = model.generate(input_ids, max_new_tokens=512)
```

The four design rules:

- **Public API = `transformers`'s `Auto*` surface.** Same class names, same
  `from_pretrained(...)` signature, same `model.generate(...)`. Drop-in.
- **One mandatory device-binding step.** `set_device(model, mesh)` takes
  exactly two arguments. Everything configurable lives on `from_pretrained`.
- **Per-module TTNN execution with a CPU safety net.** Modules that have a
  TTNN implementation run on device; anything else (or anything that hits a
  dtype / shape edge) falls back to its original CPU `nn.Module` and the
  forward pass keeps going.
- **One `transformers` version per `tt_symbiote` release branch.** This
  release ships against `transformers==5.9.0` (strict pin).

---

## Installation

`tt_symbiote` runs on **Linux, Python 3.10 – 3.12, on a host with a
Tenstorrent Wormhole device attached** and the matching `sfpi` toolchain
installed at `/opt/tenstorrent/sfpi/`.

```bash
python -m venv .venv
source .venv/bin/activate

# Text-only causal LMs (e.g. inclusionAI/Ling-mini-2.0):
pip install tt_symbiote

# Multimodal / vision models (Gemma-4, Qwen3-VL, ResNet image classification):
pip install "tt_symbiote[vision]"
```

The base install pulls `ttnn==0.68.0` plus the transitive HF stack
(`torch`, `transformers==5.9.0`, `accelerate`, `tokenizers`, …) from PyPI.
There is no separate `[ttnn]` extra — `ttnn` is a hard dependency, because
core modules import it at module-load time.

The `[vision]` extra adds `torchvision`, which HF's multimodal
`AutoProcessor` classes (`Gemma4VideoProcessor`, the Qwen3-VL preprocessor,
the ResNet image processor) require at construction. This mirrors the
upstream `transformers[vision]` extra exactly.

> **System prerequisite: `sfpi 7.35.3`.** `ttnn==0.68.0` JIT-compiles
> firmware kernels at first `open_mesh_device(...)` call using the
> Tenstorrent `sfpi` RISC-V toolchain. Mismatched `sfpi` causes
> `open_mesh_device()` to fail with `unrecognized command-line option`.
> Verify with:
>
> ```bash
> /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version
> # → expect: sfpi:7.35.3 ...
> ```
>
> See [`docs/install_prerequisites.md`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/docs/install_prerequisites.md)
> for installation and the `(ttnn, sfpi)` compatibility table.

---

## How to use it

Three archetypes covering everything the 0.1.3 release supports: a text-only
causal LM, an image classifier, and a vision-language model. Each block
runs end-to-end on real hardware.

### 1. Text generation on T3K — `inclusionAI/Ling-mini-2.0`

The most thoroughly ported recipe: every transformer block, MoE router,
KV cache, and linear projection runs on TTNN.

```python
import os
os.environ.setdefault("MESH_DEVICE", "T3K")

import torch
import ttnn
from transformers import AutoTokenizer
from tt_symbiote import AutoModelForCausalLM, set_device

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    trace_region_size=200_000_000,
    num_command_queues=1,
)

tokenizer = AutoTokenizer.from_pretrained(
    "inclusionAI/Ling-mini-2.0", trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0",
    trust_remote_code=True,
    dtype="auto",
    kv_cache_kwargs={"max_num_blocks": 512},   # paged-attention budget
)
set_device(model, mesh_device)

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain Python vs C++ in two sentences."}],
    add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(
    **inputs, max_new_tokens=256, use_cache=True,
    past_key_values=model._tt_kv_cache,
)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:]))

ttnn.close_mesh_device(mesh_device)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
```

### 2. Image classification on N150 — `microsoft/resnet-50`

A small, complete TTNN port — useful as a sanity check on a fresh install.

```python
import os
os.environ.setdefault("MESH_DEVICE", "N150")

import ttnn
from PIL import Image
from transformers import AutoImageProcessor
from tt_symbiote import AutoModelForImageClassification, set_device

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=245760,
)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
set_device(model, mesh_device)

inputs = processor(
    images=Image.open("my_image.png").convert("RGB"),
    return_tensors="pt",
)
out = model(**inputs)
print(model.config.id2label[out.logits.argmax(-1).item()])

ttnn.close_mesh_device(mesh_device)
```

### 3. Vision-language on N150 — `google/gemma-4-E2B-it`

Multimodal `AutoProcessor` + `AutoModelForImageTextToText`, identical to
the HF VLM pattern. Requires the `[vision]` extra. The image-and-text chat
template is the canonical HF VLM input format.

```python
import os
os.environ.setdefault("MESH_DEVICE", "N150")

import torch
import ttnn
from PIL import Image
from transformers import AutoProcessor
from tt_symbiote import AutoModelForImageTextToText, set_device

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    trace_region_size=200_000_000,
    l1_small_size=245760,
)

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E2B-it", dtype=torch.bfloat16,
)
set_device(model, mesh_device)

image = Image.open("my_dog.png").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text":  "What animal is in this photo?"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
)

out = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
print(processor.batch_decode(
    out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
)[0])

ttnn.close_mesh_device(mesh_device)
```

End-to-end runnable copies of every example above (and one per supported
model variant) live in
[`examples/e2e/`](https://github.com/alnah005/tt_symbiote/tree/transformers5.9.0/examples/e2e).

---

## Public API

`tt_symbiote` re-exports the entire `transformers.Auto*` surface — all
43 `Auto*` classes — and adds a small TT-specific surface:

```python
from tt_symbiote import (
    AutoModelForCausalLM,           # plus all 42 other Auto* classes
    set_device,                     # bind a loaded model to a TTNN mesh device
    register_modules,               # public hook for new recipe authors
    register_recipe,                # recipe decorator
    TT_MODEL_REGISTRY,              # {HF_class_name: Recipe}
    compatibility,                  # runtime coverage observation
)
```

`set_device(model, mesh)` is the only required new line in an HF-shaped
script. All runtime configuration (KV-cache budget, optional diagnostics)
flows through `AutoModel*.from_pretrained` keyword arguments, mirroring
the way `transformers` itself takes `torch_dtype`, `attn_implementation`,
etc.

`compatibility.report(model)` returns a JSON-serialisable dict describing
which modules were swapped to TTNN, which ran successfully on device, and
which fell back to CPU. See
[`docs/development/cpu_vs_device_coverage.md`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/docs/development/cpu_vs_device_coverage.md)
for the schema.

---

## Supported models in 0.1.3

| Architecture (HF class) | Model variants | Hardware target | Status |
|---|---|---|---|
| `BailingMoeV2ForCausalLM` | `inclusionAI/Ling-mini-2.0` | T3K (1×8) | full TTNN |
| `ResNetForImageClassification` | `microsoft/resnet-{18,34,50,101,152}` | N150 (1×1) | full TTNN |
| `Gemma4ForConditionalGeneration` | `google/gemma-4-{E2B,E4B}-it` | N150 (1×1) | partial TTNN (Wave A) |
| `Gemma4ForConditionalGeneration` | `google/gemma-4-{31B,26B-A4B}-it` | T3K (1×8) | CPU-first via budget gate |
| `Qwen3VLForConditionalGeneration` | `Qwen/Qwen3-VL-{2B,4B,8B,32B}-Instruct` | N150 / T3K | partial TTNN (Wave B) |

See
[`docs/supported_models.md`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/docs/supported_models.md)
for the full per-variant matrix and
[`docs/development/cpu_vs_device_coverage.md`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/docs/development/cpu_vs_device_coverage.md)
for the per-module CPU-vs-device split.

Tensor-parallel sharding for the oversize Gemma-4 / Qwen3-VL variants
(today gated to CPU on T3K because of single-device memory pressure) is
the headline item for the next release.

---

## Recent releases

- **0.1.3** — PyPI project description refresh. No code or API changes;
  the published wheel is byte-identical in behavior to 0.1.2. Bumped
  because PyPI metadata is immutable per-version and the project page
  needed to reflect the user-facing narrative.
- **0.1.2** — `[vision]` extra and strict `set_device(model, mesh)`
  signature. `pip install "tt_symbiote[vision]"` pulls `torchvision`,
  the implicit prerequisite for HF's multimodal `AutoProcessor` classes;
  text-only users do not pay this cost. All runtime configuration
  (`kv_cache_kwargs`, optional diagnostics) moved to `from_pretrained`
  keyword arguments, leaving the binding step pure: walk the tree,
  attach `TTNNModule`s to the mesh, allocate KV cache, return.
- **0.1.1** — `ttnn` promoted from optional extra to hard dependency.
  No `[ttnn]` opt-in step; `pip install tt_symbiote` resolves the
  matching `ttnn==0.68.0` wheel directly.

Full release history:
[CHANGELOG](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/docs/development/release_checklist.md).

---

## From-source install (contributors)

For running the bundled e2e demos and the test suite:

```bash
git clone https://github.com/alnah005/tt_symbiote.git
cd tt_symbiote
./scripts/bootstrap_venv.sh        # creates .venv, validates sfpi, installs ttnn + tt_symbiote (editable)
source .venv/bin/activate
```

The bootstrap script reads `(ttnn, sfpi)` from
[`scripts/ttnn-pin.txt`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/scripts/ttnn-pin.txt),
probes the system `sfpi` toolchain, **refuses to proceed if the versions
disagree**, and then installs `ttnn`, `torch`, `transformers==5.9.0`, and
`tt_symbiote` (editable). No `tt-metal` source checkout is required.

The repository layout mirrors `transformers/src/transformers/`:

```text
src/tt_symbiote/
├── __init__.py            # public API (Auto* + set_device + register_modules + …)
├── core/                  # TTNNModule, run_config, arch helpers (internal)
├── models/
│   ├── auto/              # 43 Auto* classes
│   ├── bailing_moe_v2/    # Ling-mini-2.0 family (full TTNN)
│   ├── gemma4/            # Gemma-4 family (Wave A TTNN + budget gate)
│   ├── qwen3_vl/          # Qwen3-VL family (Wave B TTNN)
│   └── resnet/            # Microsoft ResNet (full TTNN)
├── modules/               # generic TTNN building blocks (Linear, Embedding, …)
└── utils/                 # set_device, register_modules, compatibility, hf_compat, …
```

Examples and tests live at the repo root (not inside the package), under
[`examples/e2e/`](https://github.com/alnah005/tt_symbiote/tree/transformers5.9.0/examples/e2e)
and
[`tests/`](https://github.com/alnah005/tt_symbiote/tree/transformers5.9.0/tests).
See
[`docs/development/PROJECT_PROPOSAL.md`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/docs/development/PROJECT_PROPOSAL.md)
for the full layout rationale and the porting recipe contract.

---

## Versioning

One `transformers` version per `tt_symbiote` release branch. This release
branch (`transformers5.9.0`) ships `transformers==5.9.0`. The branch *is*
the release marker — there are no git tags. Each release is a one-line
`version` bump in `pyproject.toml` followed by a manual dispatch of
[`.github/workflows/release.yml`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/.github/workflows/release.yml).
See
[`docs/development/release_process.md`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/docs/development/release_process.md)
for the full recipe.

---

## Links

- **Repository:** <https://github.com/alnah005/tt_symbiote>
- **Issues / discussions:** <https://github.com/alnah005/tt_symbiote/issues>
- **TTNN documentation:** <https://docs.tenstorrent.com/ttnn/latest/>
- **License:** Apache-2.0
  ([`LICENSE`](https://github.com/alnah005/tt_symbiote/blob/transformers5.9.0/LICENSE))
