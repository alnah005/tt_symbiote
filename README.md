# tt_symbiote

A pip-installable Python library whose user-facing API mirrors
[`transformers`](https://github.com/huggingface/transformers) and whose
folder layout mirrors `transformers/src/transformers/`. Provides
TTNN-accelerated implementations of Hugging Face model architectures on
Tenstorrent Wormhole hardware (N150 / N300 / T3K).

If you already know how to use `transformers`, you already know how to
use `tt_symbiote`. The only mandatory difference: a single
`set_device(model, mesh)` call between `from_pretrained` and the first
forward pass.

> **Status (0.1.0 — first public release).** Four model recipes ship
> hardware-verified and CI-tested:
>
> | Recipe | Variant | HW target | Path |
> |---|---|---|---|
> | `BailingMoeV2ForCausalLM` | `inclusionAI/Ling-mini-2.0` | T3K (1×8) | full TTNN |
> | `ResNetForImageClassification` | `microsoft/resnet-50` | N150 (1×1) | full TTNN |
> | `Gemma4ForConditionalGeneration` | `google/gemma-4-{E2B,E4B}-it` | N150 (1×1) | partial TTNN (Wave A) |
> | `Gemma4ForConditionalGeneration` | `google/gemma-4-{31B,26B-A4B}-it` | T3K (1×8) | CPU-first via budget gate |
> | `Qwen3VLForConditionalGeneration` | `Qwen/Qwen3-VL-{2B,4B,8B,32B}-Instruct` | N150 / T3K | partial TTNN (Wave B) |
>
> Tensor-parallel sharding for the oversize Gemma-4 / Qwen3-VL
> variants is the headline item for the next release. See
> [`docs/supported_models.md`](docs/supported_models.md) for the full
> per-variant matrix and
> [`docs/development/cpu_vs_device_coverage.md`](docs/development/cpu_vs_device_coverage.md)
> for the per-module CPU-vs-device split.

---

## Installation

`tt_symbiote` runs on Linux with Python 3.10 – 3.12 on a host with a
Tenstorrent Wormhole device attached.

### From PyPI

```bash
python -m venv .venv
source .venv/bin/activate
pip install "tt_symbiote[ttnn]"
```

The `[ttnn]` extra pulls the exact `ttnn` wheel this release was
validated against (`ttnn==0.68.0`, matching `scripts/ttnn-pin.txt`)
plus the transitive deps (`torch`, `transformers==5.9.0`,
`accelerate`, `tokenizers`, …). Plain `pip install tt_symbiote` (no
extra) installs the Python surface only; you are then responsible for
matching a `ttnn` wheel to your host sfpi yourself.

> **System prerequisite: sfpi 7.35.3.** `ttnn` JIT-compiles firmware
> kernels at first `open_mesh_device(...)` call using the Tenstorrent
> sfpi RISC-V toolchain at `/opt/tenstorrent/sfpi/`. Every `ttnn`
> wheel pins one sfpi version; if the host's installed sfpi doesn't
> match, `open_mesh_device()` fails with `unrecognized command-line
> option`. The `[ttnn]` extra pins `ttnn==0.68.0`, which requires
> `sfpi 7.35.3`. Verify with
>
> ```bash
> /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version
> # → expect: sfpi:7.35.3 ...
> ```
>
> See [`docs/install_prerequisites.md`](docs/install_prerequisites.md)
> for how to install / upgrade the sfpi toolchain and the (ttnn, sfpi)
> compatibility table.

### From source (git clone)

For contributors, or when you want to run the bundled `examples/e2e/`
demos and tests:

```bash
git clone https://github.com/alnah005/tt_symbiote.git
cd tt_symbiote
./scripts/bootstrap_venv.sh        # creates .venv, validates sfpi, installs ttnn + tt_symbiote (editable)
source .venv/bin/activate
```

The bootstrap script reads `(ttnn, sfpi)` from
[`scripts/ttnn-pin.txt`](scripts/ttnn-pin.txt), probes the system sfpi
toolchain at `/opt/tenstorrent/sfpi/`, **refuses to proceed if the
versions disagree** (preventing the silent runtime failure described
above), and then installs `ttnn==<pinned>`, `torch`,
`transformers==5.9.0`, and `tt_symbiote` (editable) into a fresh venv.
No `tt-metal` source checkout is required.

> Note for the VLM demos. The published repo deliberately ships **no
> binary image** (`tests/images/` is gitignored). The VLM e2e scripts
> under `examples/e2e/{gemma4,qwen3_vl}/` read a picture from
> `tests/images/test-dog.png`. Before running them, drop any
> Pillow-readable picture of a dog at that path:
>
> ```bash
> mkdir -p tests/images
> cp ~/Pictures/your_dog.jpg tests/images/test-dog.png
> ```
>
> If the file is missing, each VLM script exits 0 with a clear
> `SKIP:` message before acquiring a TTNN device — so it's safe to
> run the entire `examples/e2e/` tree on a fresh clone.

---

## Run a model

### Causal LM on T3K — `inclusionAI/Ling-mini-2.0`

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

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0",
    trust_remote_code=True,
    dtype="auto",
    kv_cache_kwargs={"max_num_blocks": 512},     # paged KV cache budget
)
set_device(model, mesh_device)                    # mandatory

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain the difference between Python and C++."}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(
    **inputs, max_new_tokens=512, use_cache=True,
    past_key_values=model._tt_kv_cache,
)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:]))

ttnn.close_mesh_device(mesh_device)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
```

End-to-end runnable copy:
[`examples/e2e/run_ling_mini_2_0.py`](examples/e2e/run_ling_mini_2_0.py).

### Image classification on N150 — `microsoft/resnet-50`

```python
import os
os.environ.setdefault("MESH_DEVICE", "N150")

import ttnn
from PIL import Image
from transformers import AutoImageProcessor
from tt_symbiote import AutoModelForImageClassification, set_device

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    l1_small_size=245760,
)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
set_device(model, mesh_device)

inputs = processor(images=Image.open("my_image.png").convert("RGB"), return_tensors="pt")
out = model(**inputs)
predicted_class = out.logits.argmax(-1).item()
print(model.config.id2label[predicted_class])

ttnn.close_mesh_device(mesh_device)
```

End-to-end runnable copy:
[`examples/e2e/resnet/run_resnet50.py`](examples/e2e/resnet/run_resnet50.py).

### Vision-language model on N150 — `google/gemma-4-E2B-it`

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
model = AutoModelForImageTextToText.from_pretrained("google/gemma-4-E2B-it", dtype=torch.bfloat16)
set_device(model, mesh_device, dump_visualization=False)

image = Image.open("my_dog.png").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "What is this animal in the photo?"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
)

out = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0])

ttnn.close_mesh_device(mesh_device)
```

End-to-end runnable copies live under
[`examples/e2e/gemma4/`](examples/e2e/gemma4/) and
[`examples/e2e/qwen3_vl/`](examples/e2e/qwen3_vl/).

### One-shot smoke runs (from a `git clone`)

```bash
source .venv/bin/activate
python examples/e2e/run_ling_mini_2_0.py            # T3K causal LM
python examples/e2e/resnet/run_resnet50.py          # N150 image classification
python examples/e2e/gemma4/run_gemma4_e2b.py        # N150 VLM (needs tests/images/test-dog.png)
python examples/e2e/qwen3_vl/run_qwen3_vl_2b.py     # N150 VLM (needs tests/images/test-dog.png)
```

Each script writes a `<script>_coverage.json` next to itself
summarising what executed on device vs CPU (gitignored, regenerated on
every run).

---

## What's actually exposed

`tt_symbiote` re-exports `transformers`'s entire `Auto*` surface
(43 classes), plus three additions:

```python
from tt_symbiote import (
    AutoModelForCausalLM,            # plus all 42 other Auto* classes
    set_device,                       # bind a loaded model to a TTNN mesh
    register_modules,                 # public hook for new recipes
    register_recipe,                  # recipe decorator
    TT_MODEL_REGISTRY,                # {hf_class_name: Recipe}
    compatibility,                    # runtime coverage observation
)
```

`compatibility.report(model)` returns a JSON-friendly dict describing
which modules were swapped to TTNN, which ran on device successfully,
and which fell back to CPU. See
[`docs/development/cpu_vs_device_coverage.md`](docs/development/cpu_vs_device_coverage.md)
for the schema and the conventions.

---

## Repository layout

Mirrors `transformers/src/transformers/`:

```text
src/tt_symbiote/
├── __init__.py            # public API (Auto* + set_device + register_modules + ...)
├── core/                  # INTERNAL: TTNNModule, run_config, vendored CCL, arch helpers
├── models/
│   ├── auto/              # AutoModelForCausalLM, AutoModel, ... (43 Auto* classes)
│   ├── bailing_moe_v2/    # Ling-mini-2.0 family (full TTNN port)
│   ├── gemma4/            # Gemma-4 family (Wave A TTNN + budget gate)
│   ├── qwen3_vl/          # Qwen3-VL family (Wave B TTNN)
│   └── resnet/            # Microsoft ResNet (full TTNN port)
├── modules/               # generic, model-agnostic TTNN building blocks
└── utils/                 # set_device, register_modules, compatibility, hf_compat, ...
```

Examples and tests live at the repo root (not inside the package):

```text
examples/e2e/              # one runnable script per model variant
tests/{auto,capabilities,models}/   # CI-gated tests for the supported recipes
tests/experimental/        # quarantined tests for unsupported variants (excluded from sdist)
```

See [`docs/development/PROJECT_PROPOSAL.md`](docs/development/PROJECT_PROPOSAL.md)
§3 for the full layout and the rationale.

---

## Development

```bash
make install        # editable install + dev extras
make lint           # pre-commit on all files
make test           # capability tests under tests/auto
make smoke MODEL=bailing_moe_v2    # per-model smoke test
make dist           # build sdist + wheel + twine check
```

`tracy` (Tenstorrent's profiler) is optional — `tt_symbiote.core.run_config`
guards `from tracy import signpost` behind a `try/except` and falls
back to a no-op shim when tracy is absent. Set
`TT_SYMBIOTE_SIGNPOST_MODE` in the environment to enable profiling
hooks when tracy is installed.

`transformers==5.9.0` is the strict pin for this release branch; every
other dep in `pyproject.toml` mirrors the specifier used by HF
transformers v5.9.0's own `setup.py`. See
[`docs/development/migration_notes.md`](docs/development/migration_notes.md)
for the bump procedure when moving to a different transformers
release.

---

## Versioning

One `transformers` version per `tt_symbiote` branch. The release
branch (`transformers5.9.0`) **is** the release marker — there are no
git tags. Each release is a one-line `version` bump in
`pyproject.toml` followed by a manual `workflow_dispatch` of
[`.github/workflows/release.yml`](.github/workflows/release.yml). See
[`docs/development/release_process.md`](docs/development/release_process.md)
and
[`docs/development/PROJECT_PROPOSAL.md`](docs/development/PROJECT_PROPOSAL.md)
§10 for the branching policy.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
