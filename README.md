# tt_symbiote

**The `transformers` API, accelerated on Tenstorrent silicon.**

`tt_symbiote` is a pip-installable Python library whose public surface mirrors
[Hugging Face transformers](https://github.com/huggingface/transformers) and
whose model implementations run on Tenstorrent Wormhole hardware (N150 / N300 /
T3K) via [TTNN](https://docs.tenstorrent.com/ttnn/latest/). The only line you
add to a normal HF script is `set_device(model, mesh)`:

```python
import ttnn
from transformers import AutoTokenizer
from tt_symbiote import AutoModelForCausalLM, set_device

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8), trace_region_size=200_000_000)

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0", trust_remote_code=True, dtype="auto",
    kv_cache_kwargs={"max_num_blocks": 512},   # paged-attention budget
)
set_device(model, mesh)              # <-- the one TT-specific line

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain Python vs C++ in two sentences."}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, use_cache=True, past_key_values=model._tt_kv_cache)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:]))

ttnn.close_mesh_device(mesh)
```

Runnable copies of this and every supported variant live in
[`examples/e2e/`](examples/e2e/).

The design rules:

- **Public API = `transformers`'s `Auto*` surface** — same class names, same
  `from_pretrained(...)`, same `model.generate(...)`. Drop-in.
- **One device-binding step** — `set_device(model, mesh)` takes exactly two
  arguments; all configuration flows through `from_pretrained`.
- **Per-module TTNN with a CPU safety net** — modules with a TTNN implementation
  run on device; anything else falls back to its CPU `nn.Module`.

## Installation

`tt_symbiote` runs on **Linux, Python 3.10 – 3.12, on a host with a Tenstorrent
Wormhole device attached** (N150 / N300 / T3K).

```bash
python -m venv .venv && source .venv/bin/activate
pip install tt_symbiote            # text-only causal LMs
pip install "tt_symbiote[vision]"  # multimodal / vision (Gemma-4, Qwen3-VL, ResNet)
pip install "tt_symbiote[all]"     # every model's optional deps
```

The install pulls the transitive HF stack (`torch`, `transformers==5.9.0`,
`accelerate`, `tokenizers`, …). The `[vision]` extra adds `torchvision`, which
HF's multimodal `AutoProcessor` classes require (mirrors the upstream
`transformers[vision]` extra).

> **`ttnn` is NOT installed by pip — you must provide it from source.** `ttnn` is
> a compiled extension tied to a specific `tt-metal` commit, so `tt_symbiote`
> deliberately does **not** depend on it: a PyPI `ttnn` wheel would silently
> overwrite the source build and corrupt numerics. Build `tt-metal` at the
> model's pinned commit and set `$TT_METAL_HOME` so `ttnn` is importable;
> `import tt_symbiote` auto-wires the source-built `ttnn` (and raises a clear,
> actionable error if none is found). Each model records the specific
> `tt_metal_commit` it was verified against — metadata enforced at load time by a
> compatibility gate that warns when the installed `ttnn` was built from a
> different commit.
>
> ttnn JIT-compiles firmware kernels at the first `open_mesh_device(...)` call
> using the `sfpi` RISC-V toolchain. See
> [`docs/install_prerequisites.md`](docs/install_prerequisites.md) for details.

## Public API

`tt_symbiote` re-exports the entire `transformers` `Auto*` loader surface (same
class names) and adds a small TT-specific surface:

```python
from tt_symbiote import (
    AutoModelForCausalLM,           # plus the rest of the Auto* loaders
    set_device,                     # bind a loaded model to a TTNN mesh device
    register_modules,               # public hook for new recipe authors
    register_recipe,                # recipe decorator
    TT_MODEL_REGISTRY,              # {HF_class_name: Recipe}
    compatibility,                  # runtime coverage observation
)
```

`set_device(model, mesh)` is the only required new line. `compatibility.report(model)`
returns a JSON dict of which modules ran on TTNN vs fell back to CPU — see
[`docs/development/cpu_vs_device_coverage.md`](docs/development/cpu_vs_device_coverage.md)
for the schema.

## Supported models

| Architecture (HF class) | Variants | Hardware | Status |
|---|---|---|---|
| `BailingMoeV2ForCausalLM` | `inclusionAI/Ling-mini-2.0` | T3K (1×8) | full TTNN |
| `ResNetForImageClassification` | `microsoft/resnet-{18,34,50,101,152}` | N150 (1×1) | full TTNN |
| `Gemma4ForConditionalGeneration` | `google/gemma-4-{E2B,E4B}-it` | N150 (1×1) | partial TTNN |
| `Gemma4ForConditionalGeneration` | `google/gemma-4-{31B,26B-A4B}-it` | T3K (1×8) | CPU-first via budget gate |
| `Qwen3VLForConditionalGeneration` | `Qwen/Qwen3-VL-{2B,4B,8B,32B}-Instruct` | N150 / T3K | partial TTNN |

See [`docs/supported_models.md`](docs/supported_models.md) for the full
per-variant matrix and [`examples/e2e/README.md`](examples/e2e/README.md) for the
per-script index.

## From-source (contributors)

For the bundled e2e demos and the test suite; a built `tt-metal` at
`$TT_METAL_HOME` provides ttnn. `scripts/bootstrap_venv.sh` is optional (wires
ttnn, checks sfpi, `pip install -e .` + pre-commit hook); with `$TT_METAL_HOME`
set, a plain `pip install -e .` also works because `import tt_symbiote`
auto-wires the source-built ttnn. The layout mirrors
`transformers/src/transformers/`:

```text
src/tt_symbiote/
├── __init__.py    # public API (Auto* + set_device + register_modules + …)
├── core/          # TTNNModule, run_config, arch helpers (internal)
├── models/auto/   # transformers Auto* loaders
├── models/<name>/ # per-model recipes (bailing_moe_v2, gemma4, qwen3_vl, resnet, …)
├── modules/       # generic TTNN building blocks (Linear, Embedding, …)
└── utils/         # set_device, register_modules, compatibility, hf_compat, …
```

Examples and tests live at the repo root (not inside the package), under
[`examples/e2e/`](examples/e2e/) and [`tests/`](tests/). See
[`docs/development/PROJECT_PROPOSAL.md`](docs/development/PROJECT_PROPOSAL.md) for
the layout rationale and the porting recipe contract.

## Links

- **Repository:** <https://github.com/alnah005/tt_symbiote>
- **Issues / discussions:** <https://github.com/alnah005/tt_symbiote/issues>
- **TTNN documentation:** <https://docs.tenstorrent.com/ttnn/latest/>
- **License:** Apache-2.0 ([`LICENSE`](LICENSE))
