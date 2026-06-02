# tt_symbiote

Hugging Face of Tenstorrent: a pip-installable Python library whose user-facing API
mirrors [`transformers`](https://github.com/huggingface/transformers) and whose
folder layout mirrors `transformers/src/transformers/`. Provides TTNN-accelerated
implementations of HF model architectures.

> **Status: Phase 8 Wave A + B landed; four Gemma-4 variants and Qwen3-VL-2B verified end-to-end.**
> Phases 1–4 land the package skeleton, dispatcher removal, and the public
> API (43 `Auto*` classes + `set_device` + `register_modules` + the recipe
> registry). Phase 5 ports Ling-mini-2.0 (full TTNN on T3K). Phase 6 ports
> Microsoft ResNet (full TTNN on N150). Phase 7 ports Gemma-4 and
> Qwen3-VL-2B (CPU-first VLM template + a project-scope porting skill).
> Phase 8 turns the structurally simple Gemma-4 and Qwen3-VL compute
> (RMSNorm, scaled embedding, text/vision MLPs, patch merger, multimodal
> embedder — 9 wrappers total) into on-device execution, and adds a per-
> recipe budget/MoE gate that cleanly forces oversize or MoE Gemma-4
> variants to CPU until tensor-parallel sharding lands. All four Gemma-4
> variants and Qwen3-VL-2B run end-to-end via
> `tt_symbiote.AutoModel*.from_pretrained` + `set_device` + `generate`
> with `compatibility.report(...)["regressions"] == []`.
> See [`docs/cpu_vs_device_coverage.md`](docs/cpu_vs_device_coverage.md)
> for the per-variant CPU/device split,
> [`docs/internal/migration_notes.md`](docs/internal/migration_notes.md) for the per-phase
> design rationale, and [`docs/internal/PROJECT_PROPOSAL.md`](docs/internal/PROJECT_PROPOSAL.md) for
> the original (frozen) design intent.

## Target user-facing API

```python
from tt_symbiote import AutoModelForCausalLM, set_device
import ttnn

model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ling-mini-2.0", trust_remote_code=True, dtype="auto",
)
set_device(model, ttnn.open_mesh_device(...))   # mandatory final step

out = model.generate(**inputs, max_new_tokens=128, use_cache=True)
```

Differences from `transformers`:

1. Import path is `tt_symbiote` instead of `transformers`.
2. `set_device(model, device)` is mandatory after `from_pretrained` and before any
   forward pass. Missing it is a hard error.

Everything else (tokenizers, `generate`, `Auto*`, processors) matches Hugging Face
exactly.

## Repository layout

Mirrors `transformers/src/transformers/`:

```text
src/tt_symbiote/
├── auto/              # AutoModelForCausalLM, AutoModel, ... (43 Auto* classes)
├── integrations/      # generic, model-agnostic TTNN building blocks
├── models/<model>/    # per-model TTNN modeling code, mirrors HF layout
├── core/              # INTERNAL: TTNNModule, run_config, vendored CCL
└── utils/             # register_modules, set_device, helpers
```

Examples and tests live at the repo root (not inside the package) — see
[`docs/internal/PROJECT_PROPOSAL.md`](docs/internal/PROJECT_PROPOSAL.md) §3 for the full layout.

## Installation

### From PyPI (recommended)

```bash
pip install "tt_symbiote[ttnn]"
```

This pulls `tt_symbiote` plus the pinned `ttnn==0.68.0` wheel (matching
`scripts/ttnn-pin.txt`) plus the transitive deps (`torch`,
`transformers==5.9.0`, `accelerate`, `tokenizers`, ...).

> **System prerequisite.** `ttnn` JIT-compiles firmware kernels at first
> `open_mesh_device(...)` call using the Tenstorrent sfpi RISC-V
> toolchain at `/opt/tenstorrent/sfpi/`. Every `ttnn` wheel pins one
> sfpi version; if the host's installed sfpi doesn't match,
> `open_mesh_device()` fails with `unrecognized command-line option`.
> The `[ttnn]` extra above pins `ttnn==0.68.0` which requires
> `sfpi 7.35.3`. See [`docs/install_prerequisites.md`](docs/install_prerequisites.md)
> for how to install / verify the sfpi toolchain and a (ttnn, sfpi)
> compatibility table.

If you want `tt_symbiote` *without* the bundled ttnn pin (e.g. because
your sfpi is on a different version and you'll install a matching ttnn
manually):

```bash
pip install tt_symbiote
pip install ttnn==<version_matching_your_sfpi>
```

`tracy` (Tenstorrent's profiler) is *not* on PyPI; install it from
tt-metal only if you need performance instrumentation.

### From source (editable, contributors)

```bash
git clone https://github.com/alnah005/tt_symbiote.git
cd tt_symbiote
./scripts/bootstrap_venv.sh     # creates .venv, validates sfpi, pip installs ttnn + tt_symbiote (editable)
source .venv/bin/activate
```

The bootstrap script reads `(ttnn, sfpi)` from
[`scripts/ttnn-pin.txt`](scripts/ttnn-pin.txt), probes the system
[Tenstorrent sfpi toolchain](https://docs.tenstorrent.com/) at
`/opt/tenstorrent/sfpi/`, **refuses to proceed if the versions
disagree** (preventing the silent runtime failure described above),
and then installs `ttnn==<pinned>`, `torch`,
`transformers==5.9.0`, and `tt_symbiote` (editable) into a fresh venv.
No `tt-metal` source checkout is required. See
[`docs/ling_mini_2_0_guide.md`](docs/ling_mini_2_0_guide.md) §B.1 for
the full prerequisites.

## Quick start

```bash
source .venv/bin/activate                           # if you used the bootstrap script
python examples/e2e/run_ling_mini_2_0.py            # T3K
python examples/e2e/resnet/run_resnet50.py          # N150
python examples/e2e/gemma4/run_gemma4_e2b.py        # N150
python examples/e2e/qwen3_vl/run_qwen3_vl_2b.py     # N150
```

## Development

```bash
make install        # editable install + dev extras
make lint           # pre-commit on all files
make test           # capability tests
make smoke MODEL=bailing_moe_v2    # per-model smoke test
```

Requires `ttnn` to be importable in the active Python environment (pulled
in automatically by `pip install "tt_symbiote[ttnn]"` or by
`scripts/bootstrap_venv.sh`). `tracy` is optional — `tt_symbiote.core.run_config`
guards the `from tracy import signpost` import behind a `try/except` and
falls back to a no-op shim when tracy is absent, so the profile-only
signpost hook becomes a no-op unless you set `TT_SYMBIOTE_SIGNPOST_MODE`
in the environment.

`transformers==5.9.0` is the strict pin for this branch; every other dep in
`pyproject.toml` mirrors the specifier used by HF transformers v5.9.0's own
`setup.py`. See `docs/internal/migration_notes.md` for the bump procedure when moving
to a different transformers release.

## Versioning

One `transformers` version per `tt_symbiote` branch. First branch targets
`transformers==5.9.0`. See `docs/internal/PROJECT_PROPOSAL.md` §10 for the branching policy.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
