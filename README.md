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
> with `compatibility.report(...)["runtime_observed"]["unexpected"] == []`.
> See [`docs/cpu_vs_device_coverage.md`](docs/cpu_vs_device_coverage.md)
> for the per-variant CPU/device split,
> [`docs/migration_notes.md`](docs/migration_notes.md) for the per-phase
> design rationale, and [`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md) for
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
[`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md) §3 for the full layout.

## Quick start

```bash
git clone https://github.com/alnah005/tt_symbiote.git
cd tt_symbiote
./scripts/bootstrap_venv.sh     # creates .venv, pip installs ttnn + tt_symbiote
source .venv/bin/activate
python examples/e2e/run_ling_mini_2_0.py            # T3K
python examples/e2e/resnet/run_resnet50.py          # N150
python examples/e2e/gemma4/run_gemma4_e2b.py        # N150
python examples/e2e/qwen3_vl/run_qwen3_vl_2b.py     # N150
```

The bootstrap script reads `(ttnn, sfpi)` from
[`scripts/ttnn-pin.txt`](scripts/ttnn-pin.txt), probes the system
[Tenstorrent sfpi toolchain](https://docs.tenstorrent.com/) at
`/opt/tenstorrent/sfpi/`, and installs `ttnn==<pinned>`, `torch`,
`transformers==5.9.0`, and `tt_symbiote` (editable) into a fresh venv.
No `tt-metal` source checkout is required. See
[`docs/ling_mini_2_0_guide.md`](docs/ling_mini_2_0_guide.md) §B.1 for
the full prerequisites.

## Development

```bash
make install        # editable install + dev extras
make lint           # pre-commit on all files
make test           # capability tests
make smoke MODEL=bailing_moe_v2    # per-model smoke test
```

Requires `ttnn` and `tracy` (both tt-metal-built C extensions) to be importable
in the active Python environment. `ttnn` is now available on PyPI (consumed
by `scripts/bootstrap_venv.sh`); `tracy` still ships only via tt-metal and is
not on PyPI. See `PROJECT_PROPOSAL.md` open question OQ-1, which is partly
closed by the bootstrap script.

`transformers==5.9.0` is the strict pin for this branch; every other dep in
`pyproject.toml` mirrors the specifier used by HF transformers v5.9.0's own
`setup.py`. See `docs/migration_notes.md` for the bump procedure when moving
to a different transformers release.

## Versioning

One `transformers` version per `tt_symbiote` branch. First branch targets
`transformers==5.9.0`. See `PROJECT_PROPOSAL.md` §10 for the branching policy.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
