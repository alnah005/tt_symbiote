# tt_symbiote

Hugging Face of Tenstorrent: a pip-installable Python library whose user-facing API
mirrors [`transformers`](https://github.com/huggingface/transformers) and whose
folder layout mirrors `transformers/src/transformers/`. Provides TTNN-accelerated
implementations of HF model architectures.

> **Status: Phase 1/2 bootstrap.** API target is below; nothing runs end-to-end yet.
> See [`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md) for the full design, phased delivery
> plan, and open questions.

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

## Development

```bash
make install        # editable install + dev extras
make lint           # pre-commit on all files
make test           # capability tests
make smoke MODEL=bailing_moe_v2    # per-model smoke test
```

Requires `ttnn` to be importable in the active Python environment (currently
not declared as a pip dep — see `PROJECT_PROPOSAL.md` open question OQ-1).

## Versioning

One `transformers` version per `tt_symbiote` branch. First branch targets
`transformers==5.9.0`. See `PROJECT_PROPOSAL.md` §10 for the branching policy.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
