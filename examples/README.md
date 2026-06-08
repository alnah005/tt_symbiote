# tt_symbiote examples

Standalone scripts that exercise the `tt_symbiote` package. They live **outside
the installed package** (per
[`docs/development/PROJECT_PROPOSAL.md`](../docs/development/PROJECT_PROPOSAL.md) P10):
`pip install tt_symbiote` does not ship them — they are in-repo reference only.

[`e2e/`](e2e/) holds non-interactive, one-shot reproducers for the supported
models, one file per variant, verified on real Tenstorrent hardware. Its
[`README.md`](e2e/README.md) is also the verified-models index and per-model
recipe.

Activate the venv (see [root README → Installation](../README.md#installation))
and run:

```bash
source .venv/bin/activate
export MESH_DEVICE=T3K   # or N150, N300, ...
python examples/e2e/run_ling_mini_2_0.py
```
