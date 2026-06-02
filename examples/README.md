# tt_symbiote examples

This directory holds standalone example scripts that exercise the `tt_symbiote`
package. The contents are intentionally **outside the installed package** (per
[`docs/development/PROJECT_PROPOSAL.md`](../docs/development/PROJECT_PROPOSAL.md) P10):

- `pip install tt_symbiote` does **not** ship these files.
- They live in the repository for reference and inspiration only.
- Each subfolder may carry its own extra dependencies. Do not expect
  them to be installed as part of `tt_symbiote`.

## Index

- [`e2e/`](e2e/) — non-interactive, one-shot reproducer scripts for the
  officially supported models, verified end-to-end on real Tenstorrent
  hardware. One file per model variant; the folder's `README.md` is also
  the verified-models tracking table. See [`e2e/README.md`](e2e/README.md).

  Currently covered:
  - `e2e/run_ling_mini_2_0.py` (Ling-mini-2.0, the only causal-LM recipe today)
  - `e2e/gemma4/` (Gemma-4 variants — vision-language)
  - `e2e/qwen3_vl/` (Qwen3-VL variants — vision-language)
  - `e2e/resnet/` (ResNet18/34/50/101/152 — vision classification)

## Running an example

```bash
# One-time bootstrap (sets up a standalone venv with ttnn pinned to the
# version validated against your system sfpi):
./scripts/bootstrap_venv.sh

source .venv/bin/activate
export MESH_DEVICE=T3K   # or N150, N300, ...
python examples/e2e/run_ling_mini_2_0.py
```

See [`examples/e2e/README.md`](e2e/README.md) for the per-model recipe.
