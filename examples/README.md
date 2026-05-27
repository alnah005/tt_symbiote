# tt_symbiote examples

This directory holds standalone example scripts that exercise the `tt_symbiote`
package. The contents are intentionally **outside the installed package** (per
[`PROJECT_PROPOSAL.md`](../PROJECT_PROPOSAL.md) P10):

- `pip install tt_symbiote` does **not** ship these files.
- They live in the repository for reference and inspiration only.
- Each subfolder may carry its own extra dependencies (e.g. `gradio`,
  `streamlit`). Do not expect them to be installed as part of `tt_symbiote`.

## Index

- [`chat/`](chat/) — `HF_chat.py`, an interactive chat demo that loads a model
  through `tt_symbiote` and the Hugging Face tokenizer. See
  [`chat/README.md`](chat/README.md).

## Running an example

```bash
cd examples/chat
python HF_chat.py --help
```
