# baidu/Unlimited-OCR e2e demo

End-to-end OCR demo for `baidu/Unlimited-OCR` (a ~3.3B DeepSeek-OCR-style VLM) on
a single Blackhole (P150, mesh `(1,1)`), driving the TTNN
`TTNNUnlimitedOcrForCausalLM` forward.

| Variant | Script | Task | Hardware | Status |
|---|---|---|---|---|
| `baidu/Unlimited-OCR` | [`run_unlimited_ocr.py`](run_unlimited_ocr.py) | OCR (VLM) | P150 (1×1) | ✅ functional (unoptimized prefill decode) |

## Usage

```bash
export TT_METAL_HOME=/home/ttuser/salnahari/tt-metal
PY=$TT_METAL_HOME/python_env/bin/python

# MULTIPLE synthetic sample documents (no image / network needed) -- OCR'd sequentially
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py                    # 4 distinct sample docs
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py --num-samples 8    # 8 distinct sample docs

# multiple real documents (local paths and/or http URLs), OCR'd one after another
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py --image scan.png
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py --images a.png b.png https://host/c.jpg
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py --image-dir ~/scans/

# traced decode (~17 tok/s) + more tokens / custom output dir / prompt
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py --trace --num-samples 4
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py --max-new-tokens 64 --output-dir /tmp/ocr_out
$PY examples/e2e/unlimited_ocr/run_unlimited_ocr.py --prompt "<image>\nFree OCR."
```

The demo handles **multiple images**: `--images`/`--image-dir` OCR every supplied
image, and with no image given it generates `--num-samples` (default 4) DISTINCT
synthetic sample docs. Each `idx` is a genuinely **different document type** whose
big title is the first line — `INVOICE`, `PURCHASE ORDER`, `MEMORANDUM`,
`LEASE AGREEMENT` (cycling with distinct numbers for `--num-samples > 4`) — so
every image OCRs to a **different** output starting from the very first tokens.
Images are processed **SEQUENTIALLY** on the single P150
(mesh `(1,1)`) -- one image at a time through the shared traced pipeline. If one
image fails it is logged and the batch continues. (The
[`dots.ocr` demo](../dots_ocr/run_dots_ocr.py)'s concurrent data-parallel OCR
across multiple devices is **future work** for this model.)

Each run writes one `<stem>.md` (the OCR text) per image into `--output-dir`
(default `examples/e2e/unlimited_ocr/out/`) and a
`run_unlimited_ocr_coverage.json` summary next to the script: model, mesh shape,
per-image tokens / timing / torch-match / text (`results[]`) **plus aggregate
fields** (`num_images`, `num_ocrd`, `total_tokens`, `total_seconds`,
`aggregate_tok_s`, `decode`, `failures[]`). At the end it prints an aggregate line
(total images, total tokens, total wall time, aggregate tok/s).

## What it does

- Loads the real ~3.3B reference on CPU via
  `tt_symbiote.models.unlimited_ocr.reference_loader.load_reference_model`, builds
  `TTNNUnlimitedOcrForCausalLM.from_torch(model)`, and `set_device`s it onto a
  single Blackhole `(1,1)` mesh (`l1_small_size=32768`).
- Preprocesses the image the way the model expects: a **global-only 1024×1024
  view** (`ImageOps.pad` + mean/std=0.5 normalization) and a `"<image>\nFree OCR."`
  prompt whose single `<image>` expands to the **273-token** global-view layout
  (16×16 grid + per-row `image_newline` + `view_seperator`, token id 128815),
  scatter-merged into the text embeddings at the 273 `<image>` positions. This is
  exactly the crop-free base view path the TTNN model implements.
- **Greedy decode is prefill-based**: each step re-runs the whole TTNN VLM forward
  over the growing sequence and takes `argmax` of the last-token logits (O(n²)).
  This is a **functional** demo — there is no optimized KV-cache / trace decode
  pipeline for this model yet; the perf pipeline is future work.
- **Correctness check**: the same preprocessed inputs are also run through a CPU
  torch reference (vision block + masked-scatter + LM + `lm_head`, replicated as in
  the Tier4 `test_forcausallm_vlm_e2e` test, since the reference model's own image
  path uses `.cuda()`), and the first few greedy tokens are asserted to match TTNN
  — proving the demo output is the real model's output, not garbage.

## Dependencies

- `Pillow` — image loading / padding (the demo prints `SKIP` and returns 0 if absent).
- `torchvision` — `ToTensor` / `Normalize`.
- `addict` / `easydict` — pulled in by the reference remote code (already installed
  in the tt-metal venv; `pip install addict easydict` if missing).
- `requests` — only for `http(s)` image URLs.

## Caveats

- Functional bring-up demo; the prefill-each-step decode is O(n²) and is **not** a
  performance baseline. Device time / tokens-per-second here are informational only.
- The vision path clears PCC ≥ 0.99 with last-token argmax-match vs torch (see the
  Tier4 tests); the demo enforces greedy-token agreement with the torch reference.
