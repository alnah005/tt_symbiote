# dots.ocr batched OCR demo

Runs `rednote-hilab/dots.ocr` on a Tenstorrent data-parallel mesh, OCR-ing **`dp`
different images at once** — one per stream (device `b` → image `b`). DP = 2, 4 or 8.

## Run

```bash
pip install qwen-vl-utils                              # one-time, processor dep

python examples/e2e/dots_ocr/run_dots_ocr.py           # DP=8, 8 sample docs
python examples/e2e/dots_ocr/run_dots_ocr.py --dp 4    # DP=4, 4 sample docs
python examples/e2e/dots_ocr/run_dots_ocr.py --image-dir ~/scans/   # your images
python examples/e2e/dots_ocr/run_dots_ocr.py --images a.png b.jpg
```

`--dp 2` → N300 `(2,1)`; `--dp 8` → T3K `(8,1)`; `--dp 4` → P150x4 `(4,1)` (the mesh/arch are set automatically). Other flags: `--prompt`, `--max-new-tokens` (default 256), `--output-dir`. Images beyond `dp` run in successive batches.

## Output

Per-image OCR is printed and saved to `out/<image>.md`; a run summary goes to
`run_dots_ocr_coverage.json`. With no `--images`/`--image-dir`, `dp` synthetic
legal sample docs ([`sample_images.py`](sample_images.py)) fill the streams.

## Notes

- Every image in a batch is resized to one common grid (default `(1, 84, 132)` ≈
  2807 decoder tokens) so they share a vision trace — batch similar-shaped pages.
- Drives `TTNNDotsOCRPipeline` with `batched_vision=True` (the Auto `generate`
  shim is single-prompt only). Needs `qwen_vl_utils`; skips cleanly if absent.
- Verified on T3K (8×1 DP): 8 distinct docs OCR'd concurrently, correct per stream.
