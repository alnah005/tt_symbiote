# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end BATCHED OCR demo for ``rednote-hilab/dots.ocr`` on a DP mesh.

OCRs ``dp`` DIFFERENT images CONCURRENTLY -- one per data-parallel stream
(device b processes image b) -- on a (dp, 1) mesh, ``dp`` in {4, 8}. Drives the
optimized ``TTNNDotsOCRPipeline`` directly with ``batched_vision=True`` (the
Auto ``generate`` shim is single-prompt only): the 42-block vision tower runs
per-device on its own image, the result is scatter-merged into that stream's
text, and the 28-layer decoder generates all streams' OCR concurrently via
traced decode.

Every image in a batch is resized to a common patch grid (default (84, 132) ->
11088 patches -> 2772 merged vision tokens -> decoder seq ~2807) so the batch
shares one vision trace. With no ``--images``/``--image-dir`` it generates
same-size synthetic sample documents (see ``sample_images.py``) to fill the
streams. DP=8 -> T3K (8,1) Wormhole; DP=4 -> P150x4 (4,1) Blackhole -- the two
archs the model's ``@run_on_devices`` guard accepts.

Usage::

    python examples/e2e/dots_ocr/run_dots_ocr.py                # DP=8, sample docs
    python examples/e2e/dots_ocr/run_dots_ocr.py --dp 4         # DP=4, sample docs
    python examples/e2e/dots_ocr/run_dots_ocr.py --image-dir ~/scans/
    python examples/e2e/dots_ocr/run_dots_ocr.py --dp 4 --images a.png b.png c.jpg d.tif
    python examples/e2e/dots_ocr/run_dots_ocr.py --max-new-tokens 200 --output-dir /tmp/ocr_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Synthetic sample-doc generation lives in the sibling module (same directory);
# make it importable whether this file is run as a script or loaded by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_images  # noqa: E402

# dots.ocr is validated on T3K (8x1) data-parallel; select that before importing
# tt_symbiote so the recipe builds the DP pipeline at set_device time.
os.environ.setdefault("MESH_DEVICE", "T3K")
os.environ.setdefault("DOTS_OCR_PARALLELISM", "DP")
os.environ.setdefault("TT_SYMBIOTE_RUN_MODE", "TRACED")

import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_symbiote import AutoModelForCausalLM  # noqa: E402

MODEL_ID = "rednote-hilab/dots.ocr"
IMAGE_TOKEN = "<|imgpad|>"
IMAGE_TOKEN_ID = 151665
DEFAULT_PROMPT = "Extract the text from this image."

# The vision tower's tuned SDPA/matmul buckets top out at 11264 patch tokens
# (the 12288 bucket was removed for L1 OOM). The processor's stock max_pixels
# (11,289,600 ≈ 57.6k patches) blows past that and OOMs vision attention, so cap
# it at the bucket ceiling: patches = pixels / patch_size**2, so 11264 * 14**2 =
# 2,207,744 px. (Allows the (84,132)=11088-patch batched target = 2,173,248 px.)
MAX_PIXELS = 2_207_744

# DP-only demo: `dp` independent OCR streams on a (dp, 1) data-parallel mesh.
# The model's @run_on_devices guard accepts N300=(2,1), T3K=(8,1) and P150x4=(4,1),
# so dp 2 -> N300 (Wormhole, 1 board), dp 8 -> T3K (Wormhole), dp 4 -> P150x4 (Blackhole).
_DP_ARCH = {2: "N300", 4: "P150x4", 8: "T3K"}
_IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")


def _resolve_mesh_shape(dp: int) -> tuple[int, int]:
    return (int(dp), 1)


def _mesh_num_devices(shape) -> int:
    return int(shape[0]) * int(shape[1]) if isinstance(shape, (tuple, list)) else int(shape)


def _resolve_image_specs(args, num_devices, cache_dir=None) -> list[str]:
    if args.image_dir:
        d = Path(args.image_dir).expanduser()
        specs = sorted(str(p) for ext in _IMAGE_EXTS for p in d.glob(ext))
        if not specs:
            sys.exit(f"No images ({', '.join(_IMAGE_EXTS)}) found in {d}")
        return specs
    if args.images:
        return args.images
    # Out-of-the-box: generate `num_devices` SAME-SIZE synthetic sample documents
    # (distinct, verifiable text per page) so the run fills every DP stream cleanly
    # (no aspect distortion, no network). User images override.
    n = max(2, min(int(num_devices), 8))
    cdir = cache_dir or (Path(__file__).with_name("out") / "_sample_docs")
    return sample_images.default_batch_specs(n, cdir, _SAMPLE_DOC_SIZE)


# Vision patch geometry (dots.ocr): patch_size 14, spatial_merge 2.
_PATCH_SIZE = 14
# Target patch grid (h, w) every batched image is resized to. (84, 132) ->
# box (1848, 1176) -> 84*132 = 11088 patches (<= 11264 bucket) -> 11088/4 = 2772
TARGET_BATCH_GRID = (84, 132)

# Same-size synthetic sample documents (the default when no images are given),
# rendered at the TARGET_BATCH_GRID box so every page yields grid (1, 84, 132):
# one shared vision trace, no distortion. Landscape (W=1848 > H=1176), grid h<w.
_SAMPLE_DOC_SIZE = (TARGET_BATCH_GRID[1] * _PATCH_SIZE, TARGET_BATCH_GRID[0] * _PATCH_SIZE)  # (1848, 1176)
# Synthetic sample-document generation (the default when no images are given)
# lives in the sibling module ``sample_images.py`` -> default_batch_specs().


def _build_processor():
    """dots.ocr ships a Qwen2.5-VL-style processor; wire it the way the model expects."""
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor, Qwen2_5_VLProcessor

    local = os.environ.get("DOTS_OCR_MODEL_PATH")
    if not (local and os.path.isdir(local)):
        local = snapshot_download(MODEL_ID)
    image_processor = AutoImageProcessor.from_pretrained(local)
    # Cap max_pixels if the processor honors it; in batched mode every image is
    # resized to the common TARGET_BATCH_GRID box (within MAX_PIXELS) regardless.
    try:
        image_processor.max_pixels = MAX_PIXELS
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(local)
    with open(os.path.join(local, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = IMAGE_TOKEN
    processor.image_token_id = IMAGE_TOKEN_ID
    return processor


def _build_inputs(processor, image, prompt):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    from qwen_vl_utils import process_vision_info

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, max_length=4096, return_tensors="pt"
    )


# ---------------------------------------------------------------------------
# Batched DP vision helpers
# ---------------------------------------------------------------------------


def _load_pil_no_resize(spec):
    """Load a PIL image (RGB) without the per-image MAX_PIXELS downscale."""
    from PIL import Image

    if str(spec).startswith(("http://", "https://")):
        import requests

        img = Image.open(requests.get(spec, stream=True, timeout=30).raw).convert("RGB")
        name = str(spec)
    else:
        p = Path(spec).expanduser()
        img = Image.open(p).convert("RGB")
        name = p.name
    return img, name


def _common_target_box() -> tuple[int, int]:
    """The (W, H) box every batched image is resized to -> one shared patch grid.

    Targets ``TARGET_BATCH_GRID`` (h, w) patch counts: W = w*patch_size,
    H = h*patch_size. Both are multiples of _GRID_STEP (patch*merge) so the
    processor's smart_resize is a no-op and yields exactly grid (1, h, w).
    Default (84, 132) -> box (1848, 1176) -> 11088 patches -> 2772 merged vision
    tokens -> decoder seq ~2807 with the text prompt; 2,173,248 px <= MAX_PIXELS.
    """
    h_patches, w_patches = TARGET_BATCH_GRID
    return w_patches * _PATCH_SIZE, h_patches * _PATCH_SIZE  # (W, H)


def _build_batched_inputs(processor, images, prompt, pad_token_id):
    """Build [B,S] input_ids, batched pixel_values, [B,3] grid for B images.

    Every image is resized to a common box so all grids match. Returns
    (input_ids, pixel_values, grid_thw, all_same_grid). Per-image
    input_ids are right-padded to a common S with ``pad_token_id``.
    """
    import torch as _torch

    w, h = _common_target_box()
    per = [_build_inputs(processor, img.resize((w, h)), prompt) for img in images]

    grids = [p["image_grid_thw"][0] for p in per]
    all_same_grid = all(bool(_torch.equal(g, grids[0])) for g in grids)
    if not all_same_grid:
        return None, None, None, False

    id_rows = [p["input_ids"][0] for p in per]
    prompt_lens = [int(r.shape[0]) for r in id_rows]
    s_max = max(prompt_lens)
    padded = []
    for r in id_rows:
        if int(r.shape[0]) < s_max:
            pad = _torch.full((s_max - int(r.shape[0]),), int(pad_token_id), dtype=r.dtype)
            r = _torch.cat([r, pad], dim=0)
        padded.append(r.unsqueeze(0))
    input_ids = _torch.cat(padded, dim=0)

    pixel_values = _torch.cat([p["pixel_values"] for p in per], dim=0).to(_torch.bfloat16)
    grid_thw = _torch.cat([p["image_grid_thw"] for p in per], dim=0)
    return input_ids, pixel_values, grid_thw, True


def _record_result(results, out_dir, name, grid_row, new_ids, elapsed, processor, label):
    """Decode + print + save one image's OCR output (shared by both paths)."""
    text = processor.decode(new_ids, skip_special_tokens=True).strip()
    ntok = len(new_ids)
    print(
        f"\n  {label} {name} | grid {list(int(x) for x in grid_row)} | "
        f"{ntok} tok in {elapsed:.1f}s device time ({ntok / max(elapsed, 1e-9):.1f} tok/s)\n{text}\n"
    )
    stem = Path(name).stem or name
    (out_dir / f"{stem}.md").write_text(text + "\n")
    results.append(
        {
            "image": name,
            "grid_thw": [int(x) for x in grid_row],
            "num_tokens": ntok,
            "seconds": round(elapsed, 2),
            "output_file": str(out_dir / f"{stem}.md"),
        }
    )


def _run_batched(args, specs, out_dir, mesh_device, num_devices, results):
    """Batched DP vision: OCR num_devices DIFFERENT images concurrently (device b
    processes image b). Builds the pipeline directly with batched_vision=True (the
    Auto recipe shim is single-prompt only). Every image is resized to the common
    TARGET_BATCH_GRID box, so all share one vision trace.
    """
    from tt_symbiote.models.dots_ocr.pipeline import TTNNDotsOCRPipeline

    processor = _build_processor()
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", 0) or 0

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    # Drive the TTNN pipeline directly (bypass the single-prompt Auto recipe shim).
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        MODEL_ID,
        mesh_device,
        batch_size=num_devices,
        hf_model=model,
        batched_vision=True,
    )
    model._tt_pipeline = pipeline  # so the finally-block release still works

    # Load every image (no per-image downscale; the batch resize handles bounds).
    loaded = []
    for idx, spec in enumerate(specs):
        try:
            img, name = _load_pil_no_resize(spec)
            loaded.append((img, name))
        except Exception as e:
            print(f"  [{idx + 1}/{len(specs)}] {spec}: load failed ({type(e).__name__}: {e}); skipping")

    # Group into batches of num_devices; pad the last partial batch by repeating
    # the last image, then discard the duplicate outputs.
    for start in range(0, len(loaded), num_devices):
        group = loaded[start : start + num_devices]
        real_n = len(group)
        while len(group) < num_devices:
            group.append(group[-1])  # pad; duplicate output discarded below
        imgs = [g[0] for g in group]
        names = [g[1] for g in group]

        input_ids, pixel_values, grid_thw, ok = _build_batched_inputs(processor, imgs, args.prompt, pad_token_id)
        if not ok:
            # Every image is resized to the same common box, so grids should always
            # match; if they don't, fail loudly rather than silently degrade.
            raise RuntimeError(f"batch @ {start}: image grids differ after resize to a common box")

        t0 = time.perf_counter()
        generated = pipeline.generate(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            max_new_tokens=args.max_new_tokens,
            stop_on_eos=True,
        )
        ttnn.synchronize_device(mesh_device)
        elapsed = time.perf_counter() - t0
        if not (generated and isinstance(generated[0], list)):
            raise RuntimeError("batched pipeline.generate must return List[List[int]] (one per stream)")

        # generated[b] is image b's stream (prompt excluded); EOS already trimmed
        # each stream in generate(). All real_n streams ran CONCURRENTLY in this
        # one batched generate, so `elapsed` is the shared wall for the whole
        # batch -- report it once here, and store an amortized per-image share so
        # the run summary's total ~= true wall (not N x wall).
        batch_tok = sum(len(generated[b]) for b in range(real_n))
        print(
            f"\n  [batch {start + 1}-{start + real_n}] {real_n} images concurrently | "
            f"grid {[int(x) for x in grid_thw[0]]} | {batch_tok} tok in {elapsed:.1f}s wall "
            f"({batch_tok / max(elapsed, 1e-9):.1f} tok/s aggregate across streams)"
        )
        amortized = elapsed / max(real_n, 1)
        for b in range(real_n):  # discard padded duplicates beyond real_n
            new_ids = [int(t) for t in generated[b]]
            _record_result(results, out_dir, names[b], grid_thw[b], new_ids, amortized, processor, f"[{start + b + 1}]")

    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="dots.ocr batched DP-vision OCR demo (DP=4 or DP=8).")
    ap.add_argument("--images", nargs="+", help="image paths and/or http(s) URLs.")
    ap.add_argument("--image-dir", help="directory of images to OCR (sorted by name).")
    ap.add_argument(
        "--dp",
        type=int,
        choices=(2, 4, 8),
        default=8,
        help="data-parallel degree = number of images OCR'd concurrently. "
        "2 -> N300 mesh (2,1); 8 -> T3K mesh (8,1); 4 -> P150x4 mesh (4,1). Default 8.",
    )
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--output-dir", default=str(Path(__file__).with_name("out")))
    args = ap.parse_args()

    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        print("SKIP: this demo needs `qwen_vl_utils` (pip install qwen-vl-utils). See examples/e2e/README.md.")
        return 0

    # Select the DP arch + mesh BEFORE opening the device / set_device so the
    # model's @run_on_devices guard ({T3K:(8,1), P150x4:(4,1)}) matches.
    os.environ["DOTS_OCR_PARALLELISM"] = "DP"
    os.environ["MESH_DEVICE"] = _DP_ARCH[args.dp]
    mesh_shape = _resolve_mesh_shape(args.dp)
    num_devices = _mesh_num_devices(mesh_shape)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve images after num_devices is known so the default fills every stream.
    specs = _resolve_image_specs(args, num_devices=num_devices, cache_dir=out_dir / "_sample_docs")
    print(
        f"dots.ocr batched OCR | DP={num_devices} | mesh {tuple(mesh_shape)} "
        f"({_DP_ARCH[args.dp]}) | {len(specs)} image(s) | {MODEL_ID}"
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape), trace_region_size=300_000_000, num_command_queues=1
    )

    results = []
    model = None
    try:
        model = _run_batched(args, specs, out_dir, mesh_device, num_devices, results)

        if results:
            total = sum(
                r["seconds"] for r in results
            )  # amortized per-image share of the total wall time (not sum of per-image walls)
            tok = sum(r["num_tokens"] for r in results)
            print(
                f"\n{'=' * 60}\ndots.ocr batched DP={num_devices}: {len(results)} image(s), {tok} tokens, "
                f"{total:.1f}s total ({tok / max(total, 1e-9):.1f} tok/s aggregate)\n{'=' * 60}"
            )
        summary = {
            "model": MODEL_ID,
            "mesh_shape": list(mesh_shape),
            "num_devices": num_devices,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "results": results,
        }
        cov = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
        cov.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote run summary to {cov}")
    finally:
        if getattr(model, "_tt_pipeline", None) is not None:
            try:
                model._tt_pipeline.release()
            except Exception:
                pass
        ttnn.close_mesh_device(mesh_device)

    if not results:
        print("No images were successfully OCR'd.")
        return 1
    assert all(
        Path(r["output_file"]).read_text().strip() for r in results
    ), "every image should produce non-empty OCR output"
    print("\nOK: every image produced OCR output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
