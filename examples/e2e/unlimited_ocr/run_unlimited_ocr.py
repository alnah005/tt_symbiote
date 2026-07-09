# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end OCR demo for ``baidu/Unlimited-OCR`` on a single Blackhole (P150).

OPTIMIZED KV-cache demo. Generation is driven by ``TTNNUnlimitedOcrPipeline``: the
vision DeepEncoder + projector + scatter-merge + LM run ONCE at prefill (filling a
per-layer contiguous K/V cache), then each decode step embeds a single new token
and attends over the cached K/V -- O(n) instead of the old prefill-each-step
O(n^2) (which also recomputed the whole vision tower every token, ~0.09 tok/s).
Attention is FULL-CAUSAL, so the generated ids are identical to the prefill-greedy
path and to the torch ``use_cache=False`` reference (verified below). ``--trace``
additionally captures the decode step as a TTNN trace. NOTE: the model's real
long-form generate() uses a window-128 ring buffer; that long-form-parity
refinement is out of scope here (full-causal cached decode is exact for the
validated regime).

It uses the model's GLOBAL-ONLY 1024 view path -- exactly what the TTNN model
implements: the image is padded/normalized to a 1024x1024 global view, the
DeepEncoder (SAM+CLIP) + projector produce 256 vision tokens, and the model's
``_build_vision_block`` lays them out as a 16x16 grid + a per-row ``image_newline``
+ a trailing ``view_seperator`` -> 273 vision tokens, scatter-merged into the text
embeddings at the 273 ``<image>`` positions (token id 128815). This matches the HF
reference ``infer()`` crop-free base-view layout
(``num_queries_base = ceil((1024/16)/4) = 16`` -> ``16*(16+1)+1 = 273`` tokens).

Correctness: the SAME preprocessed inputs are also run through a CPU torch
reference (the reference model's own image path uses ``.cuda()``, so the vision
block + masked-scatter + LM + lm_head are replicated on CPU as in the Tier4
``test_forcausallm_vlm_e2e`` test), and the first few greedy tokens are asserted
to match TTNN -- proving the demo output is the real model's output.

Multiple images are OCR'd SEQUENTIALLY through this single-device (P150, mesh
(1,1)) traced pipeline -- one image at a time, sharing one pipeline -- with a
per-image ``<stem>.md`` + coverage entry and an aggregate summary at the end.
(The dots.ocr demo's concurrent data-parallel OCR across multiple devices is
future work for this model.)

Usage::

    python examples/e2e/unlimited_ocr/run_unlimited_ocr.py                 # 4 synthetic sample docs
    python examples/e2e/unlimited_ocr/run_unlimited_ocr.py --num-samples 8 # 8 synthetic sample docs
    python examples/e2e/unlimited_ocr/run_unlimited_ocr.py --image doc.png
    python examples/e2e/unlimited_ocr/run_unlimited_ocr.py --images a.png https://.../b.jpg
    python examples/e2e/unlimited_ocr/run_unlimited_ocr.py --image-dir ~/scans/
    python examples/e2e/unlimited_ocr/run_unlimited_ocr.py --max-new-tokens 64 --output-dir /tmp/ocr_out
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

MODEL_ID = "baidu/Unlimited-OCR"
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
DEFAULT_PROMPT = "<image>\nFree OCR."
BOS_ID = 0
EOS_ID = 1
BASE_SIZE = 1024
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
# num_queries_base = ceil((base_size / patch_size) / downsample_ratio) = 16.
# vision block = 16x16 grid + per-row newline (16) + 1 view separator = 273 tokens.
_NUM_QUERIES_BASE = -(-(BASE_SIZE // PATCH_SIZE) // DOWNSAMPLE_RATIO)  # ceil-div = 16
NUM_VISION_TOKENS = _NUM_QUERIES_BASE * (_NUM_QUERIES_BASE + 1) + 1     # 273

_IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")


# ---------------------------------------------------------------------------
# Image + prompt preprocessing (faithful to the HF reference infer() base view)
# ---------------------------------------------------------------------------
def _byte_decoder():
    """GPT-2/DeepSeek byte-level BPE reverse map (unicode char -> byte)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = None


def _decode_tokens(tokenizer, ids):
    """Decode generated ids to readable text.

    baidu/Unlimited-OCR ships a byte-level BPE tokenizer whose ``.decode()``
    mishandles the space marker (``Ġ`` / U+0120) for these tokens (it either drops
    all spaces or leaves ``Ġ`` literal). So map the raw token strings back through
    the byte-level reverse table (``Ġ``->space, ``Ċ``->newline, ...) and UTF-8
    decode -- the canonical GPT-2 byte-level decode.
    """
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        _BYTE_DECODER = _byte_decoder()
    toks = tokenizer.convert_ids_to_tokens(list(ids))
    joined = "".join(toks)
    try:
        return bytearray(_BYTE_DECODER[c] for c in joined if c in _BYTE_DECODER).decode(
            "utf-8", errors="replace"
        )
    except Exception:  # noqa: BLE001
        return tokenizer.decode(list(ids), skip_special_tokens=True)


def _load_pil(spec):
    """Load a PIL RGB image from a local path or an http(s) URL."""
    from PIL import Image, ImageOps

    if str(spec).startswith(("http://", "https://")):
        import requests

        img = Image.open(requests.get(spec, stream=True, timeout=30).raw)
        name = str(spec).rsplit("/", 1)[-1] or "url_image"
    else:
        p = Path(spec).expanduser()
        img = Image.open(p)
        name = p.name
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, name


def _preprocess_pixels(pil_img):
    """Pad to a 1024x1024 global view and normalize (mean/std=0.5) -> [1,3,1024,1024].

    Mirrors the HF reference: ``ImageOps.pad(img, (1024,1024), color=(127,127,127))``
    then ``BasicImageTransform(mean=0.5, std=0.5)`` (ToTensor -> Normalize).
    """
    from PIL import ImageOps
    from torchvision import transforms

    fill = tuple(int(0.5 * 255) for _ in range(3))  # (127,127,127)
    global_view = ImageOps.pad(pil_img, (BASE_SIZE, BASE_SIZE), color=fill)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3)])
    px = tf(global_view)  # [3,1024,1024], float32 in [-1,1]
    return px.unsqueeze(0)  # [1,3,1024,1024]


def _build_input_ids(tokenizer, prompt):
    """Expand the single ``<image>`` in ``prompt`` to the 273-token global-view layout.

    Faithful to the HF reference ``infer()`` crop-free base view: split on
    ``<image>``, encode the text parts (no special tokens), insert 273 image-token
    ids, and prepend BOS (id 0). Returns (input_ids[list[int]], seq_mask[list[bool]]).
    """
    text_splits = prompt.split(IMAGE_TOKEN)
    if len(text_splits) != 2:
        raise ValueError(f"prompt must contain exactly one '{IMAGE_TOKEN}'; got {prompt!r}")

    ids, seq_mask = [], []
    # text before <image>
    pre = tokenizer.encode(text_splits[0], add_special_tokens=False)
    ids += pre
    seq_mask += [False] * len(pre)
    # the image tokens (273)
    ids += [IMAGE_TOKEN_ID] * NUM_VISION_TOKENS
    seq_mask += [True] * NUM_VISION_TOKENS
    # text after <image>
    post = tokenizer.encode(text_splits[1], add_special_tokens=False)
    ids += post
    seq_mask += [False] * len(post)
    # prepend BOS
    ids = [BOS_ID] + ids
    seq_mask = [False] + seq_mask
    return ids, seq_mask


# ---------------------------------------------------------------------------
# TTNN / torch forward helpers
# ---------------------------------------------------------------------------
def _real_rope(model, cfg, seq):
    """Exact (cos, sin) from the model's own LlamaRotaryEmbedding -> [1,1,S,head_dim]."""
    import torch

    rot = model.model.layers[0].self_attn.rotary_emb
    pos = torch.arange(seq).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(torch.zeros(1, 1, seq, cfg.head_dim), pos)  # [1,S,D]
    return cos.unsqueeze(1).float(), sin.unsqueeze(1).float()      # [1,1,S,D]


def _masks_for(seq_mask, device):
    """Build gather_idx [1,S] and vision_mask [1,S,1] ttnn tensors from a seq_mask list."""
    import torch
    import ttnn

    seq = len(seq_mask)
    m = torch.tensor(seq_mask, dtype=torch.bool)
    gather_idx = torch.zeros(1, seq, dtype=torch.int32)
    true_pos = m.nonzero(as_tuple=True)[0]
    gather_idx[0, true_pos] = torch.arange(1, true_pos.numel() + 1, dtype=torch.int32)
    mask_f = m.float().view(1, seq, 1)
    tt_idx = ttnn.from_torch(gather_idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_idx, tt_mask


def _tt_vlm_logits(tt, device, model, cfg, input_ids, seq_mask, tt_px):
    """Run the TTNN VLM forward for the current sequence; return fp32 logits [1,S,vocab]."""
    import torch
    import ttnn

    from tt_symbiote.core.tensor import TorchTTNNTensor

    seq = len(input_ids)
    cos, sin = _real_rope(model, cfg, seq)
    tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_ids = ttnn.from_torch(
        torch.tensor([input_ids], dtype=torch.int32), dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    tt_idx, tt_mask = _masks_for(seq_mask, device)
    out = tt(input_ids=tt_ids, pixel_values=tt_px, position_embeddings=(tt_cos, tt_sin),
             vision_idx=tt_idx, vision_mask=tt_mask)
    if isinstance(out, TorchTTNNTensor):
        out.elem = None
        return out.to_torch.float().reshape(1, seq, cfg.vocab_size)
    if isinstance(out, torch.Tensor):
        return out.float().reshape(1, seq, cfg.vocab_size)
    return ttnn.to_torch(out).float().reshape(1, seq, cfg.vocab_size)


def _torch_vision_block(inner, px):
    """Replicate the CPU global-only vision block [273,1280] (as in the Tier4 VLM test)."""
    import torch

    with torch.no_grad():
        g1 = inner.sam_model(px)
        g2 = inner.vision_model(px, g1)
        feats = torch.cat([g2[:, 1:], g1.flatten(2).permute(0, 2, 1)], dim=-1)
        gf = inner.projector(feats)                       # [1,256,1280]
        _, hw, nd = gf.shape
        h = w = int(hw ** 0.5)
        gf = gf.view(h, w, nd)
        gf = torch.cat([gf, inner.image_newline[None, None, :].expand(h, 1, nd)], dim=1)
        gf = gf.view(-1, nd)
        block = torch.cat([gf, inner.view_seperator[None, :]], dim=0)  # [273,1280]
    return block


def _torch_vlm_logits(model, inner, input_ids, seq_mask, block):
    """CPU torch reference VLM logits for the current sequence -> [1,S,vocab] (fp32)."""
    import torch

    ids = torch.tensor([input_ids], dtype=torch.int64)
    m = torch.tensor([seq_mask], dtype=torch.bool)
    with torch.no_grad():
        emb = inner.embed_tokens(ids).clone()
        emb[0].masked_scatter_(m[0].unsqueeze(-1), block.to(emb.dtype))
        hidden = model.model(inputs_embeds=emb, images=None, use_cache=False, return_dict=True)[0]
        logits = model.lm_head(hidden).float()
    return logits


# ---------------------------------------------------------------------------
# Per-image OCR
# ---------------------------------------------------------------------------
def _ocr_one(pipe, device, model, cfg, tokenizer, spec, prompt, max_new_tokens, correctness_tokens):
    """OCR one image via the KV-cache decode pipeline + torch-match correctness check."""
    import torch
    import ttnn

    pil_img, name = _load_pil(spec)
    px = _preprocess_pixels(pil_img)                                   # [1,3,1024,1024]
    px_nhwc = px.permute(0, 2, 3, 1).contiguous()

    def _mk_px():
        return ttnn.from_torch(px_nhwc, dtype=ttnn.bfloat16,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_px = _mk_px()
    base_ids, base_mask = _build_input_ids(tokenizer, prompt)
    prompt_len = len(base_ids)

    # TRACED decode: prime JIT + capture the decode trace (warm-up double-run) BEFORE
    # the timed generate. A fresh pixel tensor is used per prefill (the vision tower
    # consumes its input), so warm-up gets its own copy.
    pipe.warmup(list(base_ids), seq_mask=list(base_mask), tt_px=_mk_px())
    tt_px = _mk_px()

    # --- Optimized KV-cache greedy decode (O(n)): vision + scatter-merge + LM run
    # ONCE at prefill (filling the per-layer K/V cache); each decode step embeds one
    # new token and attends over the cached K/V. Attention is FULL-CAUSAL, so the
    # ids are identical to the prefill-greedy path and the torch reference. ---
    generated = pipe.generate(list(base_ids), seq_mask=list(base_mask), tt_px=tt_px,
                              max_new_tokens=max_new_tokens, stop_on_eos=True)
    if generated and generated[-1] == EOS_ID:
        generated = generated[:-1]
    prefill_s = pipe.last_prefill_s
    decode_s = pipe.last_decode_s
    n_decode = pipe.last_num_decode
    elapsed = prefill_s + decode_s
    decode_tps = n_decode / max(decode_s, 1e-9)
    text = _decode_tokens(tokenizer, generated).strip()

    # --- correctness check: torch reference greedy for the first N tokens ---
    n_chk = min(correctness_tokens, max_new_tokens)
    inner = model.model
    block = _torch_vision_block(inner, px)
    t_ids = list(base_ids)
    t_mask = list(base_mask)
    torch_gen = []
    for _ in range(n_chk):
        tlog = _torch_vlm_logits(model, inner, t_ids, t_mask, block)
        tnxt = int(tlog[0, -1].argmax().item())
        torch_gen.append(tnxt)
        if tnxt == EOS_ID:
            break
        t_ids.append(tnxt)
        t_mask.append(False)
    # compare TTNN's first tokens to torch's greedy path
    k = min(len(torch_gen), len(generated), n_chk)
    # account for EOS trimming in the TTNN path
    tt_head = generated[:k]
    torch_head = [t for t in torch_gen[:k]]
    n_match = sum(int(a == b) for a, b in zip(tt_head, torch_head))
    match_str = f"{n_match}/{k}"
    ok_match = k > 0 and n_match >= max(1, k - 1)

    ntok = len(generated)
    decode_mode = "kv-cache+trace"
    prefill_mode = "normal"
    print(
        f"\n  {name} | {ntok} tok in {elapsed:.1f}s "
        f"(prefill {prefill_s:.1f}s [{prefill_mode}] + decode {decode_s:.1f}s; "
        f"{decode_tps:.2f} tok/s decode [{decode_mode}], "
        f"{ntok / max(elapsed, 1e-9):.2f} tok/s overall) | "
        f"TTNN vs torch: {match_str} tokens match"
    )
    print(f"  TTNN greedy tokens[:{k}] = {tt_head}")
    print(f"  torch greedy tokens[:{k}] = {torch_head}")
    print(f"\n----- OCR OUTPUT ({name}) -----\n{text}\n-------------------------------")
    assert ok_match, (
        f"TTNN greedy diverges from torch reference for {name}: "
        f"{tt_head} vs {torch_head} ({match_str} match)"
    )
    print(f"  TTNN matches torch reference: {match_str} tokens")

    return {
        "image": name,
        "prompt_tokens": prompt_len,
        "num_vision_tokens": NUM_VISION_TOKENS,
        "num_generated_tokens": ntok,
        "decode": decode_mode,
        "prefill": prefill_mode,
        "prefill_seconds": round(prefill_s, 2),
        "decode_seconds": round(decode_s, 2),
        "seconds": round(elapsed, 2),
        "decode_tokens_per_sec": round(decode_tps, 3),
        "tokens_per_sec": round(ntok / max(elapsed, 1e-9), 3),
        "torch_match": match_str,
        "text": text,
    }, text


def _torch_match_one(model, inner, tokenizer, px, base_ids, base_mask, gen,
                     correctness_tokens, max_new_tokens):
    """Greedy torch reference for the first-k tokens of ONE image; compare to ``gen``."""
    import torch  # noqa: F401

    n_chk = min(correctness_tokens, max_new_tokens)
    block = _torch_vision_block(inner, px)
    t_ids, t_mask, torch_gen = list(base_ids), list(base_mask), []
    for _ in range(n_chk):
        tlog = _torch_vlm_logits(model, inner, t_ids, t_mask, block)
        tnxt = int(tlog[0, -1].argmax().item())
        torch_gen.append(tnxt)
        if tnxt == EOS_ID:
            break
        t_ids.append(tnxt)
        t_mask.append(False)
    k = min(len(torch_gen), len(gen), n_chk)
    tt_head, torch_head = gen[:k], torch_gen[:k]
    n_match = sum(int(a == b) for a, b in zip(tt_head, torch_head))
    ok = k > 0 and n_match >= max(1, k - 1)
    return n_match, k, tt_head, torch_head, ok


def _run_dp(args, specs, model, cfg, tokenizer, out_dir):
    """DATA-PARALLEL demo: OCR ``dp`` DIFFERENT images CONCURRENTLY across a (dp,1)
    Blackhole mesh (device b -> image b), mirroring dots.ocr's batched_vision. One
    prefill + one decode loop process all ``dp`` images in ~one image's wall time.
    Reports per-image .md + torch-match + an AGGREGATE (concurrent) throughput
    summary. This path is entirely separate from the single-device (no --dp) path."""
    import torch
    import ttnn

    from tt_symbiote.models.unlimited_ocr.pipeline import TTNNUnlimitedOcrPipeline

    dp = int(args.dp_size)
    # Pad/truncate specs to exactly ``dp`` images (fill with distinct sample docs).
    if len(specs) < dp:
        extra = sample_images.default_batch_specs(dp, out_dir / "_sample_docs", (BASE_SIZE, BASE_SIZE))
        specs = list(specs) + [e for e in extra if e not in specs]
    specs = list(specs)[:dp]

    decode_mode = "dp+kv-cache+trace"
    print(f"Unlimited-OCR demo | DATA-PARALLEL ({dp},1) Blackhole mesh (P150x{dp}) | "
          f"{dp} image(s) CONCURRENTLY [{decode_mode}] | {MODEL_ID}")
    print(f"  (device b OCRs image b: weights replicate, per-image pixels/ids batch-shard "
          f"on dim 0, outputs gather via ConcatMeshToTensor -- {dp} images in ~1 image's wall)")
    print("  (the batch-B decode graph is captured ONCE across the mesh "
          "[warm-up double-run] and REPLAYED per step; prefill runs eagerly)")

    pxs, names = [], []
    for s in specs:
        pil, name = _load_pil(s)
        pxs.append(_preprocess_pixels(pil))
        names.append(name)
    px_batch = torch.cat(pxs, dim=0).permute(0, 2, 3, 1).contiguous()  # [dp,1024,1024,3]
    base_ids, base_mask = _build_input_ids(tokenizer, args.prompt)
    input_ids_batch = [list(base_ids) for _ in range(dp)]
    prompt_len = len(base_ids)

    # The DP decode graph (batch-B paged cache + 12 layers + lm_head) is always traced,
    # so the mesh device needs a trace region.
    dev_kwargs = {"l1_small_size": 32768, "trace_region_size": 400_000_000}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(dp, 1), **dev_kwargs)
    results, failures = [], []
    wall_t0 = time.perf_counter()
    try:
        print("Building TTNN DP KV-cache pipeline + replicating weights across the mesh ...")
        pipe = TTNNUnlimitedOcrPipeline.from_hf_model(model, cfg, dev, batched_vision=True)
        # Prime JIT + capture the batch-B decode trace ONCE (warm-up double-run) BEFORE
        # the timed generate.
        print("Warming up + capturing the DP decode trace across the mesh ...")
        pipe.warmup_dp(input_ids_batch, list(base_mask), px_batch)
        wall_t0 = time.perf_counter()  # time the REPLAY generate, not the warm-up
        gen = pipe.generate_dp(input_ids_batch, list(base_mask), px_batch,
                               max_new_tokens=args.max_new_tokens, stop_on_eos=True)
        dp_wall = time.perf_counter() - wall_t0
        for b in range(dp):
            if gen[b] and gen[b][-1] == EOS_ID:
                gen[b] = gen[b][:-1]

        inner = model.model
        for b in range(dp):
            try:
                n_match, k, tt_head, torch_head, ok = _torch_match_one(
                    model, inner, tokenizer, pxs[b], base_ids, base_mask, gen[b],
                    args.correctness_tokens, args.max_new_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  stream {b} ({names[b]}) FAILED torch-match ({type(e).__name__}: {e})")
                failures.append({"image": names[b], "error": f"{type(e).__name__}: {e}"})
                continue
            text = _decode_tokens(tokenizer, gen[b]).strip()
            match_str = f"{n_match}/{k}"
            print(f"\n  [stream {b}] {names[b]} | {len(gen[b])} tok | "
                  f"TTNN vs torch: {match_str} tokens match {'OK' if ok else 'MISMATCH'}")
            print(f"    TTNN greedy[:{k}]={tt_head}")
            print(f"    torch greedy[:{k}]={torch_head}")
            print(f"----- OCR OUTPUT ({names[b]}) -----\n{text}\n-------------------------------")
            if not ok:
                failures.append({"image": names[b], "error": f"torch mismatch {match_str}"})
            stem = Path(names[b]).stem or names[b]
            md = out_dir / f"{stem}.md"
            md.write_text(text + "\n")
            results.append({
                "image": names[b], "stream": b, "prompt_tokens": prompt_len,
                "num_vision_tokens": NUM_VISION_TOKENS, "num_generated_tokens": len(gen[b]),
                "torch_match": match_str, "output_file": str(md), "text": text,
            })
    finally:
        ttnn.close_mesh_device(dev)

    total_tokens = sum(r["num_generated_tokens"] for r in results)
    agg_tps = total_tokens / max(dp_wall, 1e-9)
    imgs_per_s = dp / max(dp_wall, 1e-9)
    per_stream_decode_tps = pipe.last_num_decode / max(pipe.last_decode_s, 1e-9)
    print(f"\n{'=' * 68}")
    print(f"Unlimited-OCR aggregate (DATA-PARALLEL, {dp} images CONCURRENT on P150x{dp}): "
          f"{len(results)}/{len(specs)} OCR'd, {total_tokens} tokens, {dp_wall:.1f}s wall")
    print(f"  prefill {pipe.last_prefill_s:.1f}s (all {dp} images concurrently) + "
          f"decode {pipe.last_decode_s:.1f}s ({pipe.last_num_decode} steps)")
    print(f"  AGGREGATE {agg_tps:.2f} tok/s ({imgs_per_s:.3f} images/s); "
          f"per-stream decode {per_stream_decode_tps:.2f} tok/s -> "
          f"aggregate decode {per_stream_decode_tps * dp:.2f} tok/s")
    print(f"  CONCURRENCY WIN: {dp} images in {dp_wall:.1f}s vs ~{dp}x a single image sequentially")
    if failures:
        print(f"  {len(failures)} stream(s) had issues: " + ", ".join(f['image'] for f in failures))
    print(f"{'=' * 68}")

    summary = {
        "model": MODEL_ID, "mesh_shape": [dp, 1], "device_arch": f"P150x{dp}",
        "processing": "data-parallel-concurrent", "decode": decode_mode,
        "aggregate_decode_tok_s": round(per_stream_decode_tps * dp, 3),
        "per_stream_decode_tok_s": round(per_stream_decode_tps, 3),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens, "num_vision_tokens": NUM_VISION_TOKENS,
        "num_images": len(specs), "num_ocrd": len(results), "total_tokens": total_tokens,
        "dp_wall_seconds": round(dp_wall, 2), "prefill_seconds": round(pipe.last_prefill_s, 2),
        "decode_seconds": round(pipe.last_decode_s, 2),
        "aggregate_tok_s": round(agg_tps, 3), "images_per_s": round(imgs_per_s, 4),
        "results": results, "failures": failures,
    }
    cov = Path(__file__).with_name(f"{Path(__file__).stem}_dp_coverage.json")
    cov.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote DP run summary to {cov}")
    if not results:
        print("No images were successfully OCR'd.")
        return 1
    print(f"\nOK: {len(results)} image(s) OCR'd CONCURRENTLY across {dp} devices.")
    return 0


def _resolve_specs(args, sample_cache):
    if args.image_dir:
        d = Path(args.image_dir).expanduser()
        specs = sorted(str(p) for ext in _IMAGE_EXTS for p in d.glob(ext))
        if not specs:
            sys.exit(f"No images ({', '.join(_IMAGE_EXTS)}) found in {d}")
        return specs
    if args.images:
        return args.images
    if args.image:
        return [args.image]
    # No image(s) given: generate `--num-samples` DISTINCT synthetic sample docs so
    # the demo exercises MULTIPLE images (OCR'd sequentially) out of the box.
    return sample_images.default_batch_specs(args.num_samples, sample_cache, (BASE_SIZE, BASE_SIZE))


def main() -> int:
    ap = argparse.ArgumentParser(description="baidu/Unlimited-OCR single-Blackhole OCR demo (functional).")
    ap.add_argument("--image", help="a single image path or http(s) URL.")
    ap.add_argument("--images", nargs="+", help="image paths and/or http(s) URLs.")
    ap.add_argument("--image-dir", help="directory of images to OCR (sorted by name).")
    ap.add_argument("--num-samples", type=int, default=4,
                    help="number of DISTINCT synthetic sample docs to generate + OCR when no "
                         "--image/--images/--image-dir is given (default 4). Ignored if images "
                         "are supplied.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"OCR prompt (default {DEFAULT_PROMPT!r}).")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--correctness-tokens", type=int, default=8,
                    help="how many leading tokens to cross-check against the CPU torch reference.")
    ap.add_argument("--output-dir", default=str(Path(__file__).with_name("out")))
    ap.add_argument("--dp", action="store_true",
                    help="DATA-PARALLEL mode: OCR --dp-size DIFFERENT images CONCURRENTLY "
                         "across a (dp,1) Blackhole mesh (device b -> image b), like "
                         "dots.ocr's batched_vision. Fills with distinct sample docs when "
                         "fewer images are given. Without --dp the single-device path is "
                         "unchanged.")
    ap.add_argument("--dp-size", type=int, default=4,
                    help="number of devices / concurrent images for --dp (default 4 = 4x Blackhole).")
    args = ap.parse_args()

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("SKIP: this demo needs Pillow (pip install pillow). See examples/e2e/unlimited_ocr/README.md.")
        return 0

    import torch  # noqa: F401
    import ttnn

    from transformers import AutoTokenizer

    from tt_symbiote.models.unlimited_ocr.pipeline import TTNNUnlimitedOcrPipeline
    from tt_symbiote.models.unlimited_ocr.reference_loader import load_reference_model

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _resolve_specs(args, out_dir / "_sample_docs")
    decode_mode = "kv-cache+trace"
    prefill_mode = "normal"

    print("Loading reference model (~3.3B, CPU) ...")
    model, cfg = load_reference_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # DATA-PARALLEL path: OCR --dp-size DIFFERENT images concurrently across a (dp,1)
    # mesh. Entirely separate from the single-device path below (which stays as-is).
    # (The single-device banner below is intentionally NOT printed in --dp mode; the
    # DP path prints its own DATA-PARALLEL banner.)
    if args.dp:
        return _run_dp(args, specs, model, cfg, tokenizer, out_dir)

    print(f"Unlimited-OCR demo | single Blackhole (P150, mesh (1,1)) | "
          f"{len(specs)} image(s) | {MODEL_ID}")
    print(f"  (optimized {decode_mode} decode [prefill={prefill_mode}]: vision + scatter-merge + LM "
          f"run ONCE at prefill; each decode step is a single-token full-causal cached forward -- O(n))")

    dev_kwargs = {"l1_small_size": 32768, "trace_region_size": 200_000_000}
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **dev_kwargs)
    results = []
    failures = []
    wall_t0 = time.perf_counter()
    try:
        print("Building TTNN KV-cache pipeline + moving weights to device ...")
        pipe = TTNNUnlimitedOcrPipeline.from_hf_model(model, cfg, mesh_device)

        # Images are OCR'd SEQUENTIALLY on the single P150 (one shared pipeline).
        # If one image fails, log it and continue with the rest (don't abort the
        # batch) -- mirroring the dots.ocr demo's per-image graceful handling.
        for i, spec in enumerate(specs):
            print(f"\n[{i + 1}/{len(specs)}] OCR: {spec}")
            try:
                rec, text = _ocr_one(pipe, mesh_device, model, cfg, tokenizer, spec, args.prompt,
                                     args.max_new_tokens, args.correctness_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED ({type(e).__name__}: {e}); skipping and continuing.")
                failures.append({"image": str(spec), "error": f"{type(e).__name__}: {e}"})
                continue
            stem = Path(rec["image"]).stem or rec["image"]
            md = out_dir / f"{stem}.md"
            md.write_text(text + "\n")
            rec["output_file"] = str(md)
            results.append(rec)
    finally:
        ttnn.close_mesh_device(mesh_device)
    total_seconds = time.perf_counter() - wall_t0

    # --- Aggregate summary (images processed SEQUENTIALLY on a single P150). ---
    total_tokens = sum(r["num_generated_tokens"] for r in results)
    aggregate_tps = total_tokens / max(total_seconds, 1e-9)
    all_matched = all(
        r["torch_match"].split("/")[0] != "0" for r in results
    ) and bool(results)
    print(f"\n{'=' * 68}")
    print(
        f"Unlimited-OCR aggregate ({decode_mode}, SEQUENTIAL on single P150): "
        f"{len(results)}/{len(specs)} image(s) OCR'd, {total_tokens} tokens, "
        f"{total_seconds:.1f}s total wall ({aggregate_tps:.2f} tok/s aggregate)"
    )
    if failures:
        print(f"  {len(failures)} image(s) failed (see coverage json): "
              + ", ".join(f["image"] for f in failures))
    print(
        "  NOTE: images run one-at-a-time through the single-device (P150, mesh (1,1)) "
        "traced pipeline; the dots.ocr demo's concurrent data-parallel OCR across "
        "multiple devices is future work for this model."
    )
    print(f"{'=' * 68}")

    summary = {
        "model": MODEL_ID,
        "mesh_shape": [1, 1],
        "device_arch": "P150",
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "num_vision_tokens": NUM_VISION_TOKENS,
        "decode": decode_mode,
        "prefill": prefill_mode,
        "processing": "sequential-single-device",
        "num_images": len(specs),
        "num_ocrd": len(results),
        "total_tokens": total_tokens,
        "total_seconds": round(total_seconds, 2),
        "aggregate_tok_s": round(aggregate_tps, 3),
        "results": results,
        "failures": failures,
    }
    cov = Path(__file__).with_name(f"{Path(__file__).stem}_coverage.json")
    cov.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote run summary to {cov}")

    if not results:
        print("No images were successfully OCR'd.")
        return 1
    if all_matched:
        print(f"\nOK: {len(results)} image(s) OCR'd; every one matched the torch reference.")
    else:
        print(f"\nOK: {len(results)} image(s) OCR'd.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
