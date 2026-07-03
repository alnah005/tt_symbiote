# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier-S2 W1 parity test: forward_logits_* vs the native token path.

The vLLM S2 serving adapter drives ``pipeline.forward_logits_prefill`` /
``forward_logits_decode`` (logits-returning graphs) instead of the native
token-emitting ``prefill`` / ``decode_step``. W1 is correct iff sampling the
returned logits greedily (``argmax``) reproduces the native greedy token path
token-for-token -- otherwise the logits refactor changed OCR output.

This is the guard for that refactor:

  * ``forward_logits_prefill`` returns ``[B, vocab]`` and its argmax == the
    native ``prefill`` first token.
  * ``forward_logits_decode`` (per-row absolute cache positions, TS-3 path) argmax
    == the native ``decode_step`` token, step for step.

Both paths share the same TTNN modules; only the terminal argmax differs, so the
sequences must be identical.

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier4/test_forward_logits_parity.py -x -s \
        --override-ini="addopts="
"""

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline

from ..dots_ocr_helpers import (
    dots_ocr_device_params,
    mesh_num_devices,
    pipeline_batch_size,
    resolve_mesh_device_shape,
    resolve_model_path,
    stack_input_ids_for_dp,
)

DOTS_OCR_LOCAL_PATH = resolve_model_path()

N_TOKENS = 16


def _first_stream(generated):
    return generated[0] if (generated and isinstance(generated[0], list)) else generated


def _row0(logits: torch.Tensor) -> int:
    """Greedy argmax of stream-0's [B, vocab] logits row."""
    assert logits.dim() == 2, f"expected [B, vocab] logits, got {tuple(logits.shape)}"
    return int(logits[0].argmax().item())


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_forward_logits_matches_token_path(mesh_device):
    """argmax(forward_logits_*) reproduces the native greedy token path."""
    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    # The S2 logits graphs must have been built by the factory.
    assert pipeline.graph_prefill_logits is not None
    assert pipeline.graph_decode_logits is not None

    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "What is optical character recognition?"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    ids = stack_input_ids_for_dp(ids)
    prompt_len = int(ids.shape[-1])

    # --- Reference: native token path (greedy on-device argmax) ---------------
    pipeline.warmup(ids)
    ref = _first_stream(pipeline.generate(ids, max_new_tokens=N_TOKENS))
    ttnn.synchronize_device(mesh_device)
    assert len(ref) >= 2 and len(set(ref)) >= 2, f"degenerate reference output: {ref[:8]}"

    # --- Candidate: logits path, sampled greedily on host --------------------
    # Capture the dedicated logits traces, then drive prefill + decode.
    pipeline.warmup(ids, return_logits=True)

    prefill_logits = pipeline.forward_logits_prefill(ids)
    # [B, vocab]: one logits row per DP stream; row 0 sampled below.
    assert prefill_logits.dim() == 2, f"prefill logits must be [B, vocab], got {tuple(prefill_logits.shape)}"
    assert int(prefill_logits.shape[-1]) >= 1000, f"unexpected vocab dim {int(prefill_logits.shape[-1])}"

    cand = [_row0(prefill_logits)]
    prev = [cand[0]] * batch  # one row per DP stream (same prompt -> same token)
    for step in range(N_TOKENS - 1):
        pos = [prompt_len + step] * batch
        dec_logits = pipeline.forward_logits_decode(prev, pos)
        nxt = _row0(dec_logits)
        cand.append(nxt)
        prev = [nxt] * batch
    ttnn.synchronize_device(mesh_device)

    print(f"\n[token path ] {ref}\n[logits path] {cand}\n")

    # Token-for-token parity is the W1 correctness contract.
    div = next((i for i, (a, b) in enumerate(zip(ref, cand)) if a != b), None)
    assert ref == cand, (
        "forward_logits_* greedy output diverges from the native token path at "
        f"index {div}: token={ref[div] if div is not None else '?'} "
        f"logits={cand[div] if div is not None else '?'}"
    )
    pipeline.release()


# 8 distinct topics, one per DP stream. Distinguishing nouns are early in the
# user turn so they survive truncation to a common length.
_DISTINCT_PROMPTS = [
    "What is optical character recognition?",
    "Explain how rivers form and reach the sea.",
    "Describe the planet Mars and its surface.",
    "What causes thunderstorms to develop in summer?",
    "How do honeybees make honey from nectar?",
    "What is a black hole and how does it form?",
    "Explain photosynthesis in green plants briefly.",
    "What is the capital city of France called?",
]


def _stack_distinct_prompts(tok, n: int) -> tuple[torch.Tensor, int]:
    """Tokenize ``n`` distinct prompts and truncate to a common length.

    The dots.ocr prefill graph runs a single ``[B, S]`` tensor, so every DP
    stream must share one sequence length. We truncate to the shortest prompt's
    length (all real causal tokens, no padding) so each stream still holds a
    valid, DISTINCT prefix.
    """
    encs = []
    for p in _DISTINCT_PROMPTS[:n]:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
        encs.append(ids)
    L = min(int(e.shape[-1]) for e in encs)
    rows = [e[:, :L] for e in encs]
    return torch.cat(rows, dim=0).contiguous(), L  # [n, L]


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_forward_logits_distinct_streams(mesh_device):
    """Continuous-batching isolation: N DISTINCT sequences decode independently.

    Drives one different prompt per DP stream (one per mesh device) concurrently
    through ``forward_logits_prefill`` + ``forward_logits_decode``, and asserts
    each stream's greedy continuation matches that stream's native token-path
    output token-for-token. This is the real continuous-batching guard: it proves
    the per-stream page tables (TS-5) and per-row decode positions (TS-6) keep the
    8 sequences ISOLATED -- no collapse to stream 0, no cross-stream KV bleed --
    and that the logits path matches the token path per stream, not just in
    aggregate.
    """
    batch = pipeline_batch_size()
    if batch <= 1 or mesh_num_devices() <= 1:
        pytest.skip("distinct-stream continuous batching requires a DP mesh (batch_size == num_devices > 1)")

    torch.set_grad_enabled(False)
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)

    distinct_ids, prompt_len = _stack_distinct_prompts(tok, batch)
    assert int(distinct_ids.shape[0]) == batch

    # --- Reference: native token path over the SAME distinct streams ----------
    # stop_on_eos=False so every stream emits exactly N_TOKENS (fixed depth,
    # clean per-stream comparison).
    pipeline.warmup(distinct_ids)
    ref = pipeline.generate(distinct_ids, max_new_tokens=N_TOKENS, stop_on_eos=False)
    ttnn.synchronize_device(mesh_device)
    assert isinstance(ref, list) and len(ref) == batch and all(isinstance(r, list) for r in ref)
    # The token path must itself keep the streams distinct (else the parity below
    # could pass trivially with a collapsed reference).
    assert len({tuple(r) for r in ref}) >= 2, f"reference streams collapsed (not distinct): {ref}"

    # --- Candidate: logits path over the SAME distinct streams ----------------
    pipeline.warmup(distinct_ids, return_logits=True)

    prefill_logits = pipeline.forward_logits_prefill(distinct_ids)
    assert prefill_logits.dim() == 2 and int(prefill_logits.shape[0]) == batch
    firsts = [int(prefill_logits[i].argmax().item()) for i in range(batch)]
    cand = [[firsts[i]] for i in range(batch)]
    prev = list(firsts)
    for step in range(N_TOKENS - 1):
        # One absolute position per stream; here all advance together from the
        # shared prefill length (streams admitted at the same step), exercising
        # the per-row position buffer for every stream.
        pos = [prompt_len + step] * batch
        dec_logits = pipeline.forward_logits_decode(prev, pos)
        assert int(dec_logits.shape[0]) == batch
        nxt = [int(dec_logits[i].argmax().item()) for i in range(batch)]
        for i in range(batch):
            cand[i].append(nxt[i])
        prev = nxt
    ttnn.synchronize_device(mesh_device)

    for i in range(batch):
        print(f"[stream {i}] token={ref[i]}\n           logits={cand[i]}")

    mism = [i for i in range(batch) if ref[i] != cand[i]]
    assert not mism, (
        f"forward_logits_decode diverges from the token path on streams {mism} "
        f"(per-stream isolation broken). token={[ref[i] for i in mism]} "
        f"logits={[cand[i] for i in mism]}"
    )
    pipeline.release()


def _long_prompt_ids(tok, min_len: int) -> torch.Tensor:
    """Build a chat-templated prompt with at least ``min_len`` tokens."""
    sentence = "Optical character recognition converts images of text into machine readable characters. "
    content = sentence
    ids = None
    for _ in range(64):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
        if int(ids.shape[-1]) >= min_len:
            break
        content += sentence
    return ids


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_chunked_prefill_matches_single_shot(mesh_device):
    """TS-7: chunked prefill reproduces single-shot prefill token-for-token.

    A prompt long enough to span multiple 256-token chunks is prefilled twice:
    once single-shot (the validated traced path) and once via the eager chunked
    path (offset RoPE + chunk-page-table fill + chunked SDPA over the paged KV).
    The greedy first token of every DP stream must match, proving the chunked
    decoder writes the same paged KV and attends over the prefix correctly.
    """
    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    assert pipeline.graph_prefill_logits is not None

    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    # >256 tokens => at least two chunks (256 + remainder), exercising the
    # absolute-offset RoPE and the chunk page table for chunk_start>0.
    ids = _long_prompt_ids(tok, min_len=320)
    ids = stack_input_ids_for_dp(ids)
    seq_len = int(ids.shape[-1])
    assert seq_len > 256, f"need a multi-chunk prompt, got seq_len={seq_len}"

    # --- Reference: single-shot prefill logits (traced) ----------------------
    pipeline.warmup(ids, return_logits=True)
    single = pipeline.forward_logits_prefill(ids)
    ttnn.synchronize_device(mesh_device)
    assert single.dim() == 2 and int(single.shape[0]) == batch

    # --- Candidate: chunked prefill logits (eager) ---------------------------
    chunked = pipeline.forward_logits_prefill(ids, chunk_size=256)
    ttnn.synchronize_device(mesh_device)
    assert chunked.shape == single.shape

    single_tok = [int(single[i].argmax().item()) for i in range(batch)]
    chunk_tok = [int(chunked[i].argmax().item()) for i in range(batch)]
    print(f"\n[single-shot] {single_tok}\n[chunked    ] {chunk_tok}\n")

    mism = [i for i in range(batch) if single_tok[i] != chunk_tok[i]]
    assert not mism, (
        f"chunked prefill diverges from single-shot on streams {mism}: "
        f"single={[single_tok[i] for i in mism]} chunked={[chunk_tok[i] for i in mism]}"
    )

    # Secondary check: logits stay numerically close. Token parity above is the
    # primary correctness gate; the residual is cross-kernel bf16 noise -- the
    # chunked path uses the paged chunked-SDPA kernel while single-shot uses the
    # in-memory full-causal SDPA, so accumulation order differs over the
    # 152K-wide logits. Greedy argmax matching on every stream bounds the impact.
    cos = torch.nn.functional.cosine_similarity(single.float().flatten(), chunked.float().flatten(), dim=0).item()
    print(f"[chunked-vs-single logits cosine] {cos:.5f}")
    assert cos > 0.98, f"chunked prefill logits cosine too low: {cos}"

    # --- Determinism across chunk counts -------------------------------------
    # chunk_size must be a multiple of 256 (SDPA q_chunk constraint). With a
    # ~320-token prompt, chunk_size=512 collapses to a single eager chunk while
    # chunk_size=256 takes two; matching tokens prove the result is independent
    # of how the prefill is segmented.
    chunked512 = pipeline.forward_logits_prefill(ids, chunk_size=512)
    ttnn.synchronize_device(mesh_device)
    chunk512_tok = [int(chunked512[i].argmax().item()) for i in range(batch)]
    print(f"[chunked-512 ] {chunk512_tok}")
    mism512 = [i for i in range(batch) if single_tok[i] != chunk512_tok[i]]
    assert not mism512, (
        f"chunk_size=512 diverges from single-shot on streams {mism512}: "
        f"single={[single_tok[i] for i in mism512]} chunk512={[chunk512_tok[i] for i in mism512]}"
    )
    assert chunk512_tok == chunk_tok, (
        "chunked prefill is not deterministic across chunk counts: " f"chunk256={chunk_tok} chunk512={chunk512_tok}"
    )
    pipeline.release()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_prefix_cache_suffix_matches_full_prefill(mesh_device):
    """TS-8: prefix-cache-aware prefill computes only the uncached suffix.

    A full chunked prefill populates the paged KV for the whole prompt. A second
    prefill of the SAME prompt with ``prefix_len`` > 0 keeps that KV (conditional
    reset + seed_seq_length at the prefix), recomputes only the suffix, and
    attends over the resident prefix via the chunked SDPA. The last-position
    result must match the full prefill token-for-token -- i.e. skipping the
    cached prefix is correct, not just cheaper.
    """
    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    assert pipeline.graph_prefill_logits is not None

    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    ids = _long_prompt_ids(tok, min_len=320)
    ids = stack_input_ids_for_dp(ids)
    seq_len = int(ids.shape[-1])
    # prefix_len must be block-aligned (block_size=64) and a 256-multiple chunk
    # boundary, with a non-empty suffix to compute.
    prefix_len = 256
    assert seq_len > prefix_len, f"need seq_len>{prefix_len}, got {seq_len}"

    # --- Reference: full chunked prefill populates KV for [0, S) -------------
    full = pipeline.forward_logits_prefill(ids, chunk_size=256)
    ttnn.synchronize_device(mesh_device)
    full_tok = [int(full[i].argmax().item()) for i in range(batch)]

    # --- Candidate: reuse the resident prefix [0, prefix_len), compute [P, S) -
    suffix = pipeline.forward_logits_prefill(ids, chunk_size=256, prefix_len=prefix_len)
    ttnn.synchronize_device(mesh_device)
    suffix_tok = [int(suffix[i].argmax().item()) for i in range(batch)]
    print(f"\n[full-prefill ] {full_tok}\n[suffix-only  ] {suffix_tok}\n")

    mism = [i for i in range(batch) if full_tok[i] != suffix_tok[i]]
    assert not mism, (
        f"prefix-cache suffix prefill diverges from full prefill on streams "
        f"{mism}: full={[full_tok[i] for i in mism]} "
        f"suffix={[suffix_tok[i] for i in mism]}"
    )
    cos = torch.nn.functional.cosine_similarity(full.float().flatten(), suffix.float().flatten(), dim=0).item()
    print(f"[suffix-vs-full logits cosine] {cos:.5f}")
    assert cos > 0.98, f"prefix-cache suffix logits cosine too low: {cos}"
    pipeline.release()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_multigrid_vision_per_stream_matches_single(mesh_device):
    """TS-9: per-request multi-grid vision -> each stream OCRs its own grid.

    Two images with TRANSPOSED grids (e.g. 16x32 vs 32x16 patches) carry the
    same patch/merged-token count but different spatial RoPE, so a batched
    prefill that mixes them per stream must reproduce, for each stream, the
    result that image would get on its own. Equal token counts keep one shared
    prompt length so the last-position logits line up across streams.

    Reference: two same-grid batched prefills (all-A, all-B) via the validated
    ``forward_logits_prefill``; candidate: one mixed prefill via
    ``forward_logits_prefill_multigrid``. Stream b (image A if even else B) must
    match the corresponding reference token.
    """
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor

    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    if batch < 2 or mesh_num_devices() < 2:
        pytest.skip("multi-grid needs a DP batch >= 2 (DOTS_OCR_PARALLELISM=DP on T3K)")

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)

    def _proc(img):
        p = image_processor(images=[img], return_tensors="pt")
        return p["pixel_values"].to(torch.bfloat16), p["image_grid_thw"]

    def _img(w, h):
        arr = np.random.default_rng(w * 7919 + h).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    # 14px patches, merge 2 => use multiples of 28. Transposed sizes => same
    # patch count, different grid (h,w).
    pvA, gA = _proc(_img(32 * 14, 16 * 14))  # ~ (1, 16, 32)
    pvB, gB = _proc(_img(16 * 14, 32 * 14))  # ~ (1, 32, 16)
    if int(gA.prod()) != int(gB.prod()):
        pytest.skip(f"processor resized to unequal patch counts gA={gA.tolist()} gB={gB.tolist()}")
    if bool(torch.equal(gA, gB)):
        pytest.skip("processor collapsed both images to one grid; cannot test multi-grid")

    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=batch,
        batched_vision=True,
    )
    assert pipeline.graph_prefill_logits is not None
    img_tok = int(pipeline.graph_prefill._image_token_id)
    sms = int(pipeline.vision_tower.spatial_merge_size)
    n_merged = int(gA.prod()) // (sms * sms)

    # Shared-length prompt: text prefix + n_merged image tokens + text suffix.
    prefix = tokenizer("Read the image:", add_special_tokens=False)["input_ids"]
    suffix = tokenizer("\nAnswer:", add_special_tokens=False)["input_ids"]
    prompt = prefix + [img_tok] * n_merged + suffix
    base_ids = torch.tensor(prompt, dtype=torch.int64).unsqueeze(0)
    ids = base_ids.repeat(batch, 1)

    def _ref(pv_one, g_one):
        pv_all = torch.cat([pv_one] * batch, dim=0)
        g_all = g_one.repeat(batch, 1)
        out = pipeline.forward_logits_prefill(ids, pixel_values=pv_all, image_grid_thw=g_all)
        ttnn.synchronize_device(mesh_device)
        return out

    refA = _ref(pvA, gA)
    refB = _ref(pvB, gB)
    refA_tok = [int(refA[i].argmax().item()) for i in range(batch)]
    refB_tok = [int(refB[i].argmax().item()) for i in range(batch)]
    # Same image on every stream => every row identical.
    assert len(set(refA_tok)) == 1, f"all-A reference not uniform: {refA_tok}"
    assert len(set(refB_tok)) == 1, f"all-B reference not uniform: {refB_tok}"
    tokA, tokB = refA_tok[0], refB_tok[0]
    # The two grids must actually drive different OCR tokens, else the test is
    # vacuous (RoPE/grid had no effect).
    assert tokA != tokB, f"transposed grids produced identical token {tokA}; test is vacuous"

    # Mixed batch: even streams -> image A, odd streams -> image B.
    pv_mixed = torch.cat([pvA if (b % 2 == 0) else pvB for b in range(batch)], dim=0)
    g_mixed = torch.cat([gA if (b % 2 == 0) else gB for b in range(batch)], dim=0)
    mixed = pipeline.forward_logits_prefill_multigrid(ids, pv_mixed, g_mixed)
    ttnn.synchronize_device(mesh_device)
    mixed_tok = [int(mixed[i].argmax().item()) for i in range(batch)]
    expected = [tokA if (b % 2 == 0) else tokB for b in range(batch)]
    print(f"\n[refA={tokA} refB={tokB}]\n[expected ] {expected}\n[multigrid] {mixed_tok}\n")

    mism = [b for b in range(batch) if mixed_tok[b] != expected[b]]
    assert not mism, (
        f"multi-grid streams {mism} diverge: expected={[expected[b] for b in mism]} "
        f"got={[mixed_tok[b] for b in mism]}"
    )
    pipeline.release()
