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
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch
    )
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
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch
    )
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
