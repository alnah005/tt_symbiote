# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 traced-execution test for dots.ocr.

dots.ocr does NOT use the framework ``TT_SYMBIOTE_RUN_MODE=TRACED`` dispatch
(that path is for register_modules / Auto-API models). Instead the bespoke
``TTNNDotsOCRPipeline`` manages its OWN trace lifecycle in NORMAL mode:
``warmup()`` runs the model once eagerly then CAPTURES the on-device
``graph_prefill`` / ``graph_decode`` traces, and ``generate()`` REPLAYS the
captured decode graph each step. (Running the pipeline under the framework
TRACED mode double-traces and ``TT_FATAL: Writes are not supported during trace
capture``.)

This test validates that native trace capture/replay is correct and
deterministic: two independent ``generate`` calls over the same prompt must
produce the IDENTICAL token sequence (greedy on-device argmax + faithful trace
replay => bit-identical), and the output must be coherent (non-degenerate).
A corrupted/freed trace buffer would surface as a divergence, a crash, or
garbage tokens.

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier4/test_traced_dots_ocr.py -x -s \
        --override-ini="addopts="
"""

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline

from ..dots_ocr_helpers import (
    dots_ocr_device_params,
    pipeline_batch_size,
    resolve_mesh_device_shape,
    resolve_model_path,
    stack_input_ids_for_dp,
)

DOTS_OCR_LOCAL_PATH = resolve_model_path()


def _first_stream(generated):
    return generated[0] if (generated and isinstance(generated[0], list)) else generated


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_traced_replay_determinism(mesh_device):
    """Pipeline-native trace capture/replay: two generates must match bit-for-bit."""
    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "What is optical character recognition?"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    ids = stack_input_ids_for_dp(ids)

    # warmup() captures graph_prefill + graph_decode traces.
    pipeline.warmup(ids)

    # Two independent replays of the captured decode graph.
    gen_a = _first_stream(pipeline.generate(ids, max_new_tokens=32))
    gen_b = _first_stream(pipeline.generate(ids, max_new_tokens=32))
    ttnn.synchronize_device(mesh_device)

    text_a = tok.decode(gen_a, skip_special_tokens=True)
    print(
        f"\n[traced replay A] {len(gen_a)} tok: {text_a!r}\n[traced replay B] {tok.decode(gen_b, skip_special_tokens=True)!r}\n"
    )

    # Trace was captured + replays cleanly (non-degenerate, coherent).
    assert len(gen_a) > 0 and len(set(gen_a)) >= 4, f"degenerate traced output: {gen_a[:8]}"
    # Deterministic greedy + faithful trace replay => identical token sequences.
    assert gen_a == gen_b, (
        "traced decode replay is non-deterministic across two generate() calls "
        f"(divergence at idx {next((i for i,(x,y) in enumerate(zip(gen_a,gen_b)) if x!=y), 'len')})"
    )
    pipeline.release()
