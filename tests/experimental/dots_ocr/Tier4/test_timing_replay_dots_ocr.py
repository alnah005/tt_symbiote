# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 traced-replay performance benchmark for dots.ocr.

Measures the STEADY-STATE traced decode-replay throughput on T3K (8, 1) DP.
The pipeline captures graph_decode in ``warmup()``; ``generate()`` replays it
once per decode step. To isolate the replay cost from the one-time prefill +
fixed Python/launch overhead, we time two ``generate`` calls of different
length and difference them:

    ms_per_decode_step = (t(N2) - t(N1)) / (N2 - N1)

Each decode step advances ALL ``num_streams`` DP streams by one token, so:
    per-stream tok/s   = 1000 / ms_per_decode_step
    aggregate tok/s    = num_streams * per-stream tok/s

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier4/test_timing_replay_dots_ocr.py -x -s \
        --override-ini="addopts="
"""

import json
import time

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
_N1, _N2 = 16, 80  # decode steps for the two timed generates (diff = 64 steps)


def _timed_generate(pipeline, ids, n, mesh_device):
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    pipeline.generate(ids, max_new_tokens=n)
    ttnn.synchronize_device(mesh_device)
    return time.perf_counter() - t0


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_profile_decode_steps(mesh_device):
    """Short traced-replay decode (4 steps) for a tracy device-time profile.

    Tracy-profiled to attribute the ~380 ms wall/step between on-device compute
    and host/fabric latency. Generates only 4 tokens/stream so tracy's op buffer
    captures cleanly. Not an assertion test -- it just exercises the replay path.
    """
    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    ids = stack_input_ids_for_dp(
        tok.apply_chat_template(
            [{"role": "user", "content": "What is OCR?"}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
    )
    pipeline.warmup(ids)
    pipeline.generate(ids, max_new_tokens=4)  # 4 traced decode-step replays
    ttnn.synchronize_device(mesh_device)
    pipeline.release()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_traced_e2e_walltime(mesh_device):
    """End-to-end walltime of ONE traced-replayed generate (prefill + decode replay).

    Reports, with NO per-step PROFILE_SYNC penalty:
      * warmup_capture_s   -- one-time trace capture (compile + capture graphs),
      * e2e_replay_128_s   -- a full generate() of 128 new tokens/stream replaying the trace,
      * prefill_s          -- generate(1) walltime (prefill + 1 decode step),
      * derived steady-state decode ms/step and tok/s.
    """
    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    streams = mesh_num_devices() if batch > 1 else 1
    n_e2e = 128

    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    ids = stack_input_ids_for_dp(
        tok.apply_chat_template(
            [{"role": "user", "content": "What is optical character recognition?"}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
    )

    ttnn.synchronize_device(mesh_device)
    tw = time.perf_counter()
    pipeline.warmup(ids)  # one-time: compile + capture graph_prefill / graph_decode
    ttnn.synchronize_device(mesh_device)
    warmup_s = time.perf_counter() - tw

    _timed_generate(pipeline, ids, 8, mesh_device)  # prime replay path (discard)
    prefill_s = _timed_generate(pipeline, ids, 1, mesh_device)  # prefill + 1 decode step
    e2e_s = _timed_generate(pipeline, ids, n_e2e, mesh_device)  # full e2e traced replay

    decode_ms = (e2e_s - prefill_s) / (n_e2e - 1) * 1000.0
    result = {
        "device": "T3K",
        "mesh": [8, 1],
        "mode": "DP",
        "num_streams": streams,
        "warmup_capture_s": round(warmup_s, 2),
        "e2e_replay_generate_s": round(e2e_s, 2),
        "e2e_new_tokens_per_stream": n_e2e,
        "e2e_total_tokens": n_e2e * streams,
        "prefill_first_token_s": round(prefill_s, 3),
        "steady_decode_ms_per_step": round(decode_ms, 1),
        "aggregate_tok_s": round(n_e2e * streams / e2e_s, 1),
        "note": "no PROFILE_SYNC; e2e = one generate() replaying the captured trace (warmup excluded -- it is one-time).",
    }
    print("\n[dots.ocr TRACED E2E WALLTIME] " + json.dumps(result, indent=2) + "\n")
    with open("tests/experimental/dots_ocr/perf_results/traced_e2e_walltime.json", "w") as f:
        json.dump(result, f, indent=2)
    assert e2e_s > 0
    pipeline.release()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
@pytest.mark.parametrize(
    "image_link",
    ["https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"],
)
def test_dots_ocr_traced_e2e_walltime_full_model(mesh_device, image_link):
    """FULL MODEL (vision tower + text decoder) e2e traced-replay walltime.

    The image path exercises the complete pipeline: 42-block vision tower ->
    patch-merger -> scatter-merge into the text stream -> 28-layer decode. The
    vision tower runs ONCE in prefill; decode is text-only thereafter. Isolates
    the vision-inclusive prefill from steady decode (NO PROFILE_SYNC):
      * warmup_capture_s        -- one-time full-pipeline trace capture,
      * vision_prefill_s        -- generate(1) walltime (vision tower + scatter-merge + 1 decode),
      * full_model_e2e_replay_s -- generate(128) replaying the full-model trace.
    """
    import json as _json
    import os

    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info

    try:
        import requests
        from PIL import Image

        image = Image.open(requests.get(image_link, stream=True, timeout=30).raw)
    except Exception as e:
        pytest.skip(f"could not fetch demo image ({type(e).__name__}: {e})")

    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = _json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    w, h = image.size
    image = image.crop((0, 0, w, int(h * 0.575)))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    proc = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, max_length=2800, return_tensors="pt"
    )

    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    streams = mesh_num_devices() if batch > 1 else 1
    n_e2e = 128
    ids = stack_input_ids_for_dp(proc["input_ids"])
    pv = proc["pixel_values"].to(torch.bfloat16)
    grid = proc["image_grid_thw"]

    def _timed_vis_generate(pipeline, n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        pipeline.generate(ids, pixel_values=pv, image_grid_thw=grid, max_new_tokens=n, stop_on_eos=False)
        ttnn.synchronize_device(mesh_device)
        return time.perf_counter() - t0

    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)

    ttnn.synchronize_device(mesh_device)
    tw = time.perf_counter()
    pipeline.warmup(ids, pixel_values=pv, image_grid_thw=grid)  # full-pipeline capture (incl. vision tower)
    ttnn.synchronize_device(mesh_device)
    warmup_s = time.perf_counter() - tw

    prefill_s = _timed_vis_generate(pipeline, 1)  # vision tower + scatter-merge + 1 decode
    e2e_s = _timed_vis_generate(pipeline, n_e2e)  # full-model e2e traced replay

    decode_ms = (e2e_s - prefill_s) / (n_e2e - 1) * 1000.0
    result = {
        "device": "T3K",
        "mesh": [8, 1],
        "mode": "DP",
        "num_streams": streams,
        "path": "vision+text (full model)",
        "image_grid_thw": [int(x) for x in grid[0].tolist()],
        "prompt_tokens": int(ids.shape[-1]),
        "warmup_capture_s": round(warmup_s, 2),
        "vision_prefill_s": round(prefill_s, 2),
        "full_model_e2e_replay_s": round(e2e_s, 2),
        "e2e_new_tokens_per_stream": n_e2e,
        "e2e_total_tokens": n_e2e * streams,
        "steady_decode_ms_per_step": round(decode_ms, 1),
        "aggregate_tok_s": round(n_e2e * streams / e2e_s, 1),
        "note": "no PROFILE_SYNC; full multimodal pipeline (42-block vision tower in prefill + 28-layer decode replay).",
    }
    print("\n[dots.ocr FULL-MODEL TRACED E2E] " + _json.dumps(result, indent=2) + "\n")
    with open("tests/experimental/dots_ocr/perf_results/traced_e2e_full_model.json", "w") as f:
        _json.dump(result, f, indent=2)
    assert e2e_s > 0
    pipeline.release()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_traced_replay_perf(mesh_device):
    torch.set_grad_enabled(False)
    batch = pipeline_batch_size()
    streams = mesh_num_devices() if batch > 1 else 1

    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=batch)
    tok = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    ids = stack_input_ids_for_dp(
        tok.apply_chat_template(
            [{"role": "user", "content": "What is optical character recognition?"}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
    )

    pipeline.warmup(ids)  # capture graph_prefill + graph_decode
    _timed_generate(pipeline, ids, 8, mesh_device)  # prime the replay path (discard)

    t1 = _timed_generate(pipeline, ids, _N1, mesh_device)
    t2 = _timed_generate(pipeline, ids, _N2, mesh_device)

    ms_per_step = (t2 - t1) / (_N2 - _N1) * 1000.0
    per_stream_tps = 1000.0 / ms_per_step
    aggregate_tps = streams * per_stream_tps
    # t1 = prefill + N1 steps  =>  prefill latency (incl. fixed overhead)
    prefill_ms = (t1 - _N1 * ms_per_step / 1000.0) * 1000.0

    result = {
        "device": "T3K",
        "mesh": [8, 1],
        "mode": "DP",
        "num_streams": streams,
        "traced_decode_ms_per_step": round(ms_per_step, 2),
        "per_stream_decode_tok_s": round(per_stream_tps, 2),
        "aggregate_decode_tok_s": round(aggregate_tps, 2),
        "prefill_first_token_ms_approx": round(prefill_ms, 1),
        "method": f"diff of generate({_N1}) and generate({_N2}) wall-clock, synchronize_device-bracketed",
    }
    print("\n[dots.ocr TRACED-REPLAY PERF] " + json.dumps(result, indent=2) + "\n")
    with open("tests/experimental/dots_ocr/perf_results/traced_replay_perf.json", "w") as f:
        json.dump(result, f, indent=2)

    assert ms_per_step > 0 and aggregate_tps > 0
    pipeline.release()
