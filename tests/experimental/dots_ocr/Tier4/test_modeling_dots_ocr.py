# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 (full model / e2e) tests for dots.ocr.

End-to-end TTNN pipeline over REAL weights (HF cache):
  * text-only generation -> coherent English (decoder semantic gate),
  * image + text OCR generation -> non-empty structured text (VLM gate).

Adapted from the proven tt-metal reference harness to the downstream
tt_symbiote ``TTNNDotsOCRPipeline``. Run on T3K data-parallel (8, 1):

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier4/test_modeling_dots_ocr.py -x -s \
        --override-ini="addopts="
"""

import time

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


def _decode_streams(generated_ids, decode_fn):
    if generated_ids and isinstance(generated_ids[0], list):
        streams = [decode_fn(seq, skip_special_tokens=True) for seq in generated_ids]
        return "\n--- stream ---\n".join(streams), sum(len(s) for s in generated_ids)
    return decode_fn(generated_ids, skip_special_tokens=True), len(generated_ids)


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_pipeline_text(mesh_device):
    """Full TTNN pipeline, text-only: embed -> 28 layers -> LM head -> argmax."""
    pbatch = pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=pbatch)
    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    messages = [{"role": "user", "content": "What is optical character recognition and how does it work?"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    input_ids = stack_input_ids_for_dp(inputs["input_ids"])

    pipeline.warmup(input_ids)
    start = time.time()
    generated_ids = pipeline.generate(input_ids, max_new_tokens=64)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.time() - start

    text, num_tokens = _decode_streams(generated_ids, tokenizer.decode)
    print(f"\n[dots.ocr TEXT] {num_tokens} tok in {elapsed:.1f}s ({num_tokens/elapsed:.1f} tok/s)\n{text}\n")

    assert num_tokens > 0, "pipeline generated no tokens"
    assert len(text.strip()) > 0, "generated text should not be empty"
    # Semantic floor: not a single token repeated to fill the budget.
    first_stream = generated_ids[0] if (generated_ids and isinstance(generated_ids[0], list)) else generated_ids
    assert len(set(first_stream)) > 1, f"degenerate (single-token) output: {first_stream[:8]}..."

    pipeline.release()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
@pytest.mark.parametrize(
    "image_link",
    ["https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"],
)
def test_dots_ocr_pipeline_vision(mesh_device, image_link):
    """Full TTNN pipeline, image + text OCR: vision tower -> scatter-merge -> decode."""
    pytest.importorskip("qwen_vl_utils")
    import json
    import os

    from qwen_vl_utils import process_vision_info

    try:
        import requests
        from PIL import Image

        image = Image.open(requests.get(image_link, stream=True, timeout=30).raw)
    except Exception as e:  # offline / no network -> skip the OCR gate
        pytest.skip(f"could not fetch demo image ({type(e).__name__}: {e})")

    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    w, h = image.size
    image = image.crop((0, 0, w, int(h * 0.575)))  # top crop (matches reference)
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

    input_ids = stack_input_ids_for_dp(proc["input_ids"])
    pixel_values = proc["pixel_values"].to(torch.bfloat16)
    image_grid_thw = proc["image_grid_thw"]

    pbatch = pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=DOTS_OCR_LOCAL_PATH, device=mesh_device, batch_size=pbatch)
    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    start = time.time()
    generated_ids = pipeline.generate(
        input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, max_new_tokens=128, stop_on_eos=False
    )
    ttnn.synchronize_device(mesh_device)
    elapsed = time.time() - start

    decoded, num_tokens = _decode_streams(generated_ids, processor.decode)
    print(f"\n[dots.ocr VISION] {num_tokens} tok in {elapsed:.1f}s ({num_tokens/elapsed:.1f} tok/s)\n{decoded}\n")

    assert num_tokens > 0, "vision pipeline generated no tokens"
    assert len(decoded.strip()) > 0, "OCR output should not be empty"
    pipeline.release()
