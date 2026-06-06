# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# TEMP timing harness (untracked) — times the trace-replay (run 3) prefill+decode
# of dots_ocr: 180 tokens out, sample image, DP=8 (T3K (8,1)). Safe to delete.
import json as _json
import os
import time

import pytest
import torch

import ttnn
from tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline

_DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
_IMAGE_LINK = "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"
_DP = {"trace_region_size": 300_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
_MAX_NEW = 180


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download(_DOTS_OCR_MODEL_ID)


@pytest.mark.parametrize("device_params", [_DP], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
def test_timing_replay(mesh_device):
    pytest.importorskip("qwen_vl_utils")
    import requests
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor, Qwen2_5_VLProcessor

    torch.set_grad_enabled(False)
    model_path = _resolve_model_path()
    num = int(mesh_device.get_num_devices())
    batch = num  # DP: one stream per chip
    print(f"\n[TIMING] mesh num_devices={num} batch(streams)={batch} max_new_tokens={_MAX_NEW}")

    pipeline = TTNNDotsOCRPipeline.from_hf_model(model_path=model_path, device=mesh_device, batch_size=batch)

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(model_path)
    with open(os.path.join(model_path, "chat_template.json")) as f:
        chat_template = _json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    image = Image.open(requests.get(_IMAGE_LINK, stream=True).raw)
    w, h = image.size
    image = image.crop((0, 0, w, int(h * 0.575)))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, max_length=2800, return_tensors="pt"
    )
    input_ids = inputs["input_ids"]
    if batch > 1 and input_ids.shape[0] == 1:
        input_ids = input_ids.expand(batch, -1).contiguous()
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]
    print(f"[TIMING] prompt seq_len={input_ids.shape[1]} image_grid_thw={image_grid_thw.tolist()}")

    # ---- runs 1 + 2: warmup (JIT prime + trace capture) ----
    ttnn.synchronize_device(mesh_device)
    tw0 = time.perf_counter()
    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    ttnn.synchronize_device(mesh_device)
    warmup_s = time.perf_counter() - tw0
    print(f"[TIMING] warmup (run1 JIT + run2 trace-capture): {warmup_s:.2f} s")

    # reset for a clean replay generation
    pipeline.paged_cache.reset()
    pipeline._decode_cache_position = None
    pipeline._decode_seq_counter = 0
    pipeline._decode_token_buffer = None
    pipeline._decode_token_buffer_has_next = False

    # ---- run 3: TRACE REPLAY — timed prefill + decode at phase boundaries ----
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    currents = pipeline.prefill(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    ttnn.synchronize_device(mesh_device)
    t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000.0

    if not isinstance(currents, list):
        currents = [currents]
    num_streams = len(currents)
    generated = [[t] for t in currents]

    n_decode = _MAX_NEW - 1
    t2 = time.perf_counter()
    for _ in range(n_decode):
        nxt = pipeline.decode_step(currents)
        if not isinstance(nxt, list):
            nxt = [nxt]
        for i in range(num_streams):
            generated[i].append(int(nxt[i]))
        currents = nxt
    ttnn.synchronize_device(mesh_device)
    t3 = time.perf_counter()
    decode_s = t3 - t2
    decode_ms = decode_s * 1000.0
    total_ms = (t3 - t0) * 1000.0

    new = generated[0]
    text = processor.decode(new, skip_special_tokens=True)

    print("\n================= DOTS_OCR TRACE-REPLAY TIMING (DP=8, T3K (8,1)) =================")
    print(f"  tokens generated / stream : {len(new)}  (prefill 1 + decode {n_decode})")
    print(f"  DP streams in parallel    : {num_streams}")
    print(f"  PREFILL  (first token)    : {prefill_ms:.1f} ms")
    print(
        f"  DECODE   ({n_decode} tokens)      : {decode_ms:.1f} ms"
        f"   = {decode_ms / n_decode:.2f} ms/token/stream"
        f"   = {n_decode / decode_s:.2f} tok/s/stream"
        f"   = {num_streams * n_decode / decode_s:.1f} tok/s aggregate"
    )
    print(f"  TOTAL    (prefill+decode) : {total_ms:.1f} ms  ({total_ms/1000.0:.2f} s)")
    print("==================================================================================")
    print(f"\n[dots_ocr replay] {len(new)} tokens decoded text:\n{text!r}\n")

    # validity (so we know the timed run was a real, coherent generation)
    assert len(new) == _MAX_NEW, f"expected {_MAX_NEW} tokens/stream, got {len(new)}"
    assert len(set(new)) >= 4, f"DEGENERATE output (timing invalid): {new[:16]}"
    assert len(text.strip()) > 0, "empty decoded output (timing invalid)"
    pipeline.release()
