# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Vision traced-execution test for rednote-hilab/dots.ocr (the validated path).

Mirrors tt-metal's ``test_dots_ocr.py::test_dots_ocr_vision`` -- the config that
traces correctly: DOTS_OCR_PARALLELISM=DP, MESH_DEVICE=T3K, TT_SYMBIOTE_RUN_MODE=
TRACED, and pytest --timeout=0 (trace capture of the full vision+decoder stack
takes several minutes and must not be killed by a timeout).

Why vision traces while text-only does not: the vision prefill graph scatter-
fuses text + vision embeddings into an *intermediate* tensor inside the graph,
so the decoder's ``deallocate(residual)`` frees that intermediate -- not the
graph's persistent trace input. (The text-only dual-stream path passes the
text embeds straight in, which IS the trace input, so the decoder frees it ->
"Buffer is not allocated". dots.ocr is an OCR model; vision is the real path.)

Run:
    export DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K TT_SYMBIOTE_RUN_MODE=TRACED
    pytest tests/models/dots_ocr/test_vision_traced_dots_ocr.py --timeout=0 -s

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import json
import os

import pytest
import torch

import ttnn
from tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline

_DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
_IMAGE_LINK = "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"

_DP = {"trace_region_size": 300_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download(_DOTS_OCR_MODEL_ID)


@pytest.mark.parametrize("device_params", [_DP], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 1), (8, 1)], indirect=True)
def test_dots_ocr_vision_traced(mesh_device):
    """Full vision pipeline traced generate on DP mesh. Run with --timeout=0.

    Validated on T3K DP (8,1) [MESH_DEVICE=T3K] and Blackhole P150x4 DP (4,1)
    [MESH_DEVICE=P150x4]. The (8,1) param skips on hosts with fewer than 8 devices.
    """
    pytest.importorskip("qwen_vl_utils")
    import json as _json

    import requests
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor, Qwen2_5_VLProcessor

    torch.set_grad_enabled(False)
    model_path = _resolve_model_path()
    num = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    batch = num  # DP: one stream per chip

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
    image = image.crop((0, 0, w, int(h * 0.575)))  # top 57.5%, matches reference

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image."},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs,
                       padding=True, max_length=2800, return_tensors="pt")

    input_ids = inputs["input_ids"]
    if batch > 1 and input_ids.shape[0] == 1:
        input_ids = input_ids.expand(batch, -1).contiguous()
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    generated = pipeline.generate(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                                  max_new_tokens=180, stop_on_eos=False)
    ttnn.synchronize_device(mesh_device)

    new = generated[0] if (generated and isinstance(generated[0], list)) else generated
    text = processor.decode(new, skip_special_tokens=True)
    print(f"\n[dots_ocr vision traced] {len(new)} tokens:\n{text!r}\n")

    assert len(new) >= 8, f"too few tokens: {len(new)}"
    assert len(set(new)) >= 4, f"degenerate output: {new[:16]}"
    assert len(text.strip()) > 0, "empty decoded output"
    pipeline.release()
