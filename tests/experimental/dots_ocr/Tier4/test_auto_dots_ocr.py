# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Canonical Auto-API end-to-end test for dots.ocr.

Exercises the HuggingFace drop-in surface backed by the TTNN pipeline recipe::

    from tt_symbiote import AutoModelForCausalLM, set_device
    model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
    set_device(model, mesh_device)
    out = model.generate(**inputs, max_new_tokens=...)

Validates that the recipe (register_recipe("DotsOCRForCausalLM")) loads the HF
model, builds the TTNN pipeline at set_device time, and that model.generate
returns an HF-style [1, prompt_len + new_len] tensor whose continuation is
coherent.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import os

import pytest
import torch

import ttnn

_DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"

_MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}
_DP_MESH_DEVICE_MAP = {"N300": (2, 1), "T3K": (8, 1), "P150x4": (4, 1)}


def _resolve_mesh_device_shape():
    name = os.environ.get("MESH_DEVICE")
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP" and name in _DP_MESH_DEVICE_MAP:
        return _DP_MESH_DEVICE_MAP[name]
    if name in _MESH_DEVICE_MAP:
        return _MESH_DEVICE_MAP[name]
    try:
        return len(ttnn.get_device_ids())
    except Exception:
        return _MESH_DEVICE_MAP["T3K"]


def _mesh_num_devices():
    shape = _resolve_mesh_device_shape()
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        return shape[0] * shape[1]
    return shape if isinstance(shape, int) else 1


def _device_params():
    num = _mesh_num_devices()
    return {
        "trace_region_size": 300_000_000,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if num > 1 else ttnn.FabricConfig.DISABLED,
    }


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(_DOTS_OCR_MODEL_ID)
    except Exception:
        return _DOTS_OCR_MODEL_ID


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_auto_generate(mesh_device):
    """tt_symbiote.AutoModelForCausalLM.from_pretrained -> set_device -> model.generate."""
    from transformers import AutoTokenizer

    from tt_symbiote import AutoModelForCausalLM, set_device

    torch.set_grad_enabled(False)
    model_path = _resolve_model_path()

    # Canonical HF drop-in load -- recipe registered for "DotsOCRForCausalLM"
    # patches model.generate; the TTNN pipeline is built at set_device time.
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    assert getattr(model, "_tt_symbiote_has_recipe", False), "recipe was not applied"

    set_device(model, mesh_device)
    assert getattr(model, "_tt_pipeline", None) is not None, "pipeline not built at set_device"

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": "The capital of France is"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]

    # Canonical generate() call.
    out = model.generate(input_ids=input_ids, max_new_tokens=24, stop_on_eos=False)

    # HF-style return: [1, prompt_len + new_len]; slice off the prompt like the model card.
    assert out.dim() == 2 and out.shape[0] == 1, f"unexpected generate shape {tuple(out.shape)}"
    assert out.shape[1] > input_ids.shape[1], "no new tokens appended"
    new_ids = out[0][input_ids.shape[1] :].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    print(f"\n[dots_ocr Auto] generate -> {len(new_ids)} new tokens: {text!r}\n")

    assert len(new_ids) >= 8, f"too few new tokens: {len(new_ids)}"
    assert len(set(new_ids)) >= 4, f"degenerate output: {new_ids[:16]}"
