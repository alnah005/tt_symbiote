# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 end-to-end pipeline test for rednote-hilab/dots.ocr.

Drives the ported ``TTNNDotsOCRPipeline`` through real text generation on the
full 28-layer model + LM head, mirroring
``models/experimental/tt_symbiote/tests/test_dots_ocr.py::test_dots_ocr_text``.

This is the bring-up functionality gate: it loads the real checkpoint, runs
warmup + generate, decodes the tokens, and asserts the output is semantically
coherent (non-empty, not NaN/garbage, not a single repeated token) -- the
semantic-correctness bar from the bring-up decision profile.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import os

import pytest
import torch

import ttnn
from tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline

_DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"

_MESH_DEVICE_MAP = {
    "N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4),
    "P150": (1, 1), "P300": (1, 2), "P150x4": (1, 4), "P150x8": (1, 8), "BHGLX": (8, 4),
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


def _device_params():
    num = 1
    shape = _resolve_mesh_device_shape()
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        num = shape[0] * shape[1]
    elif isinstance(shape, int):
        num = shape
    return {
        "trace_region_size": 300_000_000,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if num > 1 else ttnn.FabricConfig.DISABLED,
    }


def _mesh_num_devices():
    shape = _resolve_mesh_device_shape()
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        return shape[0] * shape[1]
    if isinstance(shape, int):
        return shape
    return 1


def _pipeline_batch_size():
    """DP requires batch_size == num_devices (one stream per device)."""
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() != "DP":
        return 1
    n = _mesh_num_devices()
    return n if n > 1 else 1


def _stack_input_ids_for_dp(input_ids):
    """Turn [1, S] into [B, S] by repeating the prompt on each DP stream."""
    bs = _pipeline_batch_size()
    if bs <= 1 or input_ids.shape[0] == bs:
        return input_ids
    return input_ids.expand(bs, -1).contiguous()


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(_DOTS_OCR_MODEL_ID)
    except Exception:
        return _DOTS_OCR_MODEL_ID


def _assert_semantically_coherent(token_ids, text, tokenizer):
    """Bring-up semantic-correctness bar for an OCR/vision model on text input.

    With stop_on_eos=False the model emits a fixed number of tokens. The bar
    (no human judgement needed) is that the decode stack is healthy:
      - a non-trivial number of tokens were produced,
      - they are not a single token repeated (degenerate / NaN-logit signature),
      - they show real diversity (>= 4 distinct ids), and
      - decoding without skipping specials yields printable content.
    """
    assert len(token_ids) >= 8, f"too few tokens generated: {len(token_ids)}"
    unique = set(token_ids)
    assert len(unique) > 1, f"degenerate output: single token {token_ids[:8]} repeated"
    assert len(unique) >= 4, f"low-diversity output ({len(unique)} distinct ids): {token_ids[:16]}"
    # Decode WITHOUT skipping specials so EOS-heavy OCR output still yields text.
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    printable = sum(1 for ch in raw if ch.isprintable() or ch.isspace())
    assert printable >= max(1, int(0.8 * len(raw))), "output is mostly non-printable garbage"


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_pipeline_text(mesh_device):
    """End-to-end text generation through the ported TTNNDotsOCRPipeline."""
    from transformers import AutoTokenizer

    torch.set_grad_enabled(False)
    model_path = _resolve_model_path()

    pbatch = _pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=model_path,
        device=mesh_device,
        batch_size=pbatch,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": "The capital of France is"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = _stack_input_ids_for_dp(inputs["input_ids"])

    pipeline.warmup(input_ids)
    # dots.ocr is an OCR/vision model; on a plain text prompt it tends to emit
    # EOS almost immediately. Force a fixed-length continuation (stop_on_eos=
    # False) so the semantic check can verify the forward stack + sampling +
    # KV cache produce varied, non-degenerate tokens across many decode steps
    # (matches the reference vision test's stop_on_eos=False usage).
    generated_ids = pipeline.generate(input_ids, max_new_tokens=48, stop_on_eos=False)
    ttnn.synchronize_device(mesh_device)

    if generated_ids and isinstance(generated_ids[0], list):
        token_ids = generated_ids[0]
    else:
        token_ids = generated_ids
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"\n[dots_ocr pipeline] generated {len(token_ids)} tokens")
    print(f"  distinct ids: {len(set(token_ids))}")
    print(f"  decoded (skip_special=True): {text!r}")
    print(f"  decoded (raw): {tokenizer.decode(token_ids, skip_special_tokens=False)!r}\n")

    _assert_semantically_coherent(token_ids, text, tokenizer)
    pipeline.release()
