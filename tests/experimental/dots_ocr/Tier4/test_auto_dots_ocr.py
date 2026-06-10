# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 Auto-API test for dots.ocr.

Exercises the canonical user-facing surface end to end through the registered
``DotsOCRRecipe``:

    AutoModelForCausalLM.from_pretrained(...)  # recipe build_module_dict + post_register
    set_device(model, mesh_device)             # recipe.make_kv_cache builds the TTNN pipeline
    model.generate(...)                         # recipe generate-shim delegates to the pipeline

This is the recipe glue (generate shim + make_kv_cache) that the direct-pipeline
tests (test_modeling) do not cover. Run on T3K data-parallel (8, 1):

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier4/test_auto_dots_ocr.py -x -s \
        --override-ini="addopts="
"""

import pytest
import torch
from transformers import AutoTokenizer

from tt_symbiote import AutoModelForCausalLM, set_device
from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY

from ..dots_ocr_helpers import dots_ocr_device_params, resolve_mesh_device_shape, resolve_model_path

DOTS_OCR_LOCAL_PATH = resolve_model_path()


def test_recipe_registered():
    """DotsOCRRecipe is registered for the HF class (software-only)."""
    assert "DotsOCRForCausalLM" in TT_MODEL_REGISTRY, "DotsOCRRecipe must register for DotsOCRForCausalLM"


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_auto_generate(mesh_device):
    """Full Auto-API path: from_pretrained -> set_device (make_kv_cache) -> generate."""
    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        DOTS_OCR_LOCAL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)

    # set_device invokes DotsOCRRecipe.make_kv_cache, which builds the TTNN
    # pipeline (reusing the already-loaded HF weights) and installs it on the model.
    set_device(model, mesh_device)
    assert (
        getattr(model, "_tt_pipeline", None) is not None
    ), "set_device should build model._tt_pipeline via make_kv_cache"

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is optical character recognition?"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    prompt_len = int(input_ids.shape[-1])

    # The generate shim warms up + delegates to the pipeline; it replicates the
    # single prompt across DP streams internally and returns [1, prompt+new].
    out = model.generate(input_ids=input_ids, max_new_tokens=32, stop_on_eos=False)
    new_ids = out[0][prompt_len:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"\n[dots.ocr AUTO] {len(new_ids)} new tok: {text!r}\n")

    assert len(new_ids) > 0, "Auto generate produced no new tokens"
    assert len(set(new_ids)) >= 4, f"degenerate Auto output: {new_ids[:8]}"
    assert len(text.strip()) > 0, "Auto generate output should not be empty"
