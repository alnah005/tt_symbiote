# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 (decoder) PCC tests for dots.ocr.

Validates the optimized TTNN decoder against the HuggingFace reference on
IDENTICAL random weights (``AutoModelForCausalLM.from_config`` -- no gated
checkpoint download needed). Adapted from the proven tt-metal reference
harness (``models/experimental/tt_symbiote/tests/test_dots_ocr.py``) to the
downstream tt_symbiote public surface (strict 2-arg ``set_device`` +
``tests.shared.pcc_utils``).

Run on T3K data-parallel (8, 1):
    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier3/test_decoder_dots_ocr.py -x -s \
        --override-ini="addopts="
"""

import pytest
import torch

import ttnn
from tt_symbiote.models.dots_ocr import _create_paged_kv_cache
from tt_symbiote.models.dots_ocr.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLayerStack,
)
from tt_symbiote.utils.device_management import set_device

from ..dots_ocr_helpers import (
    DOTS_OCR_MODEL_ID,
    assert_l1_resident,
    assert_pcc,
    canonical_to_torch,
    dots_ocr_device_params,
    resolve_mesh_device_shape,
    resolve_model_path,
)

DOTS_OCR_LOCAL_PATH = resolve_model_path()

# Single-layer bar (decode + prefill): tight enough to catch silent
# layout/sharding/RoPE/residual bugs (random bf16 + paged-SDPA vs eager adds
# some drift). This is the per-layer numerical-fidelity gate.
PCC_ONE_LAYER = 0.99
# 28-layer full-stack on RANDOM weights is a COARSE multi-layer regression gate,
# not an accuracy target: random (unstructured) weights compound per-layer bf16 +
# BFP4-weight drift across 28 layers (observed ~0.923), while the per-layer path
# is proven at 0.99 above and the decoder's dtype scheme makes later layers MORE
# precise (BFP4 layers 0-6, BFP8 7+). The authoritative multi-layer correctness
# gate is the Tier4 real-weight semantic OCR test. 0.90 catches gross
# accumulation regressions (a dropped residual / wrong RoPE tanks it far below)
# without flapping on benign random-init drift.
PCC_FULL_DECODER = 0.90


def _hf_one_layer_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    cfg.num_hidden_layers = 1
    return cfg


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_decode_one_layer_pcc(mesh_device):
    """One decoder layer in DECODE mode vs HF reference at cache_position=0."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    cfg = _hf_one_layer_config()
    hf_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config
    hf_layer = hf_model.model.layers[0]
    hf_rotary_emb = hf_model.model.rotary_emb

    # --- HF reference FIRST (from_torch + preprocess_weights mutate weights in place) ---
    hidden_states_torch = torch.randn(1, 1, model_config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    cos, sin = hf_rotary_emb(hidden_states_torch, position_ids)
    torch_output = hf_layer(
        hidden_states_torch,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=(cos, sin),
    )[0]
    if torch_output.dim() == 2:
        torch_output = torch_output.unsqueeze(1)

    # --- TTNN decode path ---
    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer)
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()
    set_device(layer, mesh_device)  # subsumes preprocess_weights + move_weights_to_device

    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cache_position = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = layer.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)[0]
    ttnn.synchronize_device(mesh_device)
    assert_l1_resident(output, "decoder layer output")

    ttnn_output_torch = canonical_to_torch(output, mesh_device).to(torch.bfloat16).reshape(torch_output.shape)
    assert_pcc(ttnn_output_torch, torch_output, threshold=PCC_ONE_LAYER, msg="DotsOCRDecoderLayer[decode]")


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_decode_full_decoder_pcc(mesh_device):
    """All 28 decoder layers in DECODE mode vs HF reference (accumulated drift gate)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    assert model_config.num_hidden_layers == 28, "dots.ocr decoder should have 28 layers"
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()

    # HF reference across all 28 layers BEFORE any TTNN setup (in-place weight mutation).
    hidden_states_torch = torch.randn(1, 1, model_config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    cos, sin = hf_model.model.rotary_emb(hidden_states_torch, position_ids)
    torch_hidden = hidden_states_torch
    for hf_layer in hf_model.model.layers:
        torch_hidden = hf_layer(
            torch_hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )[0]
        if torch_hidden.dim() == 2:
            torch_hidden = torch_hidden.unsqueeze(1)
    torch_output = torch_hidden

    decoder_layers = []
    for layer_idx, hf_layer in enumerate(hf_model.model.layers):
        layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer)
        layer._unique_name = f"model.layers.{layer_idx}"
        layer.override_children_module_names()
        decoder_layers.append(layer)
    del hf_model

    decoder_stack = TTNNDotsOCRLayerStack(decoder_layers)
    decoder_stack._unique_name = "model.layer_stack"
    set_device(decoder_stack, mesh_device)

    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cache_position = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = decoder_stack.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)
    ttnn.synchronize_device(mesh_device)
    assert_l1_resident(output, "decoder stack output")

    ttnn_output_torch = canonical_to_torch(output, mesh_device).to(torch.bfloat16).reshape(torch_output.shape)
    assert_pcc(ttnn_output_torch, torch_output, threshold=PCC_FULL_DECODER, msg="DotsOCRLayerStack[28L decode]")


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128], ids=["prefill_32", "prefill_128"])
def test_dots_ocr_prefill_one_layer_pcc(mesh_device, seq_len):
    """One decoder layer in PREFILL mode (seq 32/128) vs HF reference, PCC 0.99."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    cfg = _hf_one_layer_config()
    hf_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config

    # HF reference FIRST (from_torch + preprocess_weights mutate weights in place).
    hidden_states_torch = torch.randn(1, seq_len, model_config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(hidden_states_torch, position_ids)
    torch_output = hf_model.model.layers[0](
        hidden_states_torch,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=(cos, sin),
    )[0]
    if torch_output.dim() == 2:
        torch_output = torch_output.unsqueeze(1)

    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_model.model.layers[0])
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()
    set_device(layer, mesh_device)

    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cache_position = ttnn.from_torch(
        torch.arange(seq_len, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = layer.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)[0]
    ttnn.synchronize_device(mesh_device)

    ttnn_output_torch = canonical_to_torch(output, mesh_device).to(torch.bfloat16).reshape(torch_output.shape)
    assert_pcc(ttnn_output_torch, torch_output, threshold=PCC_ONE_LAYER, msg=f"DotsOCRDecoderLayer[prefill.{seq_len}]")
