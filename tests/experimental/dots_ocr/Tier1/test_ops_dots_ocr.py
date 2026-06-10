# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 (leaf op) PCC tests for dots.ocr.

Validates foundational ops (RMSNorm, token embedding) against the HF reference
on identical random weights. Runs on T3K data-parallel (8, 1) -- where the
column ("shard") mesh axis is size 1, so each device holds the full hidden dim
and ``canonical_to_torch`` reconstructs the reference output by taking stream 0.

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier1/test_ops_dots_ocr.py -x -s \
        --override-ini="addopts="
"""

import pytest
import torch

import ttnn
from tt_symbiote.models.dots_ocr import TTNNEmbedding
from tt_symbiote.models.dots_ocr.dots_ocr_decoder_layer import TTNNDotsOCRLocalShardRMSNorm
from tt_symbiote.utils.device_management import set_device

from ..dots_ocr_helpers import (
    assert_pcc,
    canonical_to_torch,
    dots_ocr_device_params,
    resolve_mesh_device_shape,
    resolve_model_path,
)

DOTS_OCR_LOCAL_PATH = resolve_model_path()


def _hf_model_1layer():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    cfg.num_hidden_layers = 1
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_rmsnorm_decode_pcc(mesh_device):
    """TTNNDotsOCRLocalShardRMSNorm (decode sharded path) vs HF RMSNorm."""
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    hf_model = _hf_model_1layer()
    hidden_size = hf_model.config.hidden_size
    torch_norm = hf_model.model.layers[0].input_layernorm

    x = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)
    torch_out = torch_norm(x)  # reference BEFORE any in-place weight mutation

    norm = TTNNDotsOCRLocalShardRMSNorm.from_torch(torch_norm)
    norm._unique_name = "model.layers.0.input_layernorm"
    set_device(norm, mesh_device)

    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    out_tt = norm.forward(x_tt)
    ttnn.synchronize_device(mesh_device)

    out_torch = canonical_to_torch(out_tt, mesh_device).to(torch.bfloat16).reshape(torch_out.shape)
    assert_pcc(out_torch, torch_out, threshold=0.99, msg="DotsOCR RMSNorm[decode]")


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_embedding_pcc(mesh_device):
    """TTNNEmbedding token lookup vs HF nn.Embedding."""
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    hf_model = _hf_model_1layer()
    vocab_size = hf_model.config.vocab_size
    torch_embed = hf_model.model.embed_tokens

    seq = 32  # multiple of 32 -> embedding's direct uint32 typecast path
    ids = torch.randint(0, vocab_size, (1, seq), dtype=torch.int64)
    torch_out = torch_embed(ids)  # [1, seq, hidden]

    emb = TTNNEmbedding.from_torch(torch_embed)
    emb._unique_name = "model.embed_tokens"
    set_device(emb, mesh_device)

    ids_tt = ttnn.from_torch(
        ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = emb.forward(ids_tt)
    ttnn.synchronize_device(mesh_device)

    out_torch = canonical_to_torch(out_tt, mesh_device).to(torch.bfloat16).reshape(torch_out.shape)
    assert_pcc(out_torch, torch_out.to(torch.bfloat16), threshold=0.99, msg="DotsOCR Embedding")
