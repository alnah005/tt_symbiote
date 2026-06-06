# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 decoder/block PCC tests for rednote-hilab/dots.ocr.

Text-side: a single ``TTNNDotsOCRDecoderLayer`` built from the HF Qwen2
decoder layer via ``from_torch``. The decoder layer pulls in the proper
sharded attention (TTNNDotsOCRAttention with paged KV cache + Bailing
rotary setup) and MLP (TTNNDotsOCRMLP with fused gate-up + DRAM-sharded
matmuls), matching the production tt-metal pipeline.

Test pattern mirrors ``models/experimental/tt_symbiote/tests/test_dots_ocr.py``
(``test_dots_ocr_decode_one_layer_l1_boundaries`` and
``test_dots_ocr_prefill_one_layer``):

  1. Build HF model with ``num_hidden_layers=1``.
  2. Compute the torch reference output BEFORE TTNN setup, because
     ``from_torch`` + ``preprocess_weights`` mutates the HF layer's QKV
     weights in place (permutes to the KV-group-interleaved layout the
     paged SDPA kernel expects), and rerunning the HF forward afterwards
     would crash inside ``apply_rotary_pos_emb`` with a head-dim mismatch.
  3. Build the TTNN layer via ``TTNNDotsOCRDecoderLayer.from_torch``,
     set ``_unique_name``, call ``override_children_module_names``.
  4. ``set_device`` -> ``preprocess_weights`` -> ``move_weights_to_device``.
  5. Allocate a paged KV cache via ``_create_paged_kv_cache``.
  6. Run TTNN forward; ``ttnn.synchronize_device``.
  7. For decode (seq_len=1): assert attention and MLP inputs/outputs are
     L1-resident at the boundaries.
  8. Read back via ``ttnn.to_torch`` (composer-aware for multi-device meshes)
     and compare to the HF reference via ``assert_pcc(threshold=0.99)``.

Vision-side: ``test_vision_dots_vision_block`` is left as a register_modules
leaf swap (no vision-side TTNN integration in this port).

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import json
import os
import pathlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tqdm import tqdm

import ttnn
from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.modules.ttnn_normalization import TTNNLocalRMSNorm
from tt_symbiote.models.dots_ocr import (
    TTNNDotsOCRDecoderLayer,
    _create_paged_kv_cache,
)
from tt_symbiote.utils.device_management import set_device
from tt_symbiote.utils.module_replacement import register_modules

from tests.shared.pcc_utils import assert_pcc, compute_pcc


_SHAPES_PATH = pathlib.Path(__file__).parent.parent / "shapes.json"
_SHAPES = json.loads(_SHAPES_PATH.read_text())

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


_DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
    "P150x4": (4, 1),
}


def _resolve_mesh_device_shape():
    name = os.environ.get("MESH_DEVICE")
    # Data-parallel mode (DOTS_OCR_PARALLELISM=DP) maps T3K -> (8, 1) so the
    # trailing mesh dim is 1; that makes _tp_requires_ccl() False and lets the
    # dots.ocr RMSNorm / linears take their full-width sharded fast paths
    # (the TP/CCL path needs width-sharded activations, which an isolated
    # single-layer test does not provide).
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP":
        if name in _DOTS_OCR_DP_MESH_DEVICE_MAP:
            return _DOTS_OCR_DP_MESH_DEVICE_MAP[name]
    if name in _MESH_DEVICE_MAP:
        return _MESH_DEVICE_MAP[name]
    # Fall back to ttnn.get_device_ids() if available; if there are no chips
    # attached (off-hardware pytest --collect-only), return the T3K default
    # so test collection still works.
    try:
        return len(ttnn.get_device_ids())
    except Exception:
        return _MESH_DEVICE_MAP["T3K"]


def _safe_device_params():
    try:
        return _device_params()
    except Exception:
        forced = _fabric_config_override()
        try:
            shape = _resolve_mesh_device_shape()
        except Exception:
            shape = None
        if forced is not None:
            fabric = forced
        else:
            fabric = (
                ttnn.FabricConfig.FABRIC_1D_RING
                if (shape is not None and _ccl_needed(shape))
                else ttnn.FabricConfig.DISABLED
            )
        return {"trace_region_size": 300_000_000, "num_command_queues": 1, "fabric_config": fabric}


def _mesh_num_devices(shape) -> int:
    if isinstance(shape, int):
        return max(1, int(shape))
    if isinstance(shape, (tuple, list)) and len(shape) >= 2:
        return int(shape[0]) * int(shape[1])
    return 1


def _ccl_needed(shape) -> bool:
    # Fabric/CCL is only required when the trailing (tensor-parallel) mesh dim is
    # > 1. Pure data-parallel shapes (8,1)/(4,1)/(2,1) replicate the hidden dim
    # per device and issue no all_gather/reduce_scatter (_tp_requires_ccl is
    # False), so the ethernet fabric is unnecessary -- and on tt-metal f2e12917
    # the live FABRIC_1D_RING ring deadlocks the large prefill program (proven by
    # the DISABLED-vs-RING A/B isolation). A bare-int / 1-D shape (no MESH_DEVICE
    # match) is treated conservatively as a TP-style row mesh so fabric stays ON.
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        return int(shape[1]) > 1
    return _mesh_num_devices(shape) > 1  # int / 1-D shape: assume TP-style row mesh


def _fabric_config_override():
    # Optional A/B isolation lever (S6b): DOTS_OCR_FABRIC=DISABLED|RING forces the
    # fabric_config independent of the _ccl_needed gate so a prefill graph can be
    # run twice varying ONLY fabric_config. Unset -> use the _ccl_needed gate.
    forced = os.environ.get("DOTS_OCR_FABRIC", "").upper()
    if forced == "DISABLED":
        return ttnn.FabricConfig.DISABLED
    if forced in ("RING", "FABRIC_1D_RING"):
        return ttnn.FabricConfig.FABRIC_1D_RING
    return None


def _device_params():
    shape = _resolve_mesh_device_shape()
    forced = _fabric_config_override()
    return {
        "trace_region_size": 300_000_000,
        "num_command_queues": 1,
        "fabric_config": (
            forced
            if forced is not None
            else (ttnn.FabricConfig.FABRIC_1D_RING if _ccl_needed(shape) else ttnn.FabricConfig.DISABLED)
        ),
    }


def _read_back_first_batch(output, mesh_device):
    """Pull a TTNN tensor back to torch, handling multi-device meshes.

    Mirrors the reference test: on multi-device meshes (DP-with-batch=1 or
    TP-after-all-reduce) every device holds identical data; ConcatMeshToTensor
    along batch and take the first slice.
    """
    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices > 1:
        return ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
    return ttnn.to_torch(output)


def _assert_l1_resident(tensor, name: str):
    assert isinstance(tensor, ttnn.Tensor), f"{name} should be a TTNN tensor"
    assert tensor.memory_config().buffer_type == ttnn.BufferType.L1, f"{name} should reside in L1"


def _materialize(modules):
    for _, mod in tqdm(modules.items(), desc="ttnn modules"):
        mod.preprocess_weights()
        mod.move_weights_to_device()


# ---------------------------------------------------------------------------
# Text backbone: TTNNDotsOCRDecoderLayer (single-layer prefill + decode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [_safe_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_resolve_mesh_device_shape()], indirect=True)
def test_text_qwen2_decoder_layer_decode(mesh_device):
    """Single-layer decode (seq_len=1) with L1-boundary checks + PCC vs HF reference."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(_DOTS_OCR_MODEL_ID, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config

    # Compute HF reference BEFORE any TTNN setup -- from_torch + preprocess_weights
    # mutates the HF layer's QKV weights in place (KV-group-interleaved layout),
    # so a post-TTNN HF forward would crash inside apply_rotary_pos_emb.
    hidden_states_torch = torch.randn(1, 1, model_config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    cos, sin = hf_model.model.rotary_emb(hidden_states_torch, position_ids)
    torch_output = hf_model.model.layers[0](
        hidden_states_torch,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=(cos, sin),
    )[0]

    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_model.model.layers[0])
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()

    set_device(layer, mesh_device)
    layer.preprocess_weights()
    layer.move_weights_to_device()

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

    seen_boundaries = {"attn": False, "mlp": False}
    original_attn_forward = layer.self_attn.forward
    original_mlp_forward = layer.mlp.forward

    def checked_attn_forward(*args, **kwargs):
        attn_input = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        _assert_l1_resident(attn_input, "attention input")
        output = original_attn_forward(*args, **kwargs)
        _assert_l1_resident(output[0], "attention output")
        seen_boundaries["attn"] = True
        return output

    def checked_mlp_forward(hidden_states):
        _assert_l1_resident(hidden_states, "MLP input")
        output = original_mlp_forward(hidden_states)
        _assert_l1_resident(output, "MLP output")
        seen_boundaries["mlp"] = True
        return output

    layer.self_attn.forward = checked_attn_forward
    layer.mlp.forward = checked_mlp_forward

    output = layer.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)[0]
    ttnn.synchronize_device(mesh_device)

    _assert_l1_resident(output, "decoder layer output")
    assert seen_boundaries == {"attn": True, "mlp": True}

    ttnn_output_torch = _read_back_first_batch(output, mesh_device).to(torch.bfloat16).reshape(torch_output.shape)
    # Random-weight bf16 + paged-SDPA vs eager attention adds some numerical
    # drift relative to fp32 ref; 0.99 is the tight-but-safe bar that still
    # catches silent regressions (layout/sharding bugs, wrong RoPE, dropped
    # residual, etc.) -- matches reference test threshold.
    assert_pcc(ttnn_output_torch, torch_output, threshold=0.99, msg="decoder.decode.seq_1")


@pytest.mark.parametrize("device_params", [_safe_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_resolve_mesh_device_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128], ids=["prefill_32", "prefill_128"])
def test_text_qwen2_decoder_layer_prefill(mesh_device, seq_len):
    """Single-layer prefill (seq_len in {32, 128}) with paged KV cache + PCC vs HF reference."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(_DOTS_OCR_MODEL_ID, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config

    # HF reference BEFORE TTNN mutations.
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

    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_model.model.layers[0])
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()

    set_device(layer, mesh_device)
    layer.preprocess_weights()
    layer.move_weights_to_device()

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

    ttnn_output_torch = _read_back_first_batch(output, mesh_device).to(torch.bfloat16).reshape(torch_output.shape)
    assert_pcc(ttnn_output_torch, torch_output, threshold=0.99, msg=f"decoder.prefill.seq_{seq_len}")


# ---------------------------------------------------------------------------
# Vision tower: DotsVisionBlock (leaf swap -- no vision integration in this port)
# ---------------------------------------------------------------------------


def _vision_block_inputs(seq_len: int, embed_dim: int, num_heads: int):
    head_dim = embed_dim // num_heads
    rotary_pos_emb = torch.randn(seq_len, head_dim // 2, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
    return cu_seqlens, rotary_pos_emb


def _vision_config_ns():
    v = _SHAPES["vision_config"]
    return SimpleNamespace(
        embed_dim=v["embed_dim"],
        hidden_size=v["hidden_size"],
        intermediate_size=v["intermediate_size"],
        num_attention_heads=v["num_attention_heads"],
        rms_norm_eps=v["rms_norm_eps"],
        use_bias=v["use_bias"],
        is_causal=False,
    )


@pytest.mark.parametrize("device_params", [_safe_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_resolve_mesh_device_shape()], indirect=True)
def test_vision_dots_vision_block(mesh_device):
    """DotsVisionBlock with SDPA attention at dots_vit dims (one packed sequence)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    DotsVisionBlock = get_class_from_dynamic_module("modeling_dots_vision.DotsVisionBlock", "rednote-hilab/dots.ocr")
    DotsVisionRMSNorm = get_class_from_dynamic_module("modeling_dots_vision.RMSNorm", "rednote-hilab/dots.ocr")

    cfg = _vision_config_ns()
    block = DotsVisionBlock(cfg, attn_implementation="eager").to(torch.bfloat16)
    block.eval()
    torch.set_grad_enabled(False)

    seq_len = 256
    hidden_states = TorchTTNNTensor(torch.randn(seq_len, cfg.embed_dim, dtype=torch.bfloat16))
    cu_seqlens, rotary_pos_emb = _vision_block_inputs(seq_len, cfg.embed_dim, cfg.num_attention_heads)

    torch_out = block(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

    swap_map = {
        nn.Linear: TTNNLinear,
        DotsVisionRMSNorm: TTNNLocalRMSNorm,
    }
    modules = register_modules(block, swap_map, model_config=None)
    set_device(block, mesh_device)
    _materialize(modules)

    ttnn_out = block(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
    assert_pcc(ttnn_out, torch_out, threshold=0.99, msg="vision.DotsVisionBlock")
