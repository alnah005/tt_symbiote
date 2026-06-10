# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware test for the additive ``set_vllm_page_table`` hook (Tier S2 seam).

This validates the single shared seam that lets tt-inference-server's vLLM
backend drive ``TTNNPagedAttentionKVCache`` with an externally-managed (vLLM
block-manager) page table, instead of the cache's default contiguous identity
mapping. See docs/development/tt_inference_server_integration.md §9.

It uses a small GLM4-MoE-Lite attention (random weights via ``from_config`` so
there is no large download) as the vehicle, because that attention module routes
through the paged ops we need to cover:

  * prefill -> ``paged_fill_on_device``
  * decode  -> ``paged_sdpa_decode``

The hook is exercised by installing a *non-identity* (reversed) page table — the
kind a real block manager hands out — and confirming that BOTH the prefill write
and the decode read-back stay numerically correct against a torch ``DynamicCache``
reference (PCC >= 0.99). The device page-table tensor is also round-tripped to
prove the hook actually re-installed our mapping on device.

Runs on a single-device ``(1, 1)`` submesh of whatever board is present (incl.
T3K). The hook's multi-device branch is the same ``ttnn.from_torch`` +
``ReplicateTensorToMesh`` pattern already exercised by ``to_device`` on the full
8-device dots_ocr runs, so single-device coverage here is sufficient for the new
logic.
"""

import pytest
import torch

import ttnn
from tests.shared.pcc_utils import assert_pcc
from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.modules.ttnn_attention import (
    PagedAttentionConfig,
    TTNNGlm4MoeLiteAttention,
    TTNNPagedAttentionKVCache,
)
from tt_symbiote.utils.device_management import set_device


def _small_glm4_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.kv_lora_rank = 128
    config.q_lora_rank = 192
    config.qk_nope_head_dim = 64
    config.qk_rope_head_dim = 64
    config.qk_head_dim = 128
    config.v_head_dim = 128
    config.moe_intermediate_size = 384
    config.num_local_experts = 4
    config.num_experts_per_tok = 2
    return config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_set_vllm_page_table_prefill_and_decode(mesh_device):
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache

    device = mesh_device

    config = _small_glm4_config()
    model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
    torch.set_grad_enabled(False)
    torch_attn = model.model.layers[0].self_attn

    paged_config = PagedAttentionConfig(block_size=32, max_num_blocks=64, batch_size=1)
    paged_cache = TTNNPagedAttentionKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=128,
        config=paged_config,
        device=None,
        dtype=torch.bfloat16,
    ).to_device(device)

    # vLLM-style block table: reverse the contiguous identity mapping so logical
    # block i -> physical block (blocks_per_sequence - 1 - i). Non-trivial layout.
    blocks_per_sequence = paged_config.blocks_per_sequence
    reversed_pt = torch.arange(blocks_per_sequence - 1, -1, -1, dtype=torch.int32).unsqueeze(0)
    paged_cache.set_vllm_page_table(reversed_pt)

    # Round-trip: confirm the device page-table tensor really holds our mapping.
    if device.get_num_devices() > 1:
        host_pt = ttnn.to_torch(
            paged_cache._tt_page_table,
            mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
        )
    else:
        host_pt = ttnn.to_torch(paged_cache._tt_page_table)
    assert torch.equal(
        host_pt[0, :blocks_per_sequence].to(torch.int32), reversed_pt[0]
    ), "set_vllm_page_table did not install the mapping on device"

    ttnn_attn = TTNNGlm4MoeLiteAttention.from_torch(torch_attn, distributed=False)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    dynamic_cache = DynamicCache()

    # --- Prefill: exercises paged_fill_on_device via the reversed table ---
    hidden = torch.randn(1, 5, 512, dtype=torch.bfloat16)
    pos = torch.arange(5).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, pos)
    torch_out_prefill = torch_attn(
        hidden,
        attention_mask=None,
        position_embeddings=(cos, sin),
        past_key_values=dynamic_cache,
    )
    ttnn_out_prefill = ttnn_attn(
        TorchTTNNTensor(hidden),
        position_embeddings=(cos, sin),
        past_key_values=paged_cache,
        cache_position=torch.arange(5).unsqueeze(0),
    )
    assert_pcc(ttnn_out_prefill, torch_out_prefill, threshold=0.99, msg="VllmPageTable_PagedPrefill")
    assert paged_cache.get_seq_length(0) == 5

    # --- Decode: exercises paged_sdpa_decode read-back via the reversed table ---
    hidden_d = torch.randn(1, 1, 512, dtype=torch.bfloat16)
    pos_d = torch.arange(5, 6).unsqueeze(0)
    cos_d, sin_d = model.model.rotary_emb(hidden_d, pos_d)
    torch_out_decode = torch_attn(
        hidden_d,
        attention_mask=None,
        position_embeddings=(cos_d, sin_d),
        past_key_values=dynamic_cache,
    )
    ttnn_out_decode = ttnn_attn(
        TorchTTNNTensor(hidden_d),
        position_embeddings=(cos_d, sin_d),
        past_key_values=paged_cache,
        cache_position=torch.arange(5, 6).unsqueeze(0),
    )
    assert_pcc(ttnn_out_decode, torch_out_decode, threshold=0.99, msg="VllmPageTable_PagedDecode")
    assert paged_cache.get_seq_length(0) == 6
