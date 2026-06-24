# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware tests for dots.ocr's Tier-S2 continuous-batching primitives.

Validates the three additive seams that let tt-inference-server's vLLM backend
drive dots.ocr's forked ``TTNNPagedAttentionKVCache`` for continuous batching
(see docs/development/tt_inference_server_integration.md §9):

  * TS-1  ``set_vllm_page_table`` -- install an externally-managed (vLLM
          block-manager) page table, DP-sharded so each mesh device (one vLLM
          sequence) gets its own row, instead of the cache's default contiguous
          identity mapping.
  * TS-2  trace-stable install -- the device page-table tensor is updated
          IN PLACE (buffer identity preserved) so a captured decode trace
          survives block-table swaps between requests.
  * TS-3  per-DP-stream cache positions -- each device decodes at its own cache
          position (no collapse to a single global scalar), with sentinel
          (clamped-to-0) tolerance for inactive / finished streams.

The page-table seam (TS-1/TS-2) is exercised at the cache level on whatever
board is present. The end-to-end paged prefill+decode and per-stream-position
tests use dots.ocr's real decoder, whose attention is arch-gated to multi-device
boards (``@run_on_devices(N300, T3K, P150x4)``), so they are skipped on a single
device. Run on T3K data-parallel:

    DOTS_OCR_PARALLELISM=DP MESH_DEVICE=T3K \
        pytest tests/experimental/dots_ocr/Tier4/test_vllm_page_table_hook.py -x -s \
        --override-ini="addopts="
"""

import pytest
import torch

import ttnn
from tt_symbiote.models.dots_ocr import _create_paged_kv_cache
from tt_symbiote.models.dots_ocr._attention import dp_batch_shard_tensor_mapper
from tt_symbiote.models.dots_ocr.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLayerStack,
)
from tt_symbiote.utils.device_management import set_device

from ..dots_ocr_helpers import (
    assert_pcc,
    canonical_to_torch,
    dots_ocr_device_params,
    mesh_num_devices,
    pipeline_batch_size,
    resolve_mesh_device_shape,
    resolve_model_path,
)

DOTS_OCR_LOCAL_PATH = resolve_model_path()

PCC_ONE_LAYER = 0.99


def _hf_one_layer_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    cfg.num_hidden_layers = 1
    return cfg


def _make_page_table(bs: int, bps: int, shift: int = 0) -> torch.Tensor:
    """Non-identity, per-row-distinct page table ``[bs, bps]``.

    Row ``r`` is a reversed arange rotated by ``r + shift`` (mod ``bps``) so
    that (a) the mapping is non-trivial -- the kind a real block manager hands
    out -- and (b) every DP stream's row differs, which proves the per-device
    shard actually lands distinct rows on distinct devices. All values stay in
    ``[0, bps)`` (valid physical block ids).
    """
    base = torch.arange(bps - 1, -1, -1, dtype=torch.int32)
    rows = [((base + r + shift) % bps).to(torch.int32) for r in range(bs)]
    return torch.stack(rows, dim=0).contiguous()


def _readback_page_table(cache, mesh_device) -> torch.Tensor:
    nd = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    if nd > 1:
        return ttnn.to_torch(cache._tt_page_table, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(
            torch.int32
        )
    return ttnn.to_torch(cache._tt_page_table).to(torch.int32)


def _assert_installed(cache, mesh_device, pt: torch.Tensor, bs: int) -> None:
    """Assert the device page table holds ``pt`` (per-device rows under DP)."""
    nd = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    host = _readback_page_table(cache, mesh_device)
    bps = pt.shape[1]
    if nd > 1 and bs == nd:
        # DP-sharded: device d holds row d. Concat along batch reassembles [bs, bps].
        assert torch.equal(host[:, :bps], pt), "DP-sharded page table rows did not land per-device"
    else:
        # Replicated (or single device): every device holds the (single) table.
        assert torch.equal(host[:1, :bps], pt[:1]), "page table was not installed on device"


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_set_vllm_page_table_roundtrip_and_inplace(mesh_device):
    """TS-1/TS-2: DP-aware install, device round-trip, and in-place (stable) buffer."""
    from transformers import AutoConfig

    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    bs = pipeline_batch_size()

    cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=bs)
    bps = cache.config.blocks_per_sequence

    # --- TS-1: install a non-identity page table and round-trip it off device ---
    pt = _make_page_table(bs, bps, shift=0)
    before_buf = cache._tt_page_table
    before_id = id(before_buf)
    cache.set_vllm_page_table(pt)
    _assert_installed(cache, mesh_device, pt, bs)

    # --- TS-2: the device buffer identity is preserved (in-place copy) ---
    assert id(cache._tt_page_table) == before_id, "set_vllm_page_table reallocated _tt_page_table (breaks trace)"
    assert cache._tt_page_table is before_buf

    # --- TS-2: a second (different) install also stays in place and updates values ---
    pt2 = _make_page_table(bs, bps, shift=7)
    cache.set_vllm_page_table(pt2)
    assert id(cache._tt_page_table) == before_id, "re-install reallocated _tt_page_table"
    _assert_installed(cache, mesh_device, pt2, bs)

    # --- validation: shape / dim guards ---
    with pytest.raises(ValueError):
        cache.set_vllm_page_table(torch.arange(bps, dtype=torch.int32))  # 1-D
    with pytest.raises(ValueError):
        cache.set_vllm_page_table(_make_page_table(bs + 1, bps))  # wrong batch dim
    with pytest.raises(ValueError):
        cache.set_vllm_page_table(torch.zeros(bs, bps + 1, dtype=torch.int32))  # too many blocks

    # --- default mapping is untouched until the hook is called (HF-preserving) ---
    fresh = _create_paged_kv_cache(model_config, mesh_device, batch_size=bs)
    fresh_host = _readback_page_table(fresh, mesh_device)
    identity = torch.arange(bps, dtype=torch.int32)
    assert torch.equal(fresh_host[0, :bps], identity), "default page table should be contiguous identity"


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128], ids=["prefill_32", "prefill_128"])
def test_dots_ocr_paged_prefill_decode_with_vllm_page_table_pcc(mesh_device, seq_len):
    """TS-1 e2e: prefill then decode through the paged ops via a non-identity page table.

    The reversed/rotated page table forces ``paged_fill_on_device`` (prefill
    write) and ``paged_sdpa_decode`` (decode read-back) to follow vLLM's block
    mapping; correctness vs a torch ``DynamicCache`` reference proves the seam.
    """
    if mesh_num_devices() <= 1:
        pytest.skip("dots.ocr attention is arch-gated to multi-device boards (N300/T3K/P150x4)")

    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    cfg = _hf_one_layer_config()
    hf_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config
    hf_layer = hf_model.model.layers[0]
    hf_rotary_emb = hf_model.model.rotary_emb
    hidden = model_config.hidden_size

    # --- HF reference (prefill then decode share one DynamicCache) BEFORE TTNN
    #     setup, since from_torch/preprocess mutate weights in place. ---
    prefill_hidden = torch.randn(1, seq_len, hidden, dtype=torch.bfloat16)
    prefill_pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos_p, sin_p = hf_rotary_emb(prefill_hidden, prefill_pos)
    dynamic_cache = DynamicCache()
    torch_prefill = hf_layer(
        prefill_hidden,
        attention_mask=None,
        position_ids=prefill_pos,
        past_key_values=dynamic_cache,
        use_cache=True,
        position_embeddings=(cos_p, sin_p),
    )[0]
    if torch_prefill.dim() == 2:
        torch_prefill = torch_prefill.unsqueeze(1)

    decode_hidden = torch.randn(1, 1, hidden, dtype=torch.bfloat16)
    decode_pos = torch.tensor([[seq_len]], dtype=torch.long)
    cos_d, sin_d = hf_rotary_emb(decode_hidden, decode_pos)
    torch_decode = hf_layer(
        decode_hidden,
        attention_mask=None,
        position_ids=decode_pos,
        past_key_values=dynamic_cache,
        use_cache=True,
        position_embeddings=(cos_d, sin_d),
    )[0]
    if torch_decode.dim() == 2:
        torch_decode = torch_decode.unsqueeze(1)

    # --- TTNN paged path ---
    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer)
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()
    set_device(layer, mesh_device)

    bs = pipeline_batch_size()
    cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=bs)
    bps = cache.config.blocks_per_sequence
    cache.set_vllm_page_table(_make_page_table(bs, bps, shift=0))

    # DP replicates the same prompt across streams; canonical_to_torch reads row 0.
    prefill_in = prefill_hidden.expand(bs, seq_len, hidden).contiguous() if bs > 1 else prefill_hidden
    decode_in = decode_hidden.expand(bs, 1, hidden).contiguous() if bs > 1 else decode_hidden

    tt_prefill_hidden = ttnn.from_torch(
        prefill_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    prefill_cache_pos = ttnn.from_torch(
        torch.arange(seq_len, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_prefill = layer.forward(tt_prefill_hidden, past_key_value=cache, cache_position=prefill_cache_pos)[0]
    ttnn.synchronize_device(mesh_device)
    tt_prefill = canonical_to_torch(out_prefill, mesh_device).to(torch.bfloat16).reshape(torch_prefill.shape)
    assert_pcc(tt_prefill, torch_prefill, threshold=PCC_ONE_LAYER, msg=f"DotsOCR[vllm_pt prefill.{seq_len}]")

    tt_decode_hidden = ttnn.from_torch(
        decode_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    decode_cache_pos = ttnn.from_torch(
        torch.tensor([seq_len], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_decode = layer.forward(tt_decode_hidden, past_key_value=cache, cache_position=decode_cache_pos)[0]
    ttnn.synchronize_device(mesh_device)
    tt_decode = canonical_to_torch(out_decode, mesh_device).to(torch.bfloat16).reshape(torch_decode.shape)
    assert_pcc(tt_decode, torch_decode, threshold=PCC_ONE_LAYER, msg=f"DotsOCR[vllm_pt decode@{seq_len}]")


@pytest.mark.parametrize("device_params", [dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_per_stream_positions_no_collapse(mesh_device):
    """TS-3: per-DP-stream positions are preserved (not collapsed to a global scalar).

    Activates per-stream positions on the layer stack, feeds a DP-sharded
    position vector with a DISTINCT value per device (including a clamped-to-0
    sentinel for an inactive stream), and round-trips the stable per-stream
    position buffer to confirm each device kept its own position.
    """
    bs = pipeline_batch_size()
    nd = mesh_num_devices()
    if nd <= 1 or bs <= 1:
        pytest.skip("per-stream positions require a DP mesh (batch_size == num_devices > 1)")

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    cfg = _hf_one_layer_config()
    hf_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(dtype=torch.bfloat16).eval()

    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_model.model.layers[0])
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()
    decoder_stack = TTNNDotsOCRLayerStack([layer])
    decoder_stack._unique_name = "model.layer_stack"
    set_device(decoder_stack, mesh_device)

    mapper = dp_batch_shard_tensor_mapper(mesh_device, bs)
    assert mapper is not None, "expected a DP batch shard mapper on a DP mesh"
    decoder_stack.enable_per_stream_positions(bs, mapper)
    assert getattr(decoder_stack, "_per_stream_positions", False) is True
    assert getattr(decoder_stack, "_shared_decode_cur_pos_dp", None) is not None

    # Distinct position per stream; device 0 uses 0 (a clamped inactive sentinel).
    # Keep the host vector 1-D [bs] so the DP shard mapper produces a [1] shard
    # per device (matching the cur_pos shape); a post-shard reshape to (bs,)
    # would run per-shard (volume 1) and fail the volume check.
    positions = torch.tensor([i * 5 for i in range(bs)], dtype=torch.int32)
    cache_position = ttnn.from_torch(
        positions,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    materialized = decoder_stack._materialize_shared_cur_pos(cache_position)
    assert materialized is decoder_stack._shared_decode_cur_pos_dp, "per-stream path must use the DP position buffer"
    ttnn.synchronize_device(mesh_device)

    host_pos = (
        ttnn.to_torch(
            decoder_stack._shared_decode_cur_pos_dp,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        .to(torch.int32)
        .reshape(-1)
    )
    assert torch.equal(host_pos[:bs], positions.reshape(-1)), (
        f"per-stream positions were collapsed/lost: got {host_pos[:bs].tolist()}, "
        f"expected {positions.reshape(-1).tolist()}"
    )
