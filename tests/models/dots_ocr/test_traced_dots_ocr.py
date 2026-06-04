# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bottom-up traced-execution tests for rednote-hilab/dots.ocr.

Validates the TracedRun 3-phase lifecycle (warmup -> capture -> replay), per
``@trace_enabled`` module, from the smallest units upward so a failure isolates
to a specific module instead of the whole pipeline:

  Tier 1 (ops):        TTNNLinear, TTNNLocalRMSNorm, TTNNEmbedding
  Tier 3 (decoder):    TTNNDotsOCRDecoderLayer  (decode, paged KV cache)
  Tier 4 (full model): TTNNDotsOCRPipeline.generate  -- RUN LAST

All phases run under TT_SYMBIOTE_RUN_MODE=TRACED (set below, before any model
call). Each module is invoked via ``__call__`` (NOT ``.forward``) so the
TracedRun dispatch drives warmup/capture/replay. Greedy/deterministic compute
means replay output must match the warmup output (PCC >= 0.999); a corrupted
or freed trace buffer surfaces as a divergence or a "Buffer is not allocated".

Mesh: MESH_DEVICE=T3K + DOTS_OCR_PARALLELISM=DP -> (8, 1) data-parallel.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import os

import pytest
import torch
from torch import nn

# TRACED must be selected before any TTNNModule.__call__ dispatches.
os.environ.setdefault("TT_SYMBIOTE_RUN_MODE", "TRACED")

import ttnn
from tt_symbiote.core.run_config import TracedRun
from tt_symbiote.core.tensor import TorchTTNNTensor
from tt_symbiote.utils.device_management import set_device

from tests.shared.pcc_utils import assert_pcc


# ---------------------------------------------------------------------------
# 3-phase helper
# ---------------------------------------------------------------------------


# Device params: the conftest is a dumb mesh-opener, so each test declares the
# exact mesh shape it needs (dots.ocr decode/pipeline run data-parallel (8,1)).
_SINGLE = {"trace_region_size": 300_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.DISABLED}
_DP = {"trace_region_size": 300_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}


def _run_3phase(call_fn):
    """Invoke call_fn() three times (warmup, capture, replay) under TRACED.

    Returns (warmup_out, capture_out, replay_out). call_fn must call the module
    via __call__ with identical-shape inputs each time (so the trace cache key
    is stable across the three phases). On any failure we reset the global
    ``_TRACE_RUNNING`` flag so a crash mid-capture doesn't leak into the next
    test (which would otherwise assert "Weights must be preprocessed...").
    """
    import tt_symbiote.core.run_config as _rc

    try:
        warmup_out = call_fn()
        capture_out = call_fn()
        replay_out = call_fn()
        return warmup_out, capture_out, replay_out
    finally:
        _rc._TRACE_RUNNING = False


# ===========================================================================
# Tier 1 -- leaf @trace_enabled ops (stateless; should trace cleanly)
# ===========================================================================


@pytest.mark.parametrize("device_params", [_SINGLE], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_traced_tier1_linear(mesh_device):
    """TTNNLinear: warmup -> capture -> replay parity."""
    from tt_symbiote.modules.ttnn_linear import TTNNLinear

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    lin = nn.Linear(1536, 1536, bias=False).to(torch.bfloat16).eval()
    tt = TTNNLinear.from_torch(lin)
    set_device(tt, mesh_device)

    inp = TorchTTNNTensor(torch.randn(1, 128, 1536, dtype=torch.bfloat16))
    w, c, r = _run_3phase(lambda: tt(inp))

    assert_pcc(c, w, threshold=0.999, msg="linear.capture_vs_warmup")
    assert_pcc(r, w, threshold=0.999, msg="linear.replay_vs_warmup")
    TracedRun.release_all()


@pytest.mark.parametrize("device_params", [_SINGLE], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_traced_tier1_rmsnorm(mesh_device):
    """TTNNLocalRMSNorm (eps attr): warmup -> capture -> replay parity."""
    from tt_symbiote.modules.ttnn_normalization import TTNNLocalRMSNorm

    class _RMS(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(dim))
            self.eps = eps

        def forward(self, x):
            out = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
            return out * self.weight

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    norm = _RMS(1536).to(torch.bfloat16).eval()
    tt = TTNNLocalRMSNorm.from_torch(norm)
    set_device(tt, mesh_device)

    inp = TorchTTNNTensor(torch.randn(1, 128, 1536, dtype=torch.bfloat16))
    w, c, r = _run_3phase(lambda: tt(inp))

    assert_pcc(c, w, threshold=0.999, msg="rmsnorm.capture_vs_warmup")
    assert_pcc(r, w, threshold=0.999, msg="rmsnorm.replay_vs_warmup")
    TracedRun.release_all()


@pytest.mark.parametrize("device_params", [_SINGLE], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_traced_tier1_embedding(mesh_device):
    """TTNNEmbedding: warmup -> capture -> replay parity."""
    from tt_symbiote.models.dots_ocr import TTNNEmbedding

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    emb = nn.Embedding(2048, 1536).to(torch.bfloat16).eval()
    tt = TTNNEmbedding.from_torch(emb)
    set_device(tt, mesh_device)

    ids = TorchTTNNTensor(torch.randint(0, 2048, (1, 128), dtype=torch.int32))
    w, c, r = _run_3phase(lambda: tt(ids))

    assert_pcc(c, w, threshold=0.999, msg="embedding.capture_vs_warmup")
    assert_pcc(r, w, threshold=0.999, msg="embedding.replay_vs_warmup")
    TracedRun.release_all()


# ===========================================================================
# Tier 3 -- decoder layer (stateful: paged KV cache)
# ===========================================================================


@pytest.mark.parametrize("device_params", [_DP], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
def test_traced_tier3_decoder_layer(mesh_device):
    """TTNNDotsOCRDecoderLayer decode: warmup -> capture -> replay parity."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from tt_symbiote.models.dots_ocr import TTNNDotsOCRDecoderLayer, _create_paged_kv_cache

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    cfg = AutoConfig.from_pretrained("rednote-hilab/dots.ocr", trust_remote_code=True)
    cfg.num_hidden_layers = 1
    hf = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(torch.bfloat16).eval()
    cfg = hf.config

    layer = TTNNDotsOCRDecoderLayer.from_torch(hf.model.layers[0])
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()
    set_device(layer, mesh_device)
    layer.preprocess_weights()
    layer.move_weights_to_device()

    paged = _create_paged_kv_cache(cfg, mesh_device, batch_size=1)
    hs = ttnn.from_torch(
        torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    def _call():
        paged.reset()
        cp = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return layer(hs, past_key_value=paged, cache_position=cp)[0]

    w, c, r = _run_3phase(_call)
    assert_pcc(c, w, threshold=0.99, msg="decoder.capture_vs_warmup")
    assert_pcc(r, w, threshold=0.99, msg="decoder.replay_vs_warmup")
    TracedRun.release_all()


# ===========================================================================
# Tier 4 -- full pipeline (RUN LAST)
# ===========================================================================


@pytest.mark.parametrize("device_params", [_DP], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
def test_traced_tier4_pipeline_last(mesh_device):
    """Full TTNNDotsOCRPipeline traced generate -- run LAST.

    Known issue under investigation: the prefill graph's trace-input buffer is
    deallocated at replay. This test is the top of the bottom-up ladder; it is
    expected to surface that until the prefill-graph input lifetime is fixed.
    """
    from transformers import AutoTokenizer

    from tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline

    torch.set_grad_enabled(False)
    num = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    batch = num if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP" and num > 1 else 1

    pipe = TTNNDotsOCRPipeline.from_hf_model(
        model_path="rednote-hilab/dots.ocr", device=mesh_device, batch_size=batch
    )
    tok = AutoTokenizer.from_pretrained("rednote-hilab/dots.ocr", trust_remote_code=True)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "The capital of France is"}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )["input_ids"]
    if batch > 1:
        ids = ids.expand(batch, -1).contiguous()

    pipe.warmup(ids)
    out = pipe.generate(ids, max_new_tokens=16, stop_on_eos=False)
    new = out[0] if (out and isinstance(out[0], list)) else out
    text = tok.decode(new, skip_special_tokens=True)
    print(f"\n[traced tier4] {len(new)} tokens: {text!r}\n")
    assert len(set(new)) >= 4, f"degenerate traced output: {new[:16]}"
    pipe.release()
