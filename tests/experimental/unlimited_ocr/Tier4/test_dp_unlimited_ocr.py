# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier4 DATA-PARALLEL (DP) tests for baidu/Unlimited-OCR on a (4,1) Blackhole mesh.

Validates the ADDITIVE DP path: N DIFFERENT streams run CONCURRENTLY, one per device
(device b -> stream b). Weights REPLICATE across the mesh (base TTNNLinear via
set_device), per-stream inputs BATCH-SHARD on dim 0 (ttnn.ShardTensorToMesh /
dp_batch_shard_tensor_mapper), shared inputs (cos/sin/vision_idx/vision_mask/cur_pos)
REPLICATE, and outputs GATHER via ttnn.ConcatMeshToTensor(dim=0).

  * ``test_dp_text_lm_e2e`` -- 4 DIFFERENT token sequences batch-sharded; each stream's
    logits PCC >= 0.99 vs its OWN torch per-sequence reference (embedding + 12 decoder
    layers [Llama MHA + dense L0 / MoE L1..11] + final RMSNorm + lm_head, batched across
    the mesh). The core DP mechanism gate.
  * ``test_dp_vlm_pipeline_generate`` -- 4 DIFFERENT sample docs OCR'd concurrently via
    ``TTNNUnlimitedOcrPipeline.generate_dp`` (batched_vision=True); each stream's first
    tokens match its own torch reference (k-1 tolerance) and the 4 outputs are distinct.

DEADLOCK AVOIDANCE (non-negotiable): the (4,1) mesh is opened IN-PROCESS by the
module-scoped ``dp_run`` fixture and closed in ``finally``. This hardware is
SINGLE-TENANT (tt-metal takes a whole-cluster CHIP_IN_USE lock across all 4 chips), so
we must NEVER open a device inside a CHILD subprocess while a parent process also holds
one, and NEVER nest device opens. A prior version spawned a ``dp_child.py`` subprocess
that opened the (4,1) mesh -> the parent waited on the child while the child waited for
chips the parent held -> deadlock. This module therefore does ALL hardware work directly
in-process and does NOT use the conftest (1,1) ``device`` fixture.

RUN THIS TEST IN ITS OWN PYTEST INVOCATION (opt-in). Opening a (4,1) mesh and later a
(1,1) mesh in the SAME process corrupts ttnn's global mesh state: a subsequently-opened
(1,1) device produces tensors carrying a multi-shard ``ConcatMeshToTensor`` composer, so
``ttnn.to_torch`` fails with ``MeshCoordinate([1, 0]) out of bounds for MeshShape([1,
1])``. That is a tt-metal framework limitation, independent of this DP code (the DP path
itself passes cleanly in isolation). To keep the single-device suite
(``pytest tests/experimental/unlimited_ocr/``) green, this module is marked
``@pytest.mark.dp`` and SKIPS unless ``--run-dp`` is passed, so merely collecting it
alongside the (1,1) tests never opens the (4,1) mesh. Run the DP validation on its own::

    python -m pytest --run-dp \
        tests/experimental/unlimited_ocr/Tier4/test_dp_unlimited_ocr.py -q

Skips when the host does not expose 4 devices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DP = 4
PCC_TEXT = 0.99
_REPO = Path(__file__).resolve().parents[4]
_DEMO_DIR = _REPO / "examples" / "e2e" / "unlimited_ocr"
# Opt-in: these tests are skipped unless `--run-dp` is passed (see conftest).
pytestmark = pytest.mark.dp


@pytest.fixture(scope="module")
def dp_run(tmp_path_factory):
    """Open the (4,1) mesh IN-PROCESS ONCE, run Stage-1 (text logits PCC) + Stage-2
    (VLM generate_dp torch-match), close the mesh in finally, return a results dict.

    ALL hardware work happens in THIS process on the (4,1) mesh only -- no subprocess,
    no nested device opens (see the module docstring for the deadlock rationale)."""
    import torch
    import ttnn

    sys.path.insert(0, str(_DEMO_DIR))
    from tests.shared.pcc_utils import compute_pcc
    from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
        TTNNUnlimitedOcrForCausalLM,
    )
    from tt_symbiote.models.unlimited_ocr.pipeline import TTNNUnlimitedOcrPipeline
    from tt_symbiote.models.unlimited_ocr.reference_loader import load_reference_model
    from tt_symbiote.utils.device_management import set_device
    import run_unlimited_ocr as demo
    import sample_images
    from transformers import AutoTokenizer

    tmp = tmp_path_factory.mktemp("dp")
    res = {"pcc": [], "match": [], "distinct": False}
    model, cfg = load_reference_model()
    model.eval()

    # ---- open (4,1) mesh IN-PROCESS (skip if unavailable) ----
    # trace_region_size: needed to capture the batch-B decode trace on the mesh.
    try:
        dev = ttnn.open_mesh_device(ttnn.MeshShape(DP, 1), l1_small_size=32768,
                                    trace_region_size=400_000_000)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"DP tests require a 4-device Blackhole host (open failed): "
                    f"{type(e).__name__}: {e}")
    if dev.get_num_devices() != DP:
        n = dev.get_num_devices()
        ttnn.close_mesh_device(dev)
        pytest.skip(f"DP tests require {DP} devices; host exposes {n}")

    try:
        # ===== Stage 1: text-only DP logits PCC =====
        seq = 8
        torch.manual_seed(0)
        ids = torch.randint(0, cfg.vocab_size, (DP, seq), dtype=torch.int64)
        with torch.no_grad():
            refs = [
                model(input_ids=ids[b:b + 1], images=None, use_cache=False,
                      return_dict=True).logits.float()
                for b in range(DP)
            ]
        ref = torch.cat(refs, dim=0)
        tt = TTNNUnlimitedOcrForCausalLM.from_torch(model)
        set_device(tt, dev)
        rot = model.model.layers[0].self_attn.rotary_emb
        pos = torch.arange(seq).unsqueeze(0)
        with torch.no_grad():
            cos, sin = rot(torch.zeros(1, 1, seq, cfg.head_dim), pos)
        cos = cos.unsqueeze(1).float()
        sin = sin.unsqueeze(1).float()
        rep = ttnn.ReplicateTensorToMesh(dev)
        tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, mesh_mapper=rep)
        tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, mesh_mapper=rep)
        tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                 mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0))
        out = tt.forward(input_ids=tt_ids, position_embeddings=(tt_cos, tt_sin))
        if hasattr(out, "ttnn_tensor") and out.ttnn_tensor is not None:
            out = out.ttnn_tensor
        got = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0)
        ).float().reshape(DP, seq, cfg.vocab_size)
        res["finite"] = bool(torch.isfinite(got).all())
        for b in range(DP):
            res["pcc"].append(float(compute_pcc(got[b:b + 1], ref[b:b + 1])[0][0]))

        # ===== Stage 2: VLM DP generate_dp + per-stream torch-match =====
        tokenizer = AutoTokenizer.from_pretrained(demo.MODEL_ID, trust_remote_code=True)
        specs = sample_images.default_batch_specs(
            DP, tmp / "docs", (demo.BASE_SIZE, demo.BASE_SIZE))
        pxs, names = [], []
        for s in specs:
            pil, name = demo._load_pil(s)
            pxs.append(demo._preprocess_pixels(pil))
            names.append(name)
        px_batch = torch.cat(pxs, dim=0).permute(0, 2, 3, 1).contiguous()
        base_ids, base_mask = demo._build_input_ids(tokenizer, demo.DEFAULT_PROMPT)
        pipe = TTNNUnlimitedOcrPipeline.from_hf_model(model, cfg, dev,
                                                      batched_vision=True)
        ids_b = [list(base_ids) for _ in range(DP)]
        # Decode is always traced -> capture the batch-B decode trace once before timing.
        pipe.warmup_dp(ids_b, list(base_mask), px_batch)
        gen = pipe.generate_dp(ids_b, list(base_mask),
                               px_batch, max_new_tokens=32, stop_on_eos=True)
        res["decode_s"] = float(pipe.last_decode_s)
        res["num_decode"] = int(pipe.last_num_decode)
        for b in range(DP):
            if gen[b] and gen[b][-1] == demo.EOS_ID:
                gen[b] = gen[b][:-1]
        inner = model.model
        outs = []
        for b in range(DP):
            n_match, k, _tt_head, _torch_head, ok = demo._torch_match_one(
                model, inner, tokenizer, pxs[b], base_ids, base_mask, gen[b], 8, 32)
            res["match"].append([int(n_match), int(k), bool(ok)])
            outs.append(demo._decode_tokens(tokenizer, gen[b]).strip())
        res["distinct"] = (len(set(outs)) == DP)
    finally:
        ttnn.close_mesh_device(dev)
    return res


def test_dp_text_lm_e2e(dp_run):
    """4 DIFFERENT token sequences batch-sharded -> each stream's logits PCC >= 0.99."""
    assert dp_run.get("finite", False), "DP text logits contain NaN/Inf"
    pccs = dp_run["pcc"]
    assert len(pccs) == DP
    for b, pcc in enumerate(pccs):
        print(f"[DP text] stream {b}: logits PCC={pcc:.5f}")
        assert pcc >= PCC_TEXT, f"stream {b} logits PCC {pcc:.5f} < {PCC_TEXT}"


def test_dp_vlm_pipeline_generate(dp_run):
    """4 DIFFERENT sample docs via generate_dp -> each stream matches torch (k-1 tol),
    and the 4 outputs are DISTINCT (device b really OCR'd image b)."""
    matches = dp_run["match"]
    assert len(matches) == DP
    for b, (n_match, k, ok) in enumerate(matches):
        print(f"[DP vlm] stream {b}: {n_match}/{k} torch-match")
        assert ok, f"stream {b} diverges from torch reference ({n_match}/{k})"
    assert dp_run["distinct"], "4 DIFFERENT images must produce 4 DISTINCT OCR outputs"


def test_dp_decode_throughput(dp_run):
    """Report the aggregate DP decode throughput (traced) across the mesh."""
    s = dp_run.get("decode_s", 0.0)
    d = dp_run.get("num_decode", 0)
    if s > 0:
        print(f"[DP throughput] TRACED-DP aggregate {d * DP / s:.2f} tok/s "
              f"({d} steps in {s:.2f}s across {DP} streams)")
