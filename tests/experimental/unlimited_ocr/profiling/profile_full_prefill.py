# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy the FULL warm VLM prefill (vision DeepEncoder + projector + LM 12 layers +
lm_head) to see the device-time breakdown -- where the stable ~10.5s wall goes, and
in particular how much is LM-prefill MoE (still the dense 64-expert loop over ~277
tokens x 11 layers) vs vision vs lm_head vs host (wall - device_sum).

Run on chips 2,3:
    export TT_VISIBLE_DEVICES=2,3 TT_METAL_VISIBLE_DEVICES=2,3 TT_SYMBIOTE_SIGNPOST_MODE=1
    export TT_METAL_HOME=/home/ttuser/salnahari/tt-metal
    python -m tracy -p -r -v --op-support-count 60000 \
      tests/experimental/unlimited_ocr/profiling/profile_full_prefill.py
    tt-perf-report --start-signpost PREFILL_MEAS <reports>/ops_perf_results_*.csv
"""
import time

import torch
import ttnn
from tracy import signpost
from transformers import AutoTokenizer

from tt_symbiote.models.unlimited_ocr.pipeline import TTNNUnlimitedOcrPipeline
from tt_symbiote.models.unlimited_ocr.reference_loader import load_reference_model

MODEL_ID = "baidu/Unlimited-OCR"


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        model, cfg = load_reference_model()
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        pipe = TTNNUnlimitedOcrPipeline.from_hf_model(model, cfg, dev, enable_trace=False)

        IMAGE_TOKEN_ID, BOS_ID, NUM_VISION_TOKENS = 128815, 0, 273
        post = tok.encode("\nFree OCR.", add_special_tokens=False)
        ids = [BOS_ID] + [IMAGE_TOKEN_ID] * NUM_VISION_TOKENS + [int(t) for t in post]
        seq_mask = [False] + [True] * NUM_VISION_TOKENS + [False] * len(post)

        def mk_px():
            px = ((torch.rand(1, 3, 1024, 1024) - 0.5) / 0.5).permute(0, 2, 3, 1).contiguous()
            return ttnn.from_torch(px, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

        # warm-up prefill (kernel compile)
        pipe.prefill(list(ids), seq_mask=list(seq_mask), tt_px=mk_px())
        ttnn.synchronize_device(dev)

        # measured prefill under signpost
        px = mk_px()
        ttnn.synchronize_device(dev)
        signpost("PREFILL_MEAS")
        t = time.perf_counter()
        pipe.prefill(list(ids), seq_mask=list(seq_mask), tt_px=px)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        signpost("PREFILL_END")
        print(f"[full-prefill] measured warm prefill wall = {wall:.2f} s", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
