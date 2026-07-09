# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy profiling driver for the Unlimited-OCR LM DECODE step (MoE grouping A/B).

Isolates 16 NORMAL (non-traced) decode steps under a ``DECODE_MEAS`` signpost so
each per-op DEVICE KERNEL DURATION is visible (traced replay would collapse them
into one blob). Text-only prompt -> skips the ~10s vision prefill; the MoE lives in
the LM decode. Run it TWICE for the grouped-vs-dense A/B:

    export TT_METAL_HOME=/home/ttuser/salnahari/tt-metal
    export TT_SYMBIOTE_SIGNPOST_MODE=1

    # BEFORE (dense 64-expert loop):
    TT_UNLIMITED_OCR_MOE_DENSE=1 python -m tracy -p -r -v --op-support-count 40000 \
      tests/experimental/unlimited_ocr/profiling/profile_lm_decode.py

    # AFTER (grouped 3D-batched matmul):
    python -m tracy -p -r -v --op-support-count 40000 \
      tests/experimental/unlimited_ocr/profiling/profile_lm_decode.py

    tt-perf-report --start-signpost DECODE_MEAS <reports>/ops_perf_results_*.csv

Device time is read SOLELY from the tracy ops_perf_results_*.csv DEVICE TIME (ns)
column -- never estimated.
"""
import ttnn
from tracy import signpost
from transformers import AutoTokenizer

from tt_symbiote.models.unlimited_ocr.pipeline import TTNNUnlimitedOcrPipeline
from tt_symbiote.models.unlimited_ocr.reference_loader import load_reference_model

MODEL_ID = "baidu/Unlimited-OCR"
WARMUP_STEPS = 2
MEAS_STEPS = 16


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        model, cfg = load_reference_model()
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        pipe = TTNNUnlimitedOcrPipeline.from_hf_model(model, cfg, dev, enable_trace=False)

        ids = tok("The capital of France is", return_tensors=None)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(t) for t in ids]
        prompt_len = len(ids)

        # prefill (text-only) + warm-up decode steps (kernel compile)
        cur = pipe.prefill(list(ids))
        for i in range(WARMUP_STEPS):
            cur = pipe.decode_step(cur, prompt_len + i)
        ttnn.synchronize_device(dev)

        # measured decode window
        signpost("DECODE_MEAS")
        for i in range(MEAS_STEPS):
            cur = pipe.decode_step(cur, prompt_len + WARMUP_STEPS + i)
        ttnn.synchronize_device(dev)
        signpost("DECODE_END")
        print(f"[decode-profile] {MEAS_STEPS} measured decode steps done", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
