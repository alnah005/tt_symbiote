# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy profiling driver for the Unlimited-OCR VISION PREFILL (DeepEncoder + projector).

Profiles a SINGLE DeepEncoder+projector forward on the fixed 1024x1024 global view
-- the compute-bound vision tower (SAM ViT-B 12 blocks + neck/net convs + CLIP-L
24 blocks + Linear(2048->1280) projector) that holds the bfloat8_b bottleneck
matmuls. This is the ~10.7s prefill bottleneck; the vision blocks are NOT limited.

A warm-up forward runs first (kernel compile), then a ``tracy.signpost``-delimited
measured forward. Report windows on the ``VISION_PREFILL`` signpost so warm-up ops
are excluded:

    export TT_METAL_HOME=/home/ttuser/salnahari/tt-metal
    export TT_SYMBIOTE_SIGNPOST_MODE=1
    python -m tracy -p -r -v --op-support-count 20000 \
      tests/experimental/unlimited_ocr/profiling/profile_vision_prefill.py

    tt-perf-report --start-signpost VISION_PREFILL \
      generated/profiler/.../ops_perf_results_*.csv > perf_report.txt

Device time is read SOLELY from the tracy ops_perf_results_*.csv DEVICE TIME (ns)
column -- never estimated.
"""

import torch
import ttnn
from tracy import signpost

from tt_symbiote.models.unlimited_ocr.reference_loader import load_reference_model
from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
    TTNNUnlimitedOcrDeepEncoder,
    TTNNUnlimitedOcrMlpProjector,
)
from tt_symbiote.utils.device_management import set_device


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        model, _ = load_reference_model()
        inner = model.model if hasattr(model, "model") else model
        sam, vis, proj_t = inner.sam_model, inner.vision_model, inner.projector

        de = TTNNUnlimitedOcrDeepEncoder.from_torch(sam, vis)
        set_device(de, dev)
        proj = TTNNUnlimitedOcrMlpProjector.from_torch(proj_t)
        set_device(proj, dev)

        # Fixed 1024x1024 global view, channels-last (NHWC) as SAM expects.
        torch.manual_seed(0)
        px = (torch.rand(1, 3, 1024, 1024) - 0.5) / 0.5
        px_nhwc_t = px.permute(0, 2, 3, 1).contiguous()

        def run_once():
            px_nhwc = ttnn.from_torch(
                px_nhwc_t, dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
            )
            feats = de.forward(px_nhwc)
            feats_bf16 = ttnn.to_torch(feats)
            feats_in = ttnn.from_torch(
                feats_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
            )
            out = proj.forward(feats_in)
            return out

        # ---- Warm-up: compile all vision kernels (NOT measured) ----
        _ = run_once()
        ttnn.synchronize_device(dev)

        # ---- Signposted measured vision prefill ----
        signpost(header="VISION_PREFILL")
        out = run_once()
        ttnn.synchronize_device(dev)
        signpost(header="VISION_PREFILL_END")

        out_t = ttnn.to_torch(out)
        print(f"[vision prefill] projector out shape={tuple(out_t.shape)} finite={torch.isfinite(out_t).all().item()}")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
