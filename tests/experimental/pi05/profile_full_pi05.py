# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-model tracy profiling driver for pi0.5, WITH signpost markers.

Runs the complete VLA pipeline (SigLIP vision -> projector -> Gemma-2B VLM
prefill -> 10-step flow-matching denoise through the Gemma-300M adaRMS expert ->
action projection) and inserts ``tracy.signpost`` markers between phases so
``tt-perf-report`` segments the device profile per phase.

Run under tracy:

    MESH_DEVICE=P150 TT_METAL_HOME=... PI05_REFERENCE_ROOT=... PI05_CHECKPOINT_DIR=... \
      python_env/bin/python -m tracy -p -r -v --op-support-count 100000 \
      tests/capabilities/pi05/profile_full_pi05.py

Then:  tt-perf-report generated/profiler/reports/*/ops_perf_results_*.csv

A warm-up full inference runs first (kernel compile + step-mod cache); the
signposted run is what the report reflects.
"""

import math
import os

import torch
import ttnn
from tracy import signpost

from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
from tt_symbiote.models.pi05.modeling_pi05 import TTNNPi05Model, load_reference_pi05_model
from tt_symbiote.utils.device_management import set_device

CKPT = os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base")


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=134_217_728)
    try:
        cfg = Pi0_5ModelConfig()
        ref = load_reference_pi05_model(CKPT)
        tt = TTNNPi05Model.from_torch(ref, cfg)
        set_device(tt, dev)

        L = 32
        imgs_t = torch.randn(1, 3, 224, 224)
        lt_t = torch.randint(0, 257152, (1, L)).to(torch.int32)
        noise = torch.randn(1, cfg.action_horizon, cfg.action_dim)

        def upload():
            imgs = [ttnn.from_torch(imgs_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
            im = [ttnn.from_torch(torch.ones(1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
            lt = ttnn.from_torch(lt_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            lm = ttnn.from_torch(torch.ones(1, L), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            return imgs, im, lt, lm

        # Warm-up: full inference (compiles kernels + caches per-step adaRMS mods).
        imgs, im, lt, lm = upload()
        _ = tt.sample_actions(imgs, im, lt, lm, noise=noise)
        ttnn.synchronize_device(dev)

        # ---- Signposted full inference ----
        imgs, im, lt, lm = upload()
        ah, ad = cfg.action_horizon, cfg.action_dim
        ahp = tt._tile_pad(ah)

        signpost(header="PREFIX_SIGLIP_VISION")  # SigLIP 27 blocks + multimodal projector
        img_embeds = tt.backbone.embed_image(imgs[0])

        signpost(header="PREFIX_LANG_EMBED")  # Gemma embedding * sqrt(width) + concat
        lang = ttnn.multiply(tt.backbone.embed_language_tokens(lt), math.sqrt(cfg.vlm_config.width))
        prefix = ttnn.concat([img_embeds, lang], dim=1)
        prefix_len = prefix.shape[1]

        signpost(header="VLM_PREFILL_18L")  # Gemma-2B VLM, 18 layers, fills KV cache
        _, vlm_cache = tt.backbone.forward_vlm(prefix, attention_mask=None, use_cache=True)

        x_t = tt._noise_to_device(noise, ahp)
        suffix_mask = tt._suffix_phantom_mask(prefix_len, ah, ahp)
        step_mods = tt._ensure_step_mods()

        signpost(header="DENOISE_LOOP_10x_EXPERT")  # 10 Euler steps, 18-layer adaRMS expert each
        n = cfg.num_denoising_steps
        for i in range(n):
            bmods, fmod = step_mods[i]
            v = tt._denoise_forward(x_t, vlm_cache, prefix_len, suffix_mask, bmods, fmod)
            v_dt = ttnn.multiply(v, -1.0 / n, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(v)
            x_nxt = ttnn.add(x_t, v_dt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(v_dt)
            ttnn.deallocate(x_t)
            x_t = x_nxt

        signpost(header="OUTPUT_SLICE")
        out = ttnn.slice(x_t, [0, 0, 0], [x_t.shape[0], ah, ad])
        ttnn.synchronize_device(dev)
        print(f"full inference profiled (prefix_len={prefix_len}, action {tuple(ttnn.to_torch(out).shape)})")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
