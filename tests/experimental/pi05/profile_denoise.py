# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy profiling driver for the pi0.5 denoise step (op-level device profile).

Run under tracy to capture an ops_perf_results CSV dominated by the per-step
denoise graph (the 18-layer Gemma-300M adaRMS expert + action projection):

    MESH_DEVICE=P150 TT_METAL_HOME=... PI05_REFERENCE_ROOT=... \
      python_env/bin/python -m tracy -p -r -v --op-support-count 50000 \
      tests/capabilities/pi05/profile_denoise.py

Then: tt-perf-report --ignore-signposts generated/profiler/reports/*/ops_perf_results_*.csv

Builds the model + prefix KV cache once, warms up, then runs the denoise step
N times so MLP / SDPA / attention-matmul / norm ops dominate the aggregate.
"""

import math
import os

import torch
import ttnn

from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
from tt_symbiote.models.pi05.modeling_pi05 import TTNNPi05Model, load_reference_pi05_model
from tt_symbiote.utils.device_management import set_device

N_PROFILE = int(os.environ.get("PI05_PROFILE_STEPS", "10"))
CKPT = os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base")


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=134_217_728)
    try:
        cfg = Pi0_5ModelConfig()
        ref = load_reference_pi05_model(CKPT)
        tt = TTNNPi05Model.from_torch(ref, cfg)
        set_device(tt, dev)

        # prefix + KV cache (profiled once -- shows SigLIP/VLM/non-matmul ops too)
        L = 32
        imgs = [ttnn.from_torch(torch.randn(1, 3, 224, 224), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
        lt = ttnn.from_torch(
            torch.randint(0, 257152, (1, L)).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
        )
        ie = tt.backbone.embed_image(imgs[0])
        le = ttnn.multiply(tt.backbone.embed_language_tokens(lt), math.sqrt(cfg.vlm_config.width))
        prefix = ttnn.concat([ie, le], dim=1)
        pl = prefix.shape[1]
        _, cache = tt.backbone.forward_vlm(prefix, attention_mask=None, use_cache=True)

        ahp = tt._tile_pad(cfg.action_horizon)
        x0 = tt._noise_to_device(torch.randn(1, cfg.action_horizon, cfg.action_dim), ahp)
        se = tt.suffix_embedding.embed_actions(x0)
        adarms = tt.suffix_embedding.embed_adarms_cond(tt._ts(0.9))
        mask = tt._suffix_phantom_mask(pl, cfg.action_horizon, ahp)
        block_mods, final_mod = tt.backbone.precompute_step_mods(adarms)

        def step():
            eo = tt.backbone.forward_expert(
                se,
                past_key_values=cache,
                attention_mask=mask,
                position_offset=pl,
                precomputed_block_mods=block_mods,
                precomputed_final_mod=final_mod,
            )
            return tt.suffix_embedding.project_output(eo)

        step()  # warm-up (kernel compile)
        ttnn.synchronize_device(dev)
        for _ in range(N_PROFILE):
            step()
        ttnn.synchronize_device(dev)
        print(f"profiled {N_PROFILE} denoise steps (prefix_len={pl})")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
