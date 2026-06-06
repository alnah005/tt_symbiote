# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure the pi0.5 inference chunk timing, stage by stage.

Run the SAME script under two run modes for a before/after on the prefill:

  TT_SYMBIOTE_RUN_MODE=TRACED  -> times trace REPLAY (warmup->capture->replay)
  TT_SYMBIOTE_RUN_MODE=NORMAL  -> times EAGER execution

Each traced stage is wrapped in a SINGLE-forward ``@trace_enabled`` adapter (the
proven pattern from the Tier 2-4 traced tests): the adapter captures the whole
stage as one graph and its nested ``@trace_enabled`` children fall back to normal
forward inside the parent capture (no fragile per-submodule auto-capture).

Stages measured (each in isolation, host wall-clock around synchronize_device):
  * tower      : embed_image (SigLIP-27 + mm_projector)
  * forward_vlm: 18-layer Gemma-2B prefill via the trace-safe VLM prefix store
  * denoise    : one flow-matching velocity step (x10 = full denoise loop)

Answers: now that forward_vlm is trace-capturable (VLM prefix store), what is the
fully-traced chunk time (prefix traced + denoise traced) vs the prior 81.4ms
(eager prefill 25.49 + denoise replay 55.86) and the reference ~64.85ms/chunk?
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch

import ttnn
from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices
from tt_symbiote.core.run_config import trace_enabled

RUN_MODE = os.environ.get("TT_SYMBIOTE_RUN_MODE", "NORMAL")
CKPT = os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base")


# --------------------------------------------------------------------------- #
# Single-forward trace adapters (one captured graph each; nested @trace_enabled
# children fall back to normal forward inside the parent capture).
# --------------------------------------------------------------------------- #
def _wrap(cls, device, **attrs):
    m = cls()
    m._bypass_tensor_wrapping = True
    m._device = device
    m._preprocessed_weight = True
    m._weights_on_device = True
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@trace_enabled
class _TowerAdapter(TTNNModule):
    @run_on_devices(DeviceArch.P150)
    def forward(self, image):
        return self._bk.embed_image(image)  # SigLIP tower + mm_projector


@trace_enabled
class _VLMAdapter(TTNNModule):
    @run_on_devices(DeviceArch.P150)
    def forward(self, prefix):
        return self._bk.forward_vlm(prefix, attention_mask=None, use_cache=True)[0]


@trace_enabled
class _DenoiseAdapter(TTNNModule):
    @run_on_devices(DeviceArch.P150)
    def forward(self, x_t):
        # mods/mask/prefix_len are constant for a single repeated step -> closed over.
        return self._m._denoise_forward(
            x_t, None, self._pl, self._mask, self._bmods, self._fmod
        )


def _time_stage(dev, fn, label, warmup=2, iters=10):
    """Call fn() (warmup+1) times to prime (TRACED: warmup->capture->1st replay;
    NORMAL: kernel compile), then time `iters` further calls. Returns ms/call.
    fn returns a ttnn.Tensor (deallocated each iter)."""

    def _one():
        r = fn()
        ttnn.synchronize_device(dev)
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)

    for _ in range(warmup + 1):
        _one()
    t0 = time.perf_counter()
    for _ in range(iters):
        _one()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    print(f"  [{RUN_MODE}] {label:<14} {ms:8.3f} ms/call (iters={iters})", flush=True)
    return ms


def main():
    from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
    from tt_symbiote.models.pi05.modeling_pi05 import TTNNPi05Model, load_reference_pi05_model
    from tt_symbiote.utils.device_management import set_device

    torch.manual_seed(0)
    cfg = Pi0_5ModelConfig()
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=134_217_728)
    ok = False
    try:
        ref_model = load_reference_pi05_model(CKPT)
        tt = TTNNPi05Model.from_torch(ref_model, cfg)
        set_device(tt, dev)

        lang_len = 32
        image_t = torch.randn(1, 3, 224, 224)
        image = ttnn.from_torch(image_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ah, ad = cfg.action_horizon, cfg.action_dim
        ahp = tt._tile_pad(ah)
        n = cfg.num_denoising_steps

        print(f"\n=== pi0.5 chunk timing (mode={RUN_MODE}) ===", flush=True)

        # --- stage: tower (embed_image), one captured graph ---
        tower = _wrap(_TowerAdapter, dev, _bk=tt.backbone)
        t_tower = _time_stage(dev, lambda: tower(image), "tower", iters=5)

        # Prefix for the VLM-prefill timing: a shape-correct input
        # ([1, num_patches+lang, vlm_width]) is all the replay-time measurement
        # needs. Built directly (not from a trace-region tower output, which
        # eager concat cannot consume) and decoupled from the tower stage.
        num_patches = cfg.siglip_config.num_patches
        prefix_len = num_patches + lang_len
        prefix = ttnn.from_torch(
            torch.randn(1, prefix_len, cfg.vlm_config.width),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )

        # --- stage: forward_vlm via the trace-safe VLM prefix store ---
        tt.backbone.init_vlm_static_kv(prefix_len)
        vlm = _wrap(_VLMAdapter, dev, _bk=tt.backbone)
        t_vlm = _time_stage(dev, lambda: vlm(prefix), "forward_vlm", iters=10)

        # Fill expert cross-attn KV from the VLM stores; set up one denoise step.
        tt.backbone.init_expert_static_kv_from_vlm(prefix_len, ahp)
        suffix_mask = tt._suffix_phantom_mask(prefix_len, ah, ahp)
        cond = tt.suffix_embedding.embed_adarms_cond(tt._ts(1.0))
        block_mods, final_mod = tt.backbone.precompute_step_mods(cond)
        x_t = tt._noise_to_device(torch.randn(1, ah, ad), ahp)

        # --- stage: one denoise velocity step ---
        denoise = _wrap(
            _DenoiseAdapter, dev, _m=tt, _pl=prefix_len, _mask=suffix_mask,
            _bmods=block_mods, _fmod=final_mod,
        )
        t_step = _time_stage(dev, lambda: denoise(x_t), "denoise_step", iters=10)

        t_denoise = t_step * n
        t_prefix = t_tower + t_vlm  # + tiny eager lang-embed/concat glue (not timed)
        t_chunk = t_prefix + t_denoise
        print(f"\n  --- summary (mode={RUN_MODE}) ---")
        print(f"  tower               {t_tower:8.3f} ms")
        print(f"  forward_vlm         {t_vlm:8.3f} ms")
        print(f"  prefix (tower+vlm)  {t_prefix:8.3f} ms")
        print(f"  denoise_step        {t_step:8.3f} ms  x{n} = {t_denoise:8.3f} ms")
        print(f"  FULL CHUNK          {t_chunk:8.3f} ms")
        sys.stdout.flush()
        ok = True
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
    # Bypass device teardown: with captured traces still resident, the
    # FDMeshCommandQueue destructor can wedge in pthread_cond_wait. Results are
    # already flushed; the device fd is reclaimed on process exit and a fresh
    # open works cleanly afterwards (verified).
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
