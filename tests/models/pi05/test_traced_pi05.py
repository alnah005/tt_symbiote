# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Traced-execution validation for pi0.5 (production perf path).

Captures the per-step flow-matching denoise graph -- ``forward_expert`` (18-layer
Gemma-300M adaRMS action expert, cross-attending the cached prefix KV) + the
action output projection -- as a TTNN trace, replays it, and checks:

  1. NUMERIC FIDELITY: replayed velocity matches the eager velocity (PCC ~1.0),
     proving trace capture/replay preserves the result.
  2. PERF DIRECTION: traced per-step replay latency vs eager per-step, to confirm
     trace amortizes host dispatch (the lever that takes the eager ~207ms/chunk
     toward the reference's ~65ms/chunk traced baseline).

Requires the real ``pi05_base`` checkpoint (skips otherwise) and a device opened
with a ``trace_region_size`` (the ``dev`` fixture sets 128 MiB).
"""

from __future__ import annotations

import math
import time

import torch

import ttnn
from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
from tt_symbiote.utils.device_management import set_device

from .pi05_helpers import assert_pcc, compute_pcc
from .pi05_helpers import SEED, require_checkpoint, require_reference

_L1 = ttnn.L1_MEMORY_CONFIG


def test_traced_denoise_step(dev):
    require_reference()
    ckpt = require_checkpoint()
    from tt_symbiote.models.pi05.modeling_pi05 import TTNNPi05Model, load_reference_pi05_model

    torch.manual_seed(SEED)
    cfg = Pi0_5ModelConfig()
    ref_model = load_reference_pi05_model(ckpt)
    tt = TTNNPi05Model.from_torch(ref_model, cfg)
    set_device(tt, dev)

    # --- Build prefix + VLM KV cache (eager; persistent inputs for the trace) ---
    lang_len = 32
    images = [ttnn.from_torch(torch.randn(1, 3, 224, 224), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
    lang_tokens = ttnn.from_torch(
        torch.randint(0, 257152, (1, lang_len)).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
    )
    import math

    img_embeds = tt.backbone.embed_image(images[0])
    lang_embeds = ttnn.multiply(tt.backbone.embed_language_tokens(lang_tokens), math.sqrt(cfg.vlm_config.width))
    prefix = ttnn.concat([img_embeds, lang_embeds], dim=1)
    prefix_len = prefix.shape[1]
    _, vlm_cache = tt.backbone.forward_vlm(prefix, attention_mask=None, use_cache=True)

    # --- Precompute fixed per-step inputs (persistent) ---
    ah, ad = cfg.action_horizon, cfg.action_dim
    ahp = tt._tile_pad(ah)
    x0 = tt._noise_to_device(torch.randn(1, ah, ad), ahp)
    suffix_embs = tt.suffix_embedding.embed_actions(x0)
    adarms = tt.suffix_embedding.embed_adarms_cond(tt._ts(0.9))
    mask = tt._suffix_phantom_mask(prefix_len, ah, ahp)
    # Precompute adaRMS modulations (DRAM) so the captured per-step graph carries
    # no mod-matmuls/slices -- the reference TIER A perf optimization.
    block_mods, final_mod = tt.backbone.precompute_step_mods(adarms)

    def step():
        eo = tt.backbone.forward_expert(
            suffix_embs,
            past_key_values=vlm_cache,
            attention_mask=mask,
            position_offset=prefix_len,
            precomputed_block_mods=block_mods,
            precomputed_final_mod=final_mod,
        )
        return tt.suffix_embedding.project_output(eo)

    # --- Eager golden + warm-up (kernel compile) ---
    golden = ttnn.to_torch(step())
    _ = step()
    ttnn.synchronize_device(dev)
    n_eager = 10
    t0 = time.perf_counter()
    for _ in range(n_eager):
        _ = step()
    ttnn.synchronize_device(dev)
    eager_ms = (time.perf_counter() - t0) / n_eager * 1000.0

    # --- Capture trace ---
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    traced_out = step()
    ttnn.end_trace_capture(dev, tid, cq_id=0)

    # --- Replay + time ---
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(dev)
    n_rep = 20
    t0 = time.perf_counter()
    for _ in range(n_rep):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    traced_ms = (time.perf_counter() - t0) / n_rep * 1000.0

    traced = ttnn.to_torch(traced_out)
    ttnn.release_trace(dev, tid)

    print(
        f"\npi0.5 denoise-step: eager={eager_ms:.2f}ms  traced={traced_ms:.2f}ms  "
        f"speedup={eager_ms / traced_ms:.2f}x  (prefix_len={prefix_len})"
    )
    # Trace replay must reproduce the eager result.
    assert_pcc(traced, golden, threshold=0.99, msg="traced-vs-eager denoise step")
    # Trace should not be slower than eager (it amortizes host dispatch).
    assert traced_ms <= eager_ms * 1.1, f"traced {traced_ms:.2f}ms not <= eager {eager_ms:.2f}ms"


def _reference_sample_actions(ref_model, images_t, img_masks_t, lang_tokens_t, lang_masks_t, noise):
    """Torch reference E2E with injected noise (TTNN and torch share it)."""
    from models.experimental.pi0_5.reference.torch_pi0_5_model import _build_prefix_mask_and_pos

    prefix_embs, ppm, pam = ref_model.embed_prefix(images_t, img_masks_t, lang_tokens_t, lang_masks_t)
    pos, mask4d = _build_prefix_mask_and_pos(ppm, pam, prefix_embs.dtype)
    _, vlm_cache = ref_model.backbone.forward_vlm(prefix_embs, attention_mask=mask4d, position_ids=pos, use_cache=True)
    x = noise.clone()
    n = ref_model.config.num_denoising_steps
    for i in range(n):
        t = torch.tensor([1.0 - i / n])
        v = ref_model._denoise_forward(x, t, vlm_cache, prefix_pad_masks=ppm)
        x = x + (-1.0 / n) * v
    return x


def test_traced_static_kv_denoise_e2e(dev):
    """TRACED E2E: capture the STATIC-KV denoise step (no concat) and replay the
    full 10-step Euler loop, validating finite/correct-shape actions and E2E PCC
    vs the torch reference.

    Proves the static KV buffer + in-place ttnn.fill_cache write is trace-safe:
    the per-step graph (forward_expert over the pre-allocated static buffers +
    action output projection) warms up, captures (begin/end_trace_capture), and
    replays (execute_trace) with no "Writes not supported during trace capture".
    """
    require_reference()
    ckpt = require_checkpoint()
    import math

    from tt_symbiote.models.pi05.modeling_pi05 import TTNNPi05Model, load_reference_pi05_model

    torch.manual_seed(SEED)
    cfg = Pi0_5ModelConfig()
    ref_model = load_reference_pi05_model(ckpt)
    tt = TTNNPi05Model.from_torch(ref_model, cfg)
    set_device(tt, dev)

    lang_len = 32
    images_t = [torch.randn(1, 3, 224, 224)]
    img_masks_t = [torch.ones(1, dtype=torch.bool)]
    lang_tokens_t = torch.randint(0, 257152, (1, lang_len))
    lang_masks_t = torch.ones(1, lang_len, dtype=torch.bool)
    noise = torch.randn(1, cfg.action_horizon, cfg.action_dim)

    actions_ref = _reference_sample_actions(ref_model, images_t, img_masks_t, lang_tokens_t, lang_masks_t, noise)

    # --- Prefix prefill + static-KV init (eager, outside trace) ---
    images = [ttnn.from_torch(images_t[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
    lang_tokens = ttnn.from_torch(lang_tokens_t.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    img_embeds = tt.backbone.embed_image(images[0])
    lang_embeds = ttnn.multiply(tt.backbone.embed_language_tokens(lang_tokens), math.sqrt(cfg.vlm_config.width))
    prefix = ttnn.concat([img_embeds, lang_embeds], dim=1)
    prefix_len = prefix.shape[1]
    _, vlm_cache = tt.backbone.forward_vlm(prefix, attention_mask=None, use_cache=True)

    ah, ad = cfg.action_horizon, cfg.action_dim
    ahp = tt._tile_pad(ah)
    # Allocate + prefill the per-layer static KV buffers (the trace-safe path).
    tt.backbone.init_expert_static_kv(vlm_cache, prefix_len, ahp)

    mask = tt._suffix_phantom_mask(prefix_len, ah, ahp)
    n = cfg.num_denoising_steps
    step_mods = tt._ensure_step_mods()

    # The trace captures one denoise step over the STATIC KV buffers. Per-step
    # adaRMS mods differ, so we capture ONE trace per step (10 traces, each a
    # distinct mod set) and replay the whole Euler loop -- a faithful E2E that
    # exercises capture (begin/end_trace_capture) + replay (execute_trace) of the
    # static-KV expert path, and reproduces the reference integration exactly.
    # x_t lives in a persistent device buffer the captured graph reads each replay.
    x_t = tt._noise_to_device(noise, ahp)

    def build_step_graph(block_mods, final_mod):
        suffix_embs = tt.suffix_embedding.embed_actions(x_t)
        eo = tt.backbone.forward_expert(
            suffix_embs,
            past_key_values=None,  # static buffers own the prefix; suffix written in-place
            attention_mask=mask,
            position_offset=prefix_len,
            precomputed_block_mods=block_mods,
            precomputed_final_mod=final_mod,
        )
        ttnn.deallocate(suffix_embs)
        return tt.suffix_embedding.project_output(eo)

    # --- Warm-up every step's graph (kernel compile), then capture a trace each ---
    import time

    traces = []
    for i in range(n):
        bm, fm = step_mods[i]
        _ = build_step_graph(bm, fm)  # warm-up
    ttnn.synchronize_device(dev)
    for i in range(n):
        bm, fm = step_mods[i]
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        v_out = build_step_graph(bm, fm)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        traces.append((tid, v_out))
    ttnn.synchronize_device(dev)
    print(f"\n[static-kv-traced] captured {len(traces)} step traces (prefix_len={prefix_len} ahp={ahp})")

    # --- Replay the Euler loop: each step reads current x_t, replays, accumulates ---
    t0 = time.perf_counter()
    for i in range(n):
        tid, v_out = traces[i]
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        v_dt = ttnn.multiply(v_out, -1.0 / n, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_next = ttnn.add(x_t, v_dt, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(v_dt)
        ttnn.copy(x_next, x_t)  # update persistent buffer in-place for next replay
        ttnn.deallocate(x_next)
    ttnn.synchronize_device(dev)
    per_step_ms = (time.perf_counter() - t0) / n * 1000.0

    actions = ttnn.to_torch(ttnn.slice(x_t, [0, 0, 0], [1, ah, ad]))
    for tid, _ in traces:
        ttnn.release_trace(dev, tid)

    print(f"[static-kv-traced] replayed {n} steps, per-step={per_step_ms:.2f}ms")

    assert actions.shape == (1, ah, ad), actions.shape
    assert torch.isfinite(actions).all(), "traced actions contain NaN/inf"
    assert actions.float().std() > 1e-4, "traced actions degenerate (constant)"

    from .pi05_helpers import compute_pcc

    pcc = compute_pcc(actions, actions_ref)
    print(f"[static-kv-traced] E2E action PCC (TRACED-replay vs torch): {pcc:.4f}")
    assert pcc >= 0.90, f"TRACED E2E action PCC {pcc:.4f} < 0.90"

