# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Block-sharded (BS) Tier-1 optimization layer for pi0.5.

Tracy-validated (single P150, real DEVICE KERNEL DURATION) per-op wins over the
shipped core_grid/interleaved path:

  * sharded RMSNorm (build_sharded_norm_pcfg)          -- 2.0-2.9x (expert + VLM)
  * attention QKV/o matmul (build_matmul_pcfg)         -- 1.24-1.55x
  * SigLIP MLP fc1/fc2 matmul (build_matmul_pcfg)      -- ~1.1-1.3x

This module is a thin wrapper over the VENDORED pure-config builders
(``build_matmul_pcfg`` / ``build_sharded_norm_pcfg`` in ``modeling_pi05_pcfg`` --
they construct ttnn program configs only, no torch, no model state). The builders
are vendored into tt_symbiote, so the forward path is self-contained.

Behavior:
  * These optimizations are the MAIN path -- always on (no env switch).
  * If no clean grid divides a given shape, the helper returns ``None`` and the
    caller falls back to its existing core_grid / interleaved path -- so the
    modeling stays correct regardless.
"""

from __future__ import annotations

import ttnn

# Pure-ttnn program-config builders, VENDORED into tt_symbiote (no external pi0_5
# reference-tree import) -- the forward path is self-contained.
from tt_symbiote.models.pi05.modeling_pi05_common import sdpa_prefill_chunk_sizes
from tt_symbiote.models.pi05.modeling_pi05_pcfg import (
    _RMS_NORM_COMPUTE_CONFIG,
    build_matmul_pcfg,
    build_sharded_norm_pcfg,
)

TT_METAL_COMMIT = "b2af0cd67b4e92dafeb2d0254e1c5b43c3ec5a25"

# The block-sharded optimizations are the MAIN (only) path -- always on. The
# pure-ttnn builders are vendored (modeling_pi05_pcfg); helpers still no-op-fall-back
# (return None -> caller uses core_grid/interleaved) when no clean grid divides a shape.
_ENABLED = True


def bs_enabled() -> bool:
    """True iff BS optimizations are active (always on; master switch)."""
    return _ENABLED


def _rms_compute():
    return _RMS_NORM_COMPUTE_CONFIG


def matmul_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, **kw):
    """2D-block / 1D-width-shard matmul program config (or None to fall back)."""
    if not bs_enabled():
        return None
    try:
        return build_matmul_pcfg(m_tiles, k_tiles, n_tiles, grid_x, grid_y, **kw)
    except Exception:
        return None


def sharded_norm_pcfg(m_tiles, hidden_tiles, *, max_grid_x=8, max_grid_y=8):
    """(program_config, memcfg_factory, grid) for sharded RMS/LayerNorm, or None."""
    if not bs_enabled():
        return None
    try:
        return build_sharded_norm_pcfg(m_tiles, hidden_tiles, max_grid_x=max_grid_x, max_grid_y=max_grid_y)
    except Exception:
        return None


def sharded_rms_norm(x, weight, eps, m_padded, hidden, *, batch=1, bias=None):
    """Sharded RMSNorm with a pre-offset ``weight`` (Gemma ``w+1``), returning an
    INTERLEAVED-L1 result. Builds the sharded grid keyed on the actual M; falls
    back to the plain interleaved ``ttnn.rms_norm`` when BS is off or no clean grid
    divides the shape. ``weight`` is passed explicitly (trace-safe: no host->device
    default-gamma write during trace capture).

    ``bias`` (optional) is added post-norm INSIDE the kernel -- used by the adaRMS
    fused modulation path (weight=1+scale, bias=shift) to fold the separate
    multiply+add into the norm op (matches upstream ``_modulated_rms_norm``)."""
    m_tiles = m_padded // 32
    # Cap grid_y at 8 (reference GemmaBlock convention): an uncapped max(1, m_tiles)
    # explodes the core grid at large M (e.g. 3-camera VLM seq=800, m_tiles=25 ->
    # gy=25 -> 8x25=200 cores > device -> bank_manager TT_FATAL). min(8, ...) keeps
    # cores <= 64 and divides cleanly for expert(2), VLM-288(9->gy3), VLM-800(25->gy5).
    cfg = sharded_norm_pcfg(m_tiles, hidden // 32, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles)))
    if cfg is None:
        return ttnn.rms_norm(x, weight=weight, bias=bias, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
    pc, memcfg_factory, _grid = cfg
    memcfg = memcfg_factory(batch, m_padded, m_padded, hidden)
    x_sh = ttnn.to_memory_config(x, memcfg)
    normed = ttnn.rms_norm(
        x_sh,
        weight=weight,
        bias=bias,
        epsilon=eps,
        program_config=pc,
        compute_kernel_config=_rms_compute(),
        memory_config=memcfg,
    )
    ttnn.deallocate(x_sh)
    out = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(normed)
    return out


def sdpa_program_config(seq_q, seq_kv, grid_x, grid_y, *, q_chunk=None, k_chunk=None):
    """SDPAProgramConfig with the tuned (divisor-aware) q/k chunk sizes (vendored
    ``modeling_pi05_common.sdpa_prefill_chunk_sizes``) on (grid_x, grid_y), or None
    to fall back to the default SDPA. exp_approx_mode is False (the measured-best /
    main-path value). q_chunk/k_chunk override the bands (per-shape sweeps); the
    caller may clamp the grid (e.g. the small-q expert SDPA over-parallelizes on the
    full 110-core grid)."""
    if not _ENABLED:
        return None
    try:
        qc, kc = sdpa_prefill_chunk_sizes(seq_q, seq_kv)
        if q_chunk is not None:
            qc = q_chunk
        if k_chunk is not None:
            kc = k_chunk
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
        )
    except Exception:
        return None


def sharded_layer_norm(x, weight, bias, eps, m_padded, hidden, *, batch=1):
    """Sharded LayerNorm (affine: weight + bias), INTERLEAVED-L1 result. Same grid
    logic as sharded_rms_norm (max_grid_y capped at 8 -> on-dispatch grid for the
    SigLIP 256-token / 1152-hidden block, which the reference's max_grid_x=12 path
    mis-sized). Falls back to plain interleaved ttnn.layer_norm when BS is off or no
    clean grid divides the shape."""
    m_tiles = m_padded // 32
    cfg = sharded_norm_pcfg(m_tiles, hidden // 32, max_grid_x=8, max_grid_y=min(8, max(1, m_tiles)))
    if cfg is None:
        return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
    pc, memcfg_factory, _grid = cfg
    memcfg = memcfg_factory(batch, m_padded, m_padded, hidden)
    x_sh = ttnn.to_memory_config(x, memcfg)
    normed = ttnn.layer_norm(
        x_sh,
        weight=weight,
        bias=bias,
        epsilon=eps,
        program_config=pc,
        compute_kernel_config=_rms_compute(),
        memory_config=memcfg,
    )
    ttnn.deallocate(x_sh)
    out = ttnn.sharded_to_interleaved(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(normed)
    return out
