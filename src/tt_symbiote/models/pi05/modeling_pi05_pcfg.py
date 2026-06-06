# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure-ttnn matmul / sharded-norm program-config builders for pi0.5.

VENDORED verbatim from the tt-metal pi0_5 reference
``models/experimental/pi0_5/tt/ttnn_gemma.py`` (branch tt/pi0.5_bh, commit
b0703a56465989da179480c93a8992ec519e1cde) so the tt_symbiote pi0.5 forward path is SELF-CONTAINED and does not
import the external reference tree at runtime. These construct ttnn program
configs only -- no torch, no model state. Re-vendor + bump the commit below if the
reference builders change.
"""

import math  # noqa: F401  (used by the vendored builders)
from typing import Optional  # noqa: F401  (used by the vendored builders)

import ttnn

REFERENCE_TT_METAL_COMMIT = "b0703a56465989da179480c93a8992ec519e1cde"

# Memoization caches the vendored builders use (keyed by shape/grid).
_pcfg_cache: dict = {}
_sharded_norm_cache: dict = {}


def build_matmul_pcfg(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid_x: int,
    grid_y: int,
    *,
    in0_block_w: Optional[int] = None,
    activation=None,
    dst_budget: int = 8,
):
    """Build a sharded-matmul program config.

    For small M (suffix-side action expert: M=2 tiles), we'd waste rows of the
    2D grid (only m_tiles of grid_y rows get work → 12×2 = 24 of 120 cores).
    For these cases we return a 1D **width-sharded** config instead
    (MatmulMultiCoreReuseMultiCast1DProgramConfig with mcast_in0=True), which
    multicasts activations and spreads N across all 120 cores. That brings the
    expert MLP/QKV/o_proj from ~22-24 cores to 60-120.

    For larger M (VLM prefix: M=9 tiles, SigLIP: M=8 tiles) the 2D config
    already uses 96-108 cores, so we keep it.

    Tuned for Blackhole Galaxy (compute grid 12x10): use grid=(12,10) and
    in0_block_w=4 for large MLP matmuls (CBs fit under the ~200 KB L1 headroom
    left by trace-persistent KV cache buffers). For smaller attention matmuls,
    pass in0_block_w=8.

    Returns None if shapes don't admit a clean config.
    """
    if m_tiles == 0 or k_tiles == 0 or n_tiles == 0:
        return None
    total_cores = grid_x * grid_y

    # --- 1D width-shard path: small M, big N -----------------------------
    # When m_tiles is much smaller than grid_y, the 2D grid wastes rows.
    # Switch to 1D width-shard so all 120 cores work on N slices.
    if m_tiles * 4 <= grid_y and n_tiles >= total_cores // 4:
        # Pick num_cores: largest divisor of n_tiles ≤ total_cores, and not
        # smaller than half the grid (otherwise stick with 2D).
        num_cores = min(total_cores, n_tiles)
        while num_cores > total_cores // 2 and n_tiles % num_cores != 0:
            num_cores -= 1
        if n_tiles % num_cores != 0:
            # No clean divisor available — fall back to ceil division on full grid
            num_cores = total_cores
            per_core_N_1d = (n_tiles + num_cores - 1) // num_cores
        else:
            per_core_N_1d = n_tiles // num_cores

        # Choose in0_block_w for 1D (full M per core → larger CBs needed).
        # 1D width-shard with small M (≤2 tiles) and small per_core_N (often 1-2)
        # has tiny CBs — we can use a much larger block_w than the 2D default of
        # 4-8 to reduce K-loop iterations. Cap at 16 to keep L1 happy when
        # per_core_N is bigger (e.g. n_tiles=128, num_cores=64, per_core_N=2).
        if in0_block_w is None:
            in0_bw = 16
        else:
            in0_bw = in0_block_w
        # Constrain by per_core_N for L1 safety: CB ~ in0_bw * per_core_N tiles.
        while in0_bw > 1 and in0_bw * per_core_N_1d > 32:
            in0_bw //= 2
        while k_tiles % in0_bw != 0 and in0_bw > 1:
            in0_bw //= 2
        if in0_bw < 2:
            in0_bw = 1

        # DST budget: out_subblock_w * out_subblock_h ≤ dst_budget.
        out_subblock_w_1d = min(per_core_N_1d, dst_budget)
        while out_subblock_w_1d > 1 and per_core_N_1d % out_subblock_w_1d != 0:
            out_subblock_w_1d -= 1
        out_subblock_h_1d = max(1, dst_budget // out_subblock_w_1d)
        out_subblock_h_1d = min(m_tiles, out_subblock_h_1d)
        while out_subblock_h_1d > 1 and m_tiles % out_subblock_h_1d != 0:
            out_subblock_h_1d -= 1

        # Grid: num_cores arranged as (grid_x, ceil(num_cores/grid_x)).
        cfg_gx = min(grid_x, num_cores)
        cfg_gy = (num_cores + cfg_gx - 1) // cfg_gx

        key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), dst_budget, "1d")
        if key in _pcfg_cache:
            return _pcfg_cache[key]
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cfg_gx, cfg_gy),
            in0_block_w=in0_bw,
            out_subblock_h=out_subblock_h_1d,
            out_subblock_w=out_subblock_w_1d,
            per_core_M=m_tiles,
            per_core_N=per_core_N_1d,
            fuse_batch=True,
            fused_activation=activation,
            mcast_in0=True,
        )
        _pcfg_cache[key] = cfg
        return cfg

    # --- 2D block-shard path (default, large M) --------------------------
    per_core_M = (m_tiles + grid_y - 1) // grid_y
    per_core_N = (n_tiles + grid_x - 1) // grid_x
    if per_core_M == 0 or per_core_N == 0:
        return None

    # in0_block_w must divide K_tiles. SigLIP MLP has K=4304 padded to 4320
    # (135 tiles) which doesn't divide cleanly into 4 or 2 — return None so
    # caller falls back to the default core_grid path.
    if in0_block_w is None:
        # Adaptive choice: per_core_N drives CB size. block_w=8 is fastest
        # but overflows L1 when per_core_N is large (CB ≈ block_w * per_core_N
        # tiles, doubled for double-buffer). With pi0's trace KV pinning
        # ~1.3 MB of L1, we have ~150 KB CB headroom per core.
        # block_w=8 cap: per_core_N * 8 * 1024 * 2 ≤ 150 KB → per_core_N ≤ 9
        if per_core_N <= 12:
            in0_block_w = 8
        else:
            in0_block_w = 4
    while k_tiles % in0_block_w != 0 and in0_block_w > 1:
        in0_block_w //= 2
    if in0_block_w == 1 and k_tiles > 32:
        # Tiny block_w on a large K is unlikely to win — bail out.
        return None

    key = (m_tiles, k_tiles, n_tiles, grid_x, grid_y, str(activation), dst_budget)
    if key in _pcfg_cache:
        return _pcfg_cache[key]

    # out_subblock_w * out_subblock_h <= dst_budget (DST register tile budget).
    # fp32_dest_acc_en=True halves the budget to 4. bf16 dest accum allows 8.
    out_subblock_w = min(per_core_N, dst_budget)
    while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    out_subblock_h_budget = max(1, dst_budget // out_subblock_w)
    out_subblock_h = min(per_core_M, out_subblock_h_budget)
    while out_subblock_h > 1 and per_core_M % out_subblock_h != 0:
        out_subblock_h -= 1

    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=activation,
    )
    _pcfg_cache[key] = cfg
    return cfg


def build_sharded_norm_pcfg(
    m_tiles: int,
    hidden_tiles: int,
    *,
    max_grid_x: int = 8,
    max_grid_y: int = 8,
) -> Optional[tuple]:
    """Build (program_config, sharded_memory_config_factory, grid) for sharded
    RMS/LayerNorm using the LayerNormShardedMultiCoreProgramConfig pattern.

    Returns None if no clean grid divides both dims.

    For the pi0.5 expert (M_tiles=2, hidden_tiles=32) this picks (8, 2) →
    16 cores instead of the default 2-core interleaved path. For VLM
    (M_tiles=16, hidden_tiles=64) this picks (8, 8) → 64 cores.
    """
    key = (m_tiles, hidden_tiles, max_grid_x, max_grid_y)
    if key in _sharded_norm_cache:
        return _sharded_norm_cache[key]

    # grid_y must divide M_tiles; grid_x must divide hidden_tiles.
    cand_y = [y for y in range(min(max_grid_y, m_tiles), 0, -1) if m_tiles % y == 0]
    cand_x = [x for x in range(min(max_grid_x, hidden_tiles), 0, -1) if hidden_tiles % x == 0]
    if not cand_y or not cand_x:
        _sharded_norm_cache[key] = None
        return None

    best = None
    best_cores = 0
    for gy in cand_y:
        for gx in cand_x:
            cores = gx * gy
            # Prefer high core count; tie-break on grid_x (sharded LN parallelizes
            # along W primarily — wider grid helps small-M shapes).
            if cores > best_cores or (cores == best_cores and gx > best[0]):
                best = (gx, gy)
                best_cores = cores
    if best is None:
        _sharded_norm_cache[key] = None
        return None

    gx, gy = best
    block_h = m_tiles // gy
    block_w = hidden_tiles // gx
    subblock_w = block_w  # full-width per-core subblock (typical for sharded LN)

    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )

    def make_memcfg(b: int, m_logical: int, m_physical: int, hidden: int):
        # Block-sharded across the chosen grid.
        return ttnn.create_sharded_memory_config(
            (b, 1, m_physical, hidden),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    grid = ttnn.CoreGrid(y=gy, x=gx)
    result = (pc, make_memcfg, grid)
    _sharded_norm_cache[key] = result
    return result


_RMS_NORM_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
