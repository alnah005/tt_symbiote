# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for the pi0.5 port.

Ported from the tt-metal reference ``models/experimental/pi0_5/tt/ttnn_common.py``
(branch ``tt/pi0.5_bh``). Covers:

* SDPA compute-kernel config (HiFi2 + fp32_dest + packer_l1 defaults, env-tunable).
* Meta-format RoPE cos/sin precompute (``[1, 1, max_seq, head_dim]``) for
  ``ttnn.experimental.rotary_embedding``.
* Sinusoidal flow-matching timestep embedding (host-side; uploaded bf16).

The denoise-loop fp32 toggle and SDPA knobs are exposed as env vars so the
op-sweep / config-optimize stages can A/B them without code edits.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch
import ttnn

__all__ = [
    "get_sdpa_math_fidelity",
    "get_sdpa_compute_kernel_config",
    "denoise_loop_fp32",
    "sdpa_prefill_chunk_sizes",
    "precompute_freqs_cis_meta",
    "create_sinusoidal_pos_embedding",
]


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_sdpa_math_fidelity() -> "ttnn.MathFidelity":
    """SDPA math fidelity. Default HiFi2 (measured-best on Blackhole with fp32 dest)."""
    return ttnn.MathFidelity.HiFi4 if _env_int("PI0_SDPA_HIFI", 2) >= 4 else ttnn.MathFidelity.HiFi2


def get_sdpa_compute_kernel_config() -> "ttnn.WormholeComputeKernelConfig":
    """SDPA compute-kernel config matching the reference env knobs."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=get_sdpa_math_fidelity(),
        math_approx_mode=False,
        fp32_dest_acc_en=_env_bool("PI0_SDPA_FP32_DEST", True),
        packer_l1_acc=_env_bool("PI0_SDPA_PACKER_L1", True),
    )


def denoise_loop_fp32() -> bool:
    """Whether to stage the flow-matching Euler integration in fp32 (PI0_DENOISE_FP32=1).

    The bf16 accumulator drifts ~bf16_eps*||x_t|| per step; fp32 keeps the
    accumulator clean (biggest single accuracy lever vs bf16 drift, ~+30 ms).
    """
    return _env_bool("PI0_DENOISE_FP32", False)


def sdpa_prefill_chunk_sizes(seq_len_q: int, seq_len_kv: int, *, tile: int = 32) -> Tuple[int, int]:
    """q_chunk / k_chunk sizes for ttnn SDPA, mirroring the tt_transformers baseline."""
    longest = max(seq_len_q, seq_len_kv)
    if longest >= 2048:
        base_q, base_k = 256, 256
    elif longest >= 512:
        base_q, base_k = 64, 128
    else:
        base_q, base_k = 64, 64
    q_aligned = ((seq_len_q + tile - 1) // tile) * tile if seq_len_q > 0 else tile
    k_aligned = ((seq_len_kv + tile - 1) // tile) * tile if seq_len_kv > 0 else tile
    return max(min(base_q, q_aligned), tile), max(min(base_k, k_aligned), tile)


def precompute_freqs_cis_meta(
    head_dim: int,
    max_seq_len: int,
    device: ttnn.Device,
    base: float = 10000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Precompute RoPE cos/sin in meta format ``[1, 1, max_seq_len, head_dim]``.

    The halves are duplicated (``cat([h, h], -1)``) so ``ttnn.experimental.
    rotary_embedding`` applies the split-half rotation matching the torch
    reference ``apply_rotary_emb``. Built on host (free) and uploaded bf16.
    """
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    outer = torch.outer(t, freqs)  # [max_seq, head_dim//2]
    cos = torch.cat([torch.cos(outer), torch.cos(outer)], dim=-1)  # [max_seq, head_dim]
    sin = torch.cat([torch.sin(outer), torch.sin(outer)], dim=-1)
    cos = cos.reshape(1, 1, max_seq_len, head_dim).contiguous()
    sin = sin.reshape(1, 1, max_seq_len, head_dim).contiguous()
    cos_tt = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return cos_tt, sin_tt


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    device: ttnn.Device,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> ttnn.Tensor:
    """Sinusoidal flow-matching timestep embedding ``[batch, dimension]``.

    Computed on host (timestep count is tiny) and uploaded bf16, matching the
    reference ``create_sinusoidal_pos_embedding`` math (linspace period schedule,
    ``cat([sin, cos])``).
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be even")
    half = dimension // 2
    fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period) * 2.0 * math.pi  # [half]
    time = time.reshape(-1, 1).to(torch.float32)  # [batch, 1]
    sin_input = time * scaling.reshape(1, half)  # [batch, half]
    emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)  # [batch, dimension]
    return ttnn.from_torch(emb.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
