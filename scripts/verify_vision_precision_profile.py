# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-hardware faithfulness check for the dots.ocr vision-precision toggle.

Verifies that ``DOTS_OCR_VISION_PRECISION`` resolves every knob to the documented
value for each tier, and in particular that the ``baseline`` tier reproduces git
HEAD (a418638) byte-faithfully:

    * LoFi math fidelity everywhere (incl. MLP)
    * fp32_dest_acc_en=False on matmuls, approx softmax/SILU on
    * BFP4 at the four hot sites (MLP up/down, SDPA V, merger fc1), BFP8 elsewhere

and that ``hifi2`` reproduces the prior working-tree "vision-bumped" state. This is
the gate that must pass BEFORE any hardware run so a broken refactor cannot silently
invalidate the A/B/C comparison.

Run:  python scripts/verify_vision_precision_profile.py
Exit code 0 = all tiers faithful; 1 = a knob drifted.
"""

from __future__ import annotations

import os
import subprocess
import sys

import ttnn

from tt_symbiote.models.dots_ocr import dots_ocr_vision as V

LoFi, HiFi2, HiFi4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
BFP4, BFP8, BF16 = ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16

# The specification — every field, per tier. baseline == git HEAD a418638.
EXPECTED = {
    "baseline": dict(
        matmul_fidelity=LoFi, sdpa_fidelity=LoFi, norm_fidelity=LoFi, mlp_fidelity=LoFi,
        matmul_fp32_acc=False, sdpa_exp_approx=True, mlp_fast_silu=True,
        dt_mlp_up=BFP4, dt_mlp_down=BFP4, dt_sdpa_v=BFP4, dt_merger_fc1=BFP4,
        dt_activations=BFP8, dt_weights=BFP8, dt_rope_tables=BFP8,
        prefer_dram_intermediates=False, use_tuned_program_configs=True,
    ),
    "hifi2": dict(
        matmul_fidelity=HiFi2, sdpa_fidelity=HiFi2, norm_fidelity=HiFi2, mlp_fidelity=LoFi,
        matmul_fp32_acc=False, sdpa_exp_approx=True, mlp_fast_silu=True,
        dt_mlp_up=BFP8, dt_mlp_down=BFP8, dt_sdpa_v=BFP8, dt_merger_fc1=BFP8,
        dt_activations=BFP8, dt_weights=BFP8, dt_rope_tables=BFP8,
        prefer_dram_intermediates=False, use_tuned_program_configs=True,
    ),
    "hifi4": dict(
        # Auto-config/DRAM path: the tuned L1 fast path is budgeted for LoFi/HiFi2 and
        # overflows at HiFi4, so hifi4 uses auto configs (which also make fp32 acc safe).
        matmul_fidelity=HiFi4, sdpa_fidelity=HiFi4, norm_fidelity=HiFi4, mlp_fidelity=HiFi4,
        matmul_fp32_acc=True, sdpa_exp_approx=False, mlp_fast_silu=False,
        dt_mlp_up=BFP8, dt_mlp_down=BFP8, dt_sdpa_v=BFP8, dt_merger_fc1=BFP8,
        dt_activations=BFP8, dt_weights=BFP8, dt_rope_tables=BFP8,
        prefer_dram_intermediates=True, use_tuned_program_configs=False,
    ),
    "bf16": dict(
        matmul_fidelity=HiFi4, sdpa_fidelity=HiFi4, norm_fidelity=HiFi4, mlp_fidelity=HiFi4,
        matmul_fp32_acc=True, sdpa_exp_approx=False, mlp_fast_silu=False,
        dt_mlp_up=BF16, dt_mlp_down=BF16, dt_sdpa_v=BF16, dt_merger_fc1=BF16,
        dt_activations=BF16, dt_weights=BF16, dt_rope_tables=BF16,
        prefer_dram_intermediates=True, use_tuned_program_configs=False,
    ),
}


def _check_resolution() -> list[str]:
    failures: list[str] = []
    saved = os.environ.get("DOTS_OCR_VISION_PRECISION")
    try:
        for tier, expected in EXPECTED.items():
            os.environ["DOTS_OCR_VISION_PRECISION"] = tier
            prof = V._resolve_vision_precision()
            if prof.tier != tier:
                failures.append(f"[{tier}] resolved tier name = {prof.tier!r}")
            for field, want in expected.items():
                got = getattr(prof, field)
                if got != want:
                    failures.append(f"[{tier}] {field}: got {got!r}, expected {want!r}")
        # Unknown tier must fall back to baseline (the safe, HEAD-equivalent default).
        os.environ["DOTS_OCR_VISION_PRECISION"] = "does-not-exist"
        prof = V._resolve_vision_precision()
        if prof.tier != "baseline":
            failures.append(f"[unknown-tier] fallback = {prof.tier!r}, expected 'baseline'")
        # Unset must also default to baseline (purely additive: HEAD behavior preserved).
        os.environ.pop("DOTS_OCR_VISION_PRECISION", None)
        prof = V._resolve_vision_precision()
        if prof.tier != "baseline":
            failures.append(f"[unset] default = {prof.tier!r}, expected 'baseline'")
    finally:
        if saved is None:
            os.environ.pop("DOTS_OCR_VISION_PRECISION", None)
        else:
            os.environ["DOTS_OCR_VISION_PRECISION"] = saved
    return failures


def _check_module_import_per_tier() -> list[str]:
    """Import the module in a fresh subprocess per tier: validates the module-level
    ``_VP`` singleton + ``VISION_*_MATH_FIDELITY`` wiring and that nothing errors at
    import for any tier."""
    failures: list[str] = []
    snippet = (
        "import tt_symbiote.models.dots_ocr.dots_ocr_vision as m;"
        "print(m._VP.tier, m.VISION_MATMUL_MATH_FIDELITY, m.VISION_SDPA_MATH_FIDELITY,"
        " m.VISION_NORM_MATH_FIDELITY)"
    )
    fidelity_name = {"baseline": "LoFi", "hifi2": "HiFi2", "hifi4": "HiFi4", "bf16": "HiFi4"}
    for tier in EXPECTED:
        env = dict(os.environ, DOTS_OCR_VISION_PRECISION=tier)
        r = subprocess.run(
            [sys.executable, "-c", snippet], env=env, capture_output=True, text=True
        )
        if r.returncode != 0:
            failures.append(f"[{tier}] import failed: {r.stderr.strip().splitlines()[-1:]!r}")
            continue
        out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        parts = out.split()
        if not parts or parts[0] != tier:
            failures.append(f"[{tier}] module _VP.tier mismatch: {out!r}")
        elif fidelity_name[tier] not in out:
            failures.append(f"[{tier}] VISION_*_MATH_FIDELITY not {fidelity_name[tier]}: {out!r}")
    return failures


def main() -> int:
    all_failures = _check_resolution() + _check_module_import_per_tier()
    if all_failures:
        print("FAIL: vision precision profile drifted from spec:")
        for f in all_failures:
            print("  -", f)
        return 1
    print("PASS: all tiers (baseline/hifi2/hifi4/bf16) resolve to spec; baseline == HEAD a418638.")
    print("      module imports cleanly per tier with correct VISION_*_MATH_FIDELITY wiring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
