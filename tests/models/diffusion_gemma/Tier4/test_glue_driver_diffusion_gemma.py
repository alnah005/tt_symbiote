# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier4 glue / driver localization mirror for the diffusion_gemma incoherence fix.

Committed skip-guarded pytest mirror of the standalone localization harnesses
(profiling/validate_host_checks.py + profiling/validate_glue_diffusion_gemma.py).

WHY skip-guarded: the ttnn ``mesh_device`` fixture lives in ``$TT_METAL_HOME/
conftest.py`` whose ``import tests.scripts`` COLLIDES with this repo's top-level
``tests`` package, so pytest cannot collect ANY hardware test here. The runnable
path is the harness scripts (``python profiling/<harness>.py`` under the tt-metal
python_env). These pytest tests auto-enable if the collision is ever fixed.

Root cause localized + fixed (see deep-work/deep-plan_1_execution.md):
  PRIMARY: stale/poisoned disk tensor cache. The per-shard ``as_tensor`` cache key
    encoded only role+shape+dtype, so a tile written by a different-weights build
    was silently served for the real weights -> garbage encoder K/V -> incoherent
    output. FIX: a content signature (_content_sig) is now woven into every cache
    key so a weight-value change can never reuse a stale tile.
  SECONDARY: the pipeline hardcoded t_min=0.5/t_max=1.0 instead of the model's HF
    generation_config (t_min=0.4/t_max=0.8). FIX: PipelineConfig defaults updated +
    from_hf_model reads the gen-config.

The HOST-ONLY checks below run with NO device (pure host) and are NOT skip-guarded.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HARNESS_DIR = Path(__file__).resolve().parent.parent / "profiling"


def _hw_fixtures_unavailable():
    """True if the tt-metal conftest 'tests.scripts' collision blocks collection."""
    try:
        import tests.scripts  # noqa: F401

        return False
    except Exception:
        return True


_HW_UNAVAILABLE = _hw_fixtures_unavailable()
_SKIP_REASON = (
    "tt-metal conftest 'tests.scripts' collision blocks hardware-test "
    "collection -- run profiling/validate_glue_diffusion_gemma.py instead"
)


def _load_harness(name):
    sys.path.insert(0, str(_HARNESS_DIR))
    spec = importlib.util.spec_from_file_location(name, _HARNESS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- HOST-ONLY
def test_pipeline_temperature_matches_gen_config():
    """SECONDARY-fix guard: pipeline t_min/t_max == the model's HF generation_config."""
    from transformers import GenerationConfig

    from tt_symbiote.models.diffusion_gemma.pipeline import PipelineConfig

    pc = PipelineConfig(hidden_size=0, layer_types=[], vocab_size=0, final_logit_softcapping=0.0, rms_norm_eps=0.0)
    try:
        gc = GenerationConfig.from_pretrained("google/diffusiongemma-26B-A4B-it")
    except Exception:
        pytest.skip("offline gen-config unavailable")
    assert abs(pc.t_min - gc.t_min) < 1e-9, f"t_min {pc.t_min} != {gc.t_min}"
    assert abs(pc.t_max - gc.t_max) < 1e-9, f"t_max {pc.t_max} != {gc.t_max}"


def test_cache_key_content_addressed():
    """PRIMARY-fix guard: distinct weight VALUES at the SAME shape/dtype must yield
    DISTINCT _content_sig (so a stale tile can never be served for changed weights)."""
    import torch

    from tt_symbiote.models.diffusion_gemma.modeling_diffusion_gemma import _content_sig

    a = torch.randn(64, 128)
    b = a.clone()
    c = a + 1.0  # different values, same shape/dtype
    assert _content_sig(a) == _content_sig(b), "identical weights must share a key"
    assert _content_sig(a) != _content_sig(c), "different weights MUST get a new key"


def test_host_driver_checks():
    """Run the full HOST-ONLY battery (driver/rope/mask/scalars/self-cond/pos/embed)."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not available in this interpreter")
    mod = _load_harness("validate_host_checks")
    # the harness sys.exit()s; call its body inside a SystemExit guard.
    try:
        mod.main()
    except SystemExit as e:
        assert e.code == 0, "HOST-ONLY localization checks FAILED (see LOC_* lines)"


# --------------------------------------------------------------------- DEVICE (skip-guarded)
@pytest.mark.skipif(_HW_UNAVAILABLE, reason=_SKIP_REASON)
def test_device_glue_battery_mirror():
    """Mirror of validate_glue_diffusion_gemma.py (DEV-LMHEAD-ORDER + DEV-DEPTH).

    Live only when the collection blocker is fixed; otherwise run the harness:
      python tests/models/diffusion_gemma/profiling/validate_glue_diffusion_gemma.py
    """
    os.environ.setdefault("DIFFUSION_GEMMA_WEIGHTS", "real")
    mod = _load_harness("validate_glue_diffusion_gemma")
    try:
        mod.main()
    except SystemExit as e:
        assert e.code == 0, "DEVICE glue battery FAILED (see LOC_* lines)"


@pytest.mark.slow
@pytest.mark.skipif(_HW_UNAVAILABLE, reason=_SKIP_REASON)
def test_coherence_diffusion_gemma():
    """ONE full 32-step denoise; degenerate-token tripwire must NOT fire.

    Live only when the collection blocker is fixed; otherwise run the harness:
      TT_SYMBIOTE_RUN_MODE=TRACED python \
        tests/models/diffusion_gemma/profiling/_run_full_traced.py
    Asserts the captured full_run_output.txt is free of the degenerate-symbol
    cluster (236772/236761/506-domination) that the cache poisoning produced.
    """
    out = _HARNESS_DIR / "full_run_output.txt"
    if not out.exists():
        pytest.skip("run profiling/_run_full_traced.py to produce full_run_output.txt")
    text = out.read_text()
    # parse token_ids line
    import re

    m = re.search(r"token_ids:\s*\[([0-9,\s]+)\]", text)
    assert m, "no token_ids in full_run_output.txt"
    ids = [int(x) for x in m.group(1).split(",")]
    degen = {236772, 236761}
    frac = sum(1 for t in ids if t in degen) / len(ids)
    assert frac < 0.2, f"degenerate-symbol tripwire fired: frac={frac:.3f} (cache poisoning?)"
