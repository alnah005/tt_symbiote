# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma-4 configuration — HF re-export plus TTNN-side tuning lookup.

Mirrors :mod:`tt_symbiote.models.resnet.configuration_resnet` in spirit:
the upstream :class:`transformers.Gemma4Config` (composite of
``text_config`` / ``vision_config`` / ``audio_config``) fully describes the
architecture so we re-export it untouched. What lives here is the per-
checkpoint TTNN runtime knobs — mesh shape, ``l1_small_size``, device
dtype, hardware-verification flag — that the recipe needs at
``set_device`` time and that aren't part of HF's hyperparameter surface.

Two checkpoints are targeted in Phase 7:

* ``google/gemma-4-E2B-it`` — 2.3 B effective parameters, ~9.6 GB BF16,
  fits a single chip. The default hardware bring-up target.
* ``google/gemma-4-31B-it`` — 30.7 B dense parameters, ~58 GB BF16,
  needs the full T3K (1×8) mesh and tensor-parallel weight sharding once
  TTNN wrappers land. For the Phase 7 CPU-first port the demo simply
  validates that the recipe loads end-to-end on T3K.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)

__all__ = [
    "Gemma4AudioConfig",
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "GEMMA4_TTNN_TUNING",
    "lookup_ttnn_tuning",
]


_DEFAULT_TUNING: Dict[str, Any] = {
    # Mesh shape recommended for this variant. Demo scripts pass it
    # straight to ``ttnn.open_mesh_device``; users with non-default
    # hardware can override.
    "mesh_shape": (1, 1),
    # ``l1_small_size`` for ``ttnn.open_mesh_device``. The CPU-first port
    # does not exercise any TTNN kernels, so the conservative ResNet
    # value carries no real cost while leaving headroom for the first
    # TTNN wrappers to land.
    "l1_small_size": 245760,
    # Target dtype used by the demos when calling ``from_pretrained``.
    # ``"bfloat16"`` keeps memory tight without forcing fp16 numerics.
    "dtype": "bfloat16",
    # ``True`` once a real end-to-end run has produced semantically
    # correct outputs on the listed mesh shape. Surfaces in
    # ``docs/supported_models.md``.
    "hw_verified": False,
    # Notes consumed by ``docs/supported_models.md`` for status flags
    # that aren't a simple yes/no.
    "notes": "",
}


GEMMA4_TTNN_TUNING: Dict[str, Dict[str, Any]] = {
    "google/gemma-4-E2B-it": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 1),
        "hw_verified": True,
        "notes": (
            "CPU-first port; image+text functional via PyTorch fallback. "
            "Verified on N150 via examples/e2e/run_gemma4_e2b.py: identifies "
            "the dog in tests/images/test-dog.png from the prompt "
            "'What is this animal in the photo?'. Zero runtime fallbacks "
            "(everything declared cpu_fallback in the recipe)."
        ),
    },
    "google/gemma-4-E4B-it": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 1),
        "hw_verified": False,
        "notes": "Same recipe as E2B; not yet exercised on hardware.",
    },
    "google/gemma-4-31B-it": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 8),  # T3K
        "hw_verified": False,
        "notes": (
            "CPU-first port on T3K. ~58 GB BF16 weights; first-time "
            "download from HF Hub is large. Future TTNN port reuses the "
            "preserved legacy text-side wrappers under legacy_modeling_gemma4.py.bak."
        ),
    },
    "google/gemma-4-26B-A4B-it": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 8),
        "hw_verified": False,
        "notes": "MoE 26B variant; not exercised in Phase 7.",
    },
}


_SHAPE_TO_CHECKPOINT: Dict[Tuple[int, int, int], str] = {
    # (text_config.num_hidden_layers, has_vision, has_audio) -> canonical id.
    # Discriminates the four official sizes without fetching the Hub.
    # E2B: 35 text layers, vision yes, audio yes.
    # E4B: 42 text layers, vision yes, audio yes.
    # 31B: 60 text layers, vision yes, audio no.
    # 26B A4B: 30 text layers, vision yes, audio no (MoE).
    (35, 1, 1): "google/gemma-4-E2B-it",
    (42, 1, 1): "google/gemma-4-E4B-it",
    (60, 1, 0): "google/gemma-4-31B-it",
    (30, 1, 0): "google/gemma-4-26B-A4B-it",
}


def lookup_ttnn_tuning(model: Any) -> Dict[str, Any]:
    """Resolve the TTNN tuning dict for a loaded Gemma-4 model.

    Same lookup ladder pattern as ResNet:

    1. Exact match on ``model.config._name_or_path``.
    2. Shape match on ``(text_config.num_hidden_layers, has_vision,
       has_audio)`` against the canonical Google variants.
    3. ``_DEFAULT_TUNING`` fallback (``hw_verified=False``).

    A *copy* is returned so downstream mutation in :meth:`Gemma4Recipe.post_register`
    does not pollute the table for the next ``from_pretrained`` call.
    """
    config = getattr(model, "config", None)

    name = getattr(config, "_name_or_path", None) if config is not None else None
    if isinstance(name, str) and name in GEMMA4_TTNN_TUNING:
        return dict(GEMMA4_TTNN_TUNING[name])

    if config is not None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
            shape_key = (
                int(text_config.num_hidden_layers),
                1 if getattr(config, "vision_config", None) is not None else 0,
                1 if getattr(config, "audio_config", None) is not None else 0,
            )
            canonical = _SHAPE_TO_CHECKPOINT.get(shape_key)
            if canonical is not None:
                return dict(GEMMA4_TTNN_TUNING[canonical])

    return dict(_DEFAULT_TUNING)
