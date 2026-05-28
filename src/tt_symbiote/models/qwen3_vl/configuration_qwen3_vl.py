# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL configuration — HF re-export plus TTNN-side tuning lookup.

Mirrors :mod:`tt_symbiote.models.gemma4.configuration_gemma4`: the
upstream :class:`transformers.Qwen3VLConfig` (composite of
``text_config`` / ``vision_config``) fully describes the architecture so
we re-export it untouched. What lives here is the per-checkpoint TTNN
runtime knobs — mesh shape, ``l1_small_size``, device dtype, hardware-
verification flag — that the recipe needs at ``set_device`` time and
that aren't part of HF's hyperparameter surface.

Six canonical Qwen3-VL checkpoints are catalogued. Phase 7 ships
``Qwen/Qwen3-VL-2B-Instruct`` as the verified single-chip target; the
larger dense variants and the two MoE variants are listed so the
shape-fallback resolver can place a finetune onto the right tuning
profile, but they are not exercised on hardware in this commit.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

__all__ = [
    "Qwen3VLConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
    "QWEN3_VL_TTNN_TUNING",
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
    "dtype": "bfloat16",
    # ``True`` once a real end-to-end run has produced semantically
    # correct outputs on the listed mesh shape.
    "hw_verified": False,
    # Notes consumed by ``docs/supported_models.md``.
    "notes": "",
}


QWEN3_VL_TTNN_TUNING: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3-VL-2B-Instruct": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 1),
        "hw_verified": True,
        "notes": (
            "Single-chip N150 target. CPU-first port produced via the "
            "`port-hf-model-to-tt-symbiote` skill (first model landed "
            "through the skill). Verified on N150 via "
            "examples/e2e/run_qwen3_vl_2b.py: identifies the dog in "
            "tests/images/test-dog.png as a 'Golden Retriever puppy' "
            "from the prompt 'What is this animal in the photo?'. "
            "Zero runtime fallbacks (everything declared cpu_fallback "
            "in the recipe)."
        ),
    },
    "Qwen/Qwen3-VL-4B-Instruct": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 1),
        "hw_verified": False,
        "notes": "Same recipe as the 2B; not yet exercised on hardware.",
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 1),
        "hw_verified": False,
        "notes": "Same recipe as the smaller dense variants; not yet exercised on hardware.",
    },
    "Qwen/Qwen3-VL-32B-Instruct": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 8),  # T3K
        "hw_verified": False,
        "notes": (
            "T3K target once TTNN tensor-parallel wrappers land. CPU-first "
            "execution requires substantial host RAM (~60 GB BF16 + KV cache)."
        ),
    },
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 8),
        "hw_verified": False,
        "notes": (
            "MoE variant; uses a distinct top-level head (Qwen3VLMoeForConditionalGeneration) "
            "in transformers, so it lives under qwen3_vl_moe rather than this package. "
            "Listed here for users searching for the model name; route to qwen3_vl_moe once "
            "that recipe lands."
        ),
    },
    "Qwen/Qwen3-VL-235B-A22B-Instruct": {
        **_DEFAULT_TUNING,
        "mesh_shape": (1, 8),
        "hw_verified": False,
        "notes": (
            "Largest MoE variant; routes via qwen3_vl_moe once that recipe lands. "
            "Listed here only for variant-name discoverability."
        ),
    },
}


_SHAPE_TO_CHECKPOINT: Dict[Tuple[int, int], str] = {
    # (text_config.num_hidden_layers, text_config.hidden_size) -> canonical id.
    # Numbers derived from the published Qwen3-VL configs on HF Hub.
    # The (layers, hidden_size) tuple disambiguates each dense variant
    # uniquely without needing to download weights.
    (28, 2048): "Qwen/Qwen3-VL-2B-Instruct",
    (36, 2560): "Qwen/Qwen3-VL-4B-Instruct",
    (36, 4096): "Qwen/Qwen3-VL-8B-Instruct",
    (64, 5120): "Qwen/Qwen3-VL-32B-Instruct",
}


def lookup_ttnn_tuning(model: Any) -> Dict[str, Any]:
    """Resolve the TTNN tuning dict for a loaded Qwen3-VL model.

    Same lookup ladder as :func:`tt_symbiote.models.gemma4.lookup_ttnn_tuning`:

    1. Exact match on ``model.config._name_or_path``.
    2. Shape match on ``(text_config.num_hidden_layers,
       text_config.hidden_size)`` against the canonical dense variants.
    3. ``_DEFAULT_TUNING`` fallback (``hw_verified=False``, single-chip mesh).

    A *copy* is returned so downstream mutation in
    :meth:`Qwen3VLRecipe.post_register` does not pollute the table for
    the next ``from_pretrained`` call.
    """
    config = getattr(model, "config", None)

    name = getattr(config, "_name_or_path", None) if config is not None else None
    if isinstance(name, str) and name in QWEN3_VL_TTNN_TUNING:
        return dict(QWEN3_VL_TTNN_TUNING[name])

    if config is not None:
        text_config = getattr(config, "text_config", None)
        if (
            text_config is not None
            and hasattr(text_config, "num_hidden_layers")
            and hasattr(text_config, "hidden_size")
        ):
            shape_key = (
                int(text_config.num_hidden_layers),
                int(text_config.hidden_size),
            )
            canonical = _SHAPE_TO_CHECKPOINT.get(shape_key)
            if canonical is not None:
                return dict(QWEN3_VL_TTNN_TUNING[canonical])

    return dict(_DEFAULT_TUNING)
