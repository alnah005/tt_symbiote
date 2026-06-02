# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""ResNet configuration — HF config re-export plus TTNN-side tuning lookup.

This module mirrors :mod:`transformers.models.resnet.configuration_resnet`
in filename so a contributor looking at ``tt_symbiote/models/resnet/``
finds the same shape they know from the upstream ``transformers``
source tree. Two things live here:

1. A re-export of :class:`transformers.ResNetConfig`, so callers
   can write ``from tt_symbiote.models.resnet import ResNetConfig``
   symmetrically with ``from transformers.models.resnet import ResNetConfig``.
   We do **not** subclass or modify the upstream config — the HF
   hyperparameters (``depths``, ``hidden_sizes``, ``layer_type``,
   ``downsample_in_first_stage``, ``downsample_in_bottleneck``,
   ``embedding_size``, ``num_channels``, ``hidden_act``) fully describe
   the architecture and ``tt_symbiote`` is happy to consume them as-is.

2. :data:`RESNET_TTNN_TUNING`, a per-variant lookup table of
   *TTNN runtime* knobs that the recipe needs at module-replacement time
   (slice strategy for the first 7x7 stem conv, L1 small-region size for
   single-device runs, etc.). These are *not* model hyperparameters and
   therefore do not belong in HF's ``ResNetConfig`` — they're the
   data-driven equivalent of the old ``test_resnet50.py``'s hard-coded
   ``device_params=[{"l1_small_size": 245760}]``.

:func:`lookup_ttnn_tuning` resolves which entry to use for a loaded model
by first trying the HF checkpoint name (``model.config._name_or_path``)
and falling back to a shape match on ``(tuple(config.depths),
config.layer_type)`` so locally-trained checkpoints that follow the
canonical Microsoft sizing still pick up the right TTNN knobs.
"""

from __future__ import annotations

from typing import Any, Dict

from transformers.models.resnet.configuration_resnet import ResNetConfig

__all__ = ["ResNetConfig", "RESNET_TTNN_TUNING", "lookup_ttnn_tuning"]


_DEFAULT_TUNING: Dict[str, Any] = {
    "l1_small_size": 245760,
    "first_conv_slice_strategy": None,
    "dtype": "bfloat16",
    "hw_verified": False,
}


RESNET_TTNN_TUNING: Dict[str, Dict[str, Any]] = {
    "microsoft/resnet-18": {
        **_DEFAULT_TUNING,
    },
    "microsoft/resnet-34": {
        **_DEFAULT_TUNING,
    },
    "microsoft/resnet-50": {
        **_DEFAULT_TUNING,
        "hw_verified": True,
    },
    "microsoft/resnet-101": {
        **_DEFAULT_TUNING,
    },
    "microsoft/resnet-152": {
        **_DEFAULT_TUNING,
    },
}


_SHAPE_TO_CHECKPOINT: Dict[tuple, str] = {
    ((2, 2, 2, 2), "basic"): "microsoft/resnet-18",
    ((3, 4, 6, 3), "basic"): "microsoft/resnet-34",
    ((3, 4, 6, 3), "bottleneck"): "microsoft/resnet-50",
    ((3, 4, 23, 3), "bottleneck"): "microsoft/resnet-101",
    ((3, 8, 36, 3), "bottleneck"): "microsoft/resnet-152",
}


def lookup_ttnn_tuning(model: Any) -> Dict[str, Any]:
    """Resolve the TTNN tuning dict for a loaded ResNet model.

    The lookup is a small ladder, each step strictly more permissive than
    the last so that the recipe gets a useful answer even on
    non-canonical checkpoints:

    1. Exact match on ``model.config._name_or_path`` (the HF-style
       checkpoint id, set by ``from_pretrained``).
    2. Shape match on ``(tuple(config.depths), config.layer_type)``
       against the canonical Microsoft variants. Catches locally-trained
       or community checkpoints that share an architecture with the
       upstream Microsoft sizing.
    3. The shared ``_DEFAULT_TUNING`` fallback. Marked
       ``hw_verified=False`` so downstream code can warn that the user is
       running an un-tuned variant.

    Returning a *copy* (not a reference into the module-level dict) so
    downstream mutation in ``post_register`` does not pollute the table
    for the next ``from_pretrained`` call.
    """
    config = getattr(model, "config", None)
    name = getattr(config, "_name_or_path", None) if config is not None else None
    if isinstance(name, str) and name in RESNET_TTNN_TUNING:
        return dict(RESNET_TTNN_TUNING[name])

    if config is not None and hasattr(config, "depths") and hasattr(config, "layer_type"):
        shape_key = (tuple(config.depths), config.layer_type)
        canonical = _SHAPE_TO_CHECKPOINT.get(shape_key)
        if canonical is not None:
            return dict(RESNET_TTNN_TUNING[canonical])

    return dict(_DEFAULT_TUNING)
