# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL TTNN recipe — CPU-first port produced via the
``port-hf-model-to-tt-symbiote`` skill.

Targets :class:`transformers.Qwen3VLForConditionalGeneration` (the
image-text-to-text head over the vision tower + text decoder).

What this commit ships
----------------------

A *CPU-first* recipe identical in shape to
:class:`tt_symbiote.models.gemma4.Gemma4Recipe`:

* :meth:`Qwen3VLRecipe.build_module_dict` returns an empty dict; no
  PyTorch submodule is replaced. Execution runs entirely through the
  HuggingFace reference modeling on the CPU.
* :meth:`Qwen3VLRecipe.post_register` patches ``model.device`` to
  ``cpu`` and stashes the per-variant TTNN runtime config under
  ``_tt_runtime_config`` for downstream wrappers to consult.
* :meth:`Qwen3VLRecipe.make_kv_cache` is intentionally not implemented;
  the :func:`register_recipe` decorator installs a no-op default which
  leaves HF's :class:`DynamicCache` in place — sufficient for the short
  ``"dog"`` generation in :file:`examples/e2e/run_qwen3_vl_2b.py`.

The three class-level lists — ``tt_implemented``, ``cpu_fallback``,
``out_of_scope`` — declare *design-time* coverage and feed
:func:`tt_symbiote.compatibility.report`. The runtime ledger maintained
by :mod:`tt_symbiote.utils.compatibility` will stay empty during a
CPU-first run, and that absence is the correctness signal (nothing
tried to use TTNN and silently fell back).

Relative to Gemma-4 the OOS list is shorter — Qwen3-VL has no audio
tower — and the vision-side enumeration is different (Qwen3 uses
``Qwen3VLVisionBlock`` + ``Qwen3VLVisionPatchMerger`` instead of
Gemma-4's ``Gemma4VisionEncoderLayer`` / ``Gemma4MultimodalEmbedder``).
"""

from __future__ import annotations

import torch

from tt_symbiote.auto.auto_mappings import register_recipe
from tt_symbiote.models.qwen3_vl.configuration_qwen3_vl import lookup_ttnn_tuning

__all__ = ["Qwen3VLRecipe"]


# ---------------------------------------------------------------------------
# Design-time coverage manifests
# ---------------------------------------------------------------------------
#
# Source of truth: the class definitions in
# ``transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py``.
# Order within each list follows the HF file top-down (vision first,
# then text, then top-level composites) so a diff against upstream is
# easy to read.


_TT_IMPLEMENTED: list[str] = []


_CPU_FALLBACK: list[str] = [
    # ----- Vision tower (Qwen3VLVisionModel) -----
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionRotaryEmbedding",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionModel",
    # ----- Text decoder (Qwen3VLTextModel) -----
    "Qwen3VLTextRotaryEmbedding",
    "Qwen3VLTextRMSNorm",
    "Qwen3VLTextAttention",
    "Qwen3VLTextMLP",
    "Qwen3VLTextDecoderLayer",
    "Qwen3VLTextModel",
    # ----- Top-level composites -----
    "Qwen3VLPreTrainedModel",
    "Qwen3VLModel",
    "Qwen3VLForConditionalGeneration",
]


_OUT_OF_SCOPE: list[str] = [
    # ----- Output dataclasses (not torch modules) -----
    "BaseModelOutputWithDeepstackFeatures",
    "Qwen3VLModelOutputWithPast",
    "Qwen3VLCausalLMOutputWithPast",
]


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


@register_recipe(hf_class_name="Qwen3VLForConditionalGeneration")
class Qwen3VLRecipe:
    """CPU-first TTNN recipe for HuggingFace ``Qwen3VLForConditionalGeneration``."""

    tt_implemented: list[str] = _TT_IMPLEMENTED
    cpu_fallback: list[str] = _CPU_FALLBACK
    out_of_scope: list[str] = _OUT_OF_SCOPE

    def build_module_dict(self, model):
        """Return the (currently empty) PyTorch -> TTNN replacement map.

        The first Qwen3-VL commit ships no wrappers; the full HF
        reference modeling runs on the CPU. As wrappers are added in
        subsequent commits each entry moves from :data:`_CPU_FALLBACK`
        to :data:`_TT_IMPLEMENTED` and is registered in this dict.
        """
        return {}

    def post_register(self, model):
        """Patch ``model.device`` to ``cpu`` and stash the TTNN tuning.

        ``device`` is normally a property defined on
        :class:`transformers.PreTrainedModel` that walks the parameters.
        Overriding it on the *class* (not the instance) matches the
        convention from :class:`Gemma4Recipe` and :class:`ResNetRecipe`.

        ``_tt_runtime_config`` is consumed by :file:`examples/e2e/
        run_qwen3_vl_2b.py` and by downstream wrappers once they land.
        """
        type(model).device = property(lambda self: torch.device("cpu"))
        model._tt_runtime_config = lookup_ttnn_tuning(model)

    # ``make_kv_cache`` is intentionally not implemented. The
    # ``register_recipe`` decorator installs a no-op default which
    # leaves HF's ``DynamicCache`` in place — sufficient for the
    # CPU-first short-generation demos.
