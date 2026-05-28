# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma-4 VLM TTNN recipe — Phase 7 CPU-first port.

Targets :class:`transformers.Gemma4ForConditionalGeneration` (the
image-text-to-text head over the audio + vision + text backbones).

What this commit ships
----------------------

A *CPU-first* recipe:

* :meth:`Gemma4Recipe.build_module_dict` returns an empty dict; no
  PyTorch submodule is replaced. Execution runs entirely through the
  HuggingFace reference modeling on the CPU.
* :meth:`Gemma4Recipe.post_register` patches ``model.device`` to
  ``cpu`` (matches the Phase 5 / Phase 6 convention so other
  ``tt_symbiote`` plumbing — graph viz, weight introspection — does not
  trip over a missing device attribute) and stashes the per-variant
  TTNN runtime config under ``_tt_runtime_config`` for downstream
  wrappers to consult.
* :meth:`Gemma4Recipe.make_kv_cache` returns ``None``; HF's
  :class:`DynamicCache` handles short generation just fine. The hook is
  kept so the next-phase TTNN port can swap in a paged cache without
  touching ``set_device``.

The three class-level lists — ``tt_implemented``, ``cpu_fallback``,
``out_of_scope`` — declare *design-time* coverage and feed
:func:`tt_symbiote.compatibility.report`. They are deliberately
exhaustive so a contributor can ``grep -F Gemma4`` against
:mod:`transformers.models.gemma4.modeling_gemma4` and see every class
accounted for. The runtime ledger maintained by
:mod:`tt_symbiote.utils.compatibility` will stay empty during a
CPU-first Gemma-4 run, and that absence is the correctness signal
(nothing tried to use TTNN and silently fell back).

What this commit does **not** ship (tracked as follow-ups)
----------------------------------------------------------

* TTNN wrappers for the **vision tower** (~6 modules + 2-D RoPE work).
* TTNN wrappers for the **text decoder**. The Phase 2 mechanical
  migration of the 31B dense text-side is preserved verbatim in
  :file:`legacy_modeling_gemma4.py.bak` (not imported) so the next
  iteration starts from a known scaffold.
* **Audio** support. Gemma-4 E2B/E4B natively process audio, but the
  Phase 7 demos only exercise the image+text path; the audio classes
  are catalogued under ``out_of_scope``.

Compatibility surface
---------------------

Run the dog demo (``examples/e2e/run_gemma4_e2b.py``) and call
:func:`tt_symbiote.compatibility.report` to see this layout:

.. code-block:: python

    {
      "model_class": "Gemma4ForConditionalGeneration",
      "design_time": {
        "tt_implemented": [],
        "cpu_fallback":   [... 19 entries: text + vision + multimodal ...],
        "out_of_scope":   [... 9 entries: audio + LM-only top-level + output dataclasses ...],
      },
      "runtime_observed": {"by_class": {}, "by_module": {}, "unexpected": []},
      ...
    }
"""

from __future__ import annotations

import torch

from tt_symbiote.auto.auto_mappings import register_recipe
from tt_symbiote.models.gemma4.configuration_gemma4 import lookup_ttnn_tuning

__all__ = ["Gemma4Recipe"]


# ---------------------------------------------------------------------------
# Design-time coverage manifests
# ---------------------------------------------------------------------------
#
# Source of truth for each list: the class definitions in
# ``transformers/src/transformers/models/gemma4/modeling_gemma4.py`` and
# its sibling ``configuration_gemma4.py``. Order within each list is
# top-down following the HF file (text -> vision -> multimodal -> top-level),
# so a diff against the upstream file is easy to read.
#
# Membership rules:
#   * ``tt_implemented``: this commit ships a TTNN wrapper that
#     ``build_module_dict`` swaps in. Empty in Phase 7.
#   * ``cpu_fallback``: HF class is *exercised* by the image-text-to-text
#     demo but stays as PyTorch in this commit. The runtime hook in
#     :mod:`tt_symbiote.core.run_config` will flag every entry here as
#     "expected" (not "unexpected") in :func:`compatibility.report`.
#   * ``out_of_scope``: HF class exists in the model file but is *not
#     touched* by the documented Phase 7 demos (audio tower, the
#     text-only ``Gemma4ForCausalLM`` head, output dataclasses).


_TT_IMPLEMENTED: list[str] = []


_CPU_FALLBACK: list[str] = [
    # ----- Shared building blocks (text + vision share these) -----
    "Gemma4ClippableLinear",
    "Gemma4RMSNorm",
    # ----- Vision tower (Gemma4VisionModel) -----
    "Gemma4VisionPatchEmbedder",
    "Gemma4VisionRotaryEmbedding",
    "Gemma4VisionAttention",
    "Gemma4VisionMLP",
    "Gemma4VisionEncoderLayer",
    "Gemma4VisionEncoder",
    "Gemma4VisionPooler",
    "Gemma4VisionModel",
    # ----- Multimodal projection -----
    "Gemma4MultimodalEmbedder",
    # ----- Text decoder (Gemma4TextModel) -----
    "Gemma4TextScaledWordEmbedding",
    "Gemma4TextRotaryEmbedding",
    "Gemma4TextAttention",
    "Gemma4TextMLP",
    # MoE pair only used by the 26B-A4B variant; harmless for dense models
    "Gemma4TextExperts",
    "Gemma4TextRouter",
    "Gemma4TextDecoderLayer",
    "Gemma4TextModel",
    # ----- Top-level composites -----
    "Gemma4Model",
    "Gemma4ForConditionalGeneration",
]


_OUT_OF_SCOPE: list[str] = [
    # ----- Output dataclasses (not torch modules) -----
    "Gemma4ModelOutputWithPast",
    "Gemma4CausalLMOutputWithPast",
    "Gemma4TextModelOutputWithPast",
    "Gemma4AudioModelOutput",
    # ----- Audio tower (Phase 7 demos are image+text only) -----
    "Gemma4AudioRelPositionalEncoding",
    "Gemma4AudioAttention",
    "Gemma4AudioSubSampleConvProjectionLayer",
    "Gemma4AudioSubSampleConvProjection",
    "Gemma4AudioFeedForward",
    "Gemma4AudioCausalConv1d",
    "Gemma4AudioLightConv1d",
    "Gemma4AudioLayer",
    "Gemma4AudioModel",
    # ----- Alternative top-level head not exercised in Phase 7 -----
    "Gemma4ForCausalLM",
]


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


@register_recipe(hf_class_name="Gemma4ForConditionalGeneration")
class Gemma4Recipe:
    """CPU-first TTNN recipe for HuggingFace ``Gemma4ForConditionalGeneration``."""

    tt_implemented: list[str] = _TT_IMPLEMENTED
    cpu_fallback: list[str] = _CPU_FALLBACK
    out_of_scope: list[str] = _OUT_OF_SCOPE

    def build_module_dict(self, model):
        """Return the (currently empty) PyTorch -> TTNN replacement map.

        Phase 7 ships no wrappers; the full HF reference modeling runs
        on the CPU. As wrappers are added in subsequent commits each
        entry moves from :data:`_CPU_FALLBACK` to :data:`_TT_IMPLEMENTED`
        and is registered in this dict.
        """
        return {}

    def post_register(self, model):
        """Patch ``model.device`` to ``cpu`` and stash the TTNN tuning.

        ``device`` is normally a property defined on
        :class:`transformers.PreTrainedModel` that walks the parameters.
        Overriding it on the *class* (not the instance) keeps the
        ``@strict`` Gemma-4 config happy and matches the convention
        from :class:`BailingMoEV2Recipe` and :class:`ResNetRecipe`.

        ``_tt_runtime_config`` is consumed by the example scripts (mesh
        shape, ``l1_small_size``, etc.) and by downstream wrappers once
        they land.
        """
        type(model).device = property(lambda self: torch.device("cpu"))
        model._tt_runtime_config = lookup_ttnn_tuning(model)

    # ``make_kv_cache`` is intentionally not implemented. The
    # ``register_recipe`` decorator installs a no-op default which
    # leaves HF's ``DynamicCache`` in place — sufficient for the Phase 7
    # short-generation demos.
