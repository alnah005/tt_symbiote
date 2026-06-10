# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for per-model runtime pins and dependency groups.

Design (scalable to 100+ models, mirroring how ``transformers`` scales):

  PIN  (per-model, metadata)   -> ``RUNTIME_PINS[hf_class]["tt_metal_commit"]``
  INSTALL (single, shared)     -> ``RELEASE_TTNN`` + ``CAPABILITY_EXTRAS``

A Python process can import exactly ONE ``ttnn`` (a compiled extension with a
global device singleton), so we CANNOT pip-install a different ttnn per model.
We therefore separate two concerns that are easy to conflate:

* **Pin (per model):** the tt-metal commit a recipe was *verified against*. This
  is metadata -- it scales to 100+ entries, drives the runtime compatibility gate
  (``tt_symbiote.utils.runtime_compat``), and gives provenance. It is NEVER a pip
  dependency.
* **Install (one per release):** ``RELEASE_TTNN`` is the single ttnn runtime the
  release ships, declared once in ``pyproject.toml``'s base ``dependencies``.
  Per-capability Python extras (``CAPABILITY_EXTRAS``) are shared across models,
  exactly like ``transformers``' ``[vision]`` / ``[audio]`` extras.

The gate reconciles the two: a model whose ``tt_metal_commit`` matches the
installed ttnn's commit runs correctly; one that differs is flagged (the
"ported but not yet re-verified against the current runtime" frontier).

This module MUST stay import-light: no imports from ``tt_symbiote``,
``transformers``, or ``ttnn``. ``scripts/sync_ttnn_extras.py`` (a bare build-env
tool) and the software-only ``tests/auto`` tree import it without a hardware
stack present.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# INSTALL side (single, shared across every model in the release)
# --------------------------------------------------------------------------- #

# The one ttnn runtime this release ships. Pinned once into pyproject.toml's base
# `dependencies` by scripts/sync_ttnn_extras.py. Empty string => not yet chosen;
# the package is source-build-only until an empirically-verified version is set
# (see docs/development/ttnn_pinning.md, "Choosing RELEASE_TTNN").
RELEASE_TTNN: str = "==0.68.0"

# Shared, per-capability Python dependency groups (à la transformers extras).
# Models reference these by name in their RUNTIME_PINS["extras"] list. The
# `all` extra (auto-generated) is the union of every group below.
CAPABILITY_EXTRAS: dict[str, list[str]] = {
    # HF VLM processors (Gemma-4, Qwen3-VL, dots.ocr, …) hard-import torchvision.
    "vision": ["torchvision"],
    # Qwen2.x-VL-style processors used by dots.ocr and the Qwen-VL family.
    "qwen-vl": ["qwen-vl-utils"],
    # Add "audio", "video", … here as models need them; each is shared by every
    # model that lists it, so this stays O(capabilities), not O(models).
}

# version -> tt-metal commit, so the runtime gate can resolve the commit of the
# INSTALLED ttnn (the ttnn wheel does not expose its build commit today). Populate
# as the mapping is learned. Unknown versions => "cannot determine" advisory.
TTNN_VERSION_COMMITS: dict[str, str] = {
    # "0.69.0": "<tt-metal commit ttnn 0.69.0 was built from>",
}

# --------------------------------------------------------------------------- #
# SERVING side (per-model metadata; consumed by tt-inference-server's vLLM
# Generator adapter — NOT used by the HF surface, so it changes no behavior)
# --------------------------------------------------------------------------- #
#
# How deeply a recipe can be driven by the vLLM Generator contract. This is pure
# metadata: it lets the generic serving adapter dispatch prefill/decode without
# per-model code. See docs/development/tt_inference_server_integration.md §9.
#
#   S0_GREEDY_ENGINE   model emits tokens (on-device argmax); served greedy,
#                      model-managed KV, max_num_seqs=1. (e.g. dots.ocr pipeline)
#   S1_LOGITS_UNPAGED  model.forward returns logits but KV is not vLLM-paged;
#                      served one request at a time, vLLM samples.
#   S2_PAGED           model.forward returns logits over a vLLM-page-table-aware
#                      paged KV cache; full continuous batching + sampling.
SERVING_TIERS: frozenset[str] = frozenset(
    {"S0_GREEDY_ENGINE", "S1_LOGITS_UNPAGED", "S2_PAGED"}
)

# --------------------------------------------------------------------------- #
# PIN side (per model; scales to 100+ — one entry per supported recipe)
# --------------------------------------------------------------------------- #
#
# Schema per entry:
#   "tt_metal_commit": str   the tt-metal commit this recipe was verified against
#                            (its OWN value -- models do NOT share one commit).
#   "extras": list[str]      capability groups (keys of CAPABILITY_EXTRAS) the
#                            model's processor/runtime needs. May be empty.
#   "serving_tier": str      (optional) one of SERVING_TIERS; how the model is
#                            driven under vLLM. Defaults to S1_LOGITS_UNPAGED.
#
# Adding a model is a single dict entry here (+ its recipe). See
# docs/development/ttnn_pinning.md.
RUNTIME_PINS: dict[str, dict] = {
    "DotsOCRForCausalLM": {
        "tt_metal_commit": "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195",
        "extras": ["vision", "qwen-vl"],
        "serving_tier": "S0_GREEDY_ENGINE",
    },
    # Example of the per-model independence (each pins its OWN commit):
    # "BailingMoeV2ForCausalLM": {
    #     "tt_metal_commit": "f2e12917564cfdfd50f81debcc12970a557412c8",
    #     "extras": [],
    #     "serving_tier": "S2_PAGED",
    # },
}

# Default tier for entries that omit "serving_tier" (conservative: logits model,
# served one request at a time, no assumptions about paged KV).
_DEFAULT_SERVING_TIER = "S1_LOGITS_UNPAGED"


def serving_tier_for(hf_class: str) -> str:
    """Serving tier for an HF architecture (metadata; see SERVING_TIERS)."""
    pin = RUNTIME_PINS.get(hf_class, {})
    return pin.get("serving_tier", _DEFAULT_SERVING_TIER)


def all_extra_packages() -> list[str]:
    """Sorted union of every CAPABILITY_EXTRAS package (powers the `all` extra)."""
    pkgs: set[str] = set()
    for packages in CAPABILITY_EXTRAS.values():
        pkgs.update(packages)
    return sorted(pkgs)
