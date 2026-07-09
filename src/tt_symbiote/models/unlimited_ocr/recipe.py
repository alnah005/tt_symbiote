# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Auto-API recipe for baidu/Unlimited-OCR (DeepSeek-OCR-style VLM + MoE).

Follows the dots.ocr Pattern B: Unlimited-OCR runs through a bespoke TTNN
pipeline (vision DeepEncoder + projector + DeepSeek-V2 MoE decoder + paged/
sliding-window KV cache), which HF's own ``generate()`` loop cannot drive. So:

  * ``build_module_dict`` -> ``{}``  (no in-place ``register_modules`` swaps),
  * ``post_register``     -> stash a pipeline placeholder on the model,
  * ``make_kv_cache``     -> build the pipeline at ``set_device`` time.

The bespoke pipeline (``TTNNUnlimitedOcrPipeline.from_hf_model``; see
``examples/e2e/unlimited_ocr/``) is the production entry point, since HF's own
``generate()`` cannot drive it -- so ``make_kv_cache`` is intentionally a no-op.
Registration fires on import so the Auto factory recognizes
``UnlimitedOCRForCausalLM``.

TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"
"""

from __future__ import annotations

from tt_symbiote.models.auto.auto_mappings import register_recipe

TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"


@register_recipe(hf_class_name="UnlimitedOCRForCausalLM")
class UnlimitedOCRRecipe:
    """Recipe backing the canonical Auto/``generate`` surface with the (future)
    Unlimited-OCR TTNN pipeline. Methods are INSTANCE methods (not staticmethod)."""

    def build_module_dict(self, model):
        # No in-place module swap -- the bespoke pipeline is constructed in
        # make_kv_cache once the live mesh device is known (Pattern B).
        return {}

    def post_register(self, model):
        # Stash where the weights came from (best effort) and a pipeline slot.
        # The generate shim + pipeline build land in the traced-execution stage.
        model._tt_unlimited_ocr_model_path = getattr(model.config, "_name_or_path", None)
        model._tt_pipeline = None

    def make_kv_cache(self, model, device, batch_size: int = 1, **kwargs):
        # No-op: this VLM is driven by TTNNUnlimitedOcrPipeline.from_hf_model (see
        # examples/e2e/unlimited_ocr/), which owns the paged / sliding-window (128)
        # KV cache -- there is no Auto-API cache to build here.
        return None
