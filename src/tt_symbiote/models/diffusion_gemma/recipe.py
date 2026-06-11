# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Auto/recipe surface for DiffusionGemma (block-diffusion), dots_ocr-style.

Unlike the in-place ``register_modules`` recipes, DiffusionGemma owns a custom
generation flow (block-diffusion denoising), so the recipe:
  * build_module_dict -> {}                  (no in-place module swaps)
  * post_register     -> install a ``model.generate`` shim that delegates to the
                         TTNN pipeline stashed on ``model._tt_pipeline``
  * make_kv_cache     -> build the TTNNDiffusionGemmaPipeline at set_device time
                         (reusing the already-loaded HF weights; tied enc/dec ->
                         one on-device copy)

    from tt_symbiote import AutoModelForCausalLM, set_device
    model = AutoModelForCausalLM.from_pretrained("google/diffusiongemma-26B-A4B-it")
    set_device(model, mesh_device)                 # builds the pipeline
    canvas = model.generate(input_ids, max_denoising_steps=48)  # denoised token block
"""

import warnings

from tt_symbiote.models.auto.auto_mappings import register_recipe
from tt_symbiote.models.diffusion_gemma.pipeline import TTNNDiffusionGemmaPipeline

TT_METAL_COMMIT = "a9e84ad6ce70d53729ba2f558c113136e1c5cb20"


def _make_generate_shim(model):
    """HF-compatible ``generate`` bound to ``model`` delegating to the pipeline."""

    def generate(input_ids=None, attention_mask=None, max_denoising_steps=48, **gen_kwargs):
        pipeline = getattr(model, "_tt_pipeline", None)
        if pipeline is None:
            raise RuntimeError(
                "DiffusionGemma TTNN pipeline is not built. Call "
                "tt_symbiote.set_device(model, mesh_device) before model.generate(...)."
            )
        if input_ids is None:
            raise ValueError("generate() requires input_ids")
        for k in ("do_sample", "num_beams"):
            if gen_kwargs.get(k) not in (None, False, 1):
                warnings.warn(
                    f"DiffusionGemma block-diffusion ignores generate(..., {k}=...); "
                    "decoding uses the entropy-bound sampler + temperature schedule.",
                    stacklevel=2,
                )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        pipeline.warmup(input_ids)
        # Returns the denoised canvas [batch, canvas_length] of token ids.
        return pipeline.generate(input_ids, max_denoising_steps=max_denoising_steps)

    return generate


@register_recipe(hf_class_name="DiffusionGemmaForBlockDiffusion")
class DiffusionGemmaRecipe:
    """Backs Auto/``generate`` with the block-diffusion TTNN pipeline."""

    def build_module_dict(self, model):
        # No in-place swaps -- the pipeline is constructed in make_kv_cache.
        return {}

    def post_register(self, model):
        model._tt_pipeline = None
        model._tt_device = None
        model.generate = _make_generate_shim(model)

    def make_kv_cache(self, model, device, **kwargs):
        # Called by set_device with the live mesh device -> build the sharded TTNN
        # pipeline now, reusing the already-loaded HF weights (no second multi-GB load).
        model._tt_pipeline = TTNNDiffusionGemmaPipeline.from_hf_model(model, device)
        model._tt_device = device
        return None  # the pipeline owns its own (encoder KV) state
