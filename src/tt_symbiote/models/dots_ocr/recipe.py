# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Auto-API recipe for dots.ocr.

Lets a HuggingFace user drive dots.ocr through the canonical surface::

    from tt_symbiote import AutoModelForCausalLM, set_device
    import ttnn

    model = AutoModelForCausalLM.from_pretrained(
        "rednote-hilab/dots.ocr", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    set_device(model, ttnn.open_mesh_device(ttnn.MeshShape(8, 1)))   # DP on T3K
    out = model.generate(**inputs, max_new_tokens=24000)             # delegates to the TTNN pipeline

Unlike the in-place ``register_modules`` recipes (e.g. bailing_moe_v2), dots.ocr
runs through the bespoke :class:`TTNNDotsOCRPipeline` (paged-KV cache, DP batch
sharding, on-device argmax, traced decode). HF's own ``generate()`` loop cannot
drive that, so this recipe:

  * ``build_module_dict`` -> ``{}`` (no in-place swaps),
  * ``post_register`` -> patch ``model.generate`` to delegate to the pipeline,
  * ``make_kv_cache`` -> build the pipeline at ``set_device`` time (when the
    live mesh device is known) reusing the already-loaded HF weights.

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"
"""

from __future__ import annotations

import os
import warnings

import torch

from tt_symbiote.models._runtime_pins import RUNTIME_PINS
from tt_symbiote.models.auto.auto_mappings import register_recipe
from tt_symbiote.models.dots_ocr.pipeline import TTNNDotsOCRPipeline

# Single source of truth: tt_symbiote.models._runtime_pins.RUNTIME_PINS.
TT_METAL_COMMIT = RUNTIME_PINS["DotsOCRForCausalLM"]["tt_metal_commit"]


def _wants_dp() -> bool:
    return os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP"


def _pipeline_batch_size(device) -> int:
    """DP requires batch == num_devices (one stream per chip); else single stream."""
    num_devices = int(device.get_num_devices()) if hasattr(device, "get_num_devices") else 1
    if _wants_dp() and num_devices > 1:
        return num_devices
    return 1


def _make_generate_shim(model):
    """Return an HF-compatible ``generate`` bound to ``model`` that delegates to
    the TTNN pipeline stashed on ``model._tt_pipeline``."""

    def generate(  # noqa: D401 - mirrors HF signature loosely
        input_ids=None,
        attention_mask=None,  # accepted for HF-compat; pipeline derives its own masking
        pixel_values=None,
        image_grid_thw=None,
        max_new_tokens: int = 512,
        stop_on_eos: bool = True,
        **gen_kwargs,
    ):
        pipeline = getattr(model, "_tt_pipeline", None)
        if pipeline is None:
            raise RuntimeError(
                "dots.ocr TTNN pipeline is not built. Call "
                "tt_symbiote.set_device(model, mesh_device) before model.generate(...)."
            )
        if input_ids is None:
            raise ValueError("generate() requires input_ids")

        # Greedy on-device argmax only: warn (don't fail) on sampling knobs.
        for k in ("do_sample", "temperature", "top_p", "top_k", "num_beams"):
            if gen_kwargs.get(k) not in (None, False, 1, 0):
                warnings.warn(
                    f"dots.ocr TTNN pipeline ignores generate(..., {k}=...); "
                    f"decoding is deterministic greedy argmax.",
                    stacklevel=2,
                )

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if int(input_ids.shape[0]) != 1:
            raise ValueError(
                "The dots.ocr Auto/generate shim handles one prompt at a time "
                f"(got batch {int(input_ids.shape[0])}). For multi-stream data-parallel "
                "batching, call TTNNDotsOCRPipeline.generate directly with [num_devices, S] ids."
            )

        # Replicate the single prompt across the pipeline's DP streams.
        bs = int(getattr(model, "_tt_pipeline_batch", 1))
        run_ids = input_ids.expand(bs, -1).contiguous() if bs > 1 else input_ids

        pv = pixel_values.to(torch.bfloat16) if pixel_values is not None else None

        pipeline.warmup(run_ids, pixel_values=pv, image_grid_thw=image_grid_thw)
        generated = pipeline.generate(
            run_ids,
            pixel_values=pv,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            stop_on_eos=stop_on_eos,
        )

        # Normalize to stream 0's new-token list, then return HF-style
        # [1, prompt_len + new_len] so out_ids[len(in_ids):] / batch_decode work.
        new_tokens = generated[0] if (generated and isinstance(generated[0], list)) else generated
        full = input_ids[0].tolist() + list(new_tokens)
        return torch.tensor([full], dtype=torch.long)

    return generate


@register_recipe(hf_class_name="DotsOCRForCausalLM")
class DotsOCRRecipe:
    """Recipe that backs the canonical Auto/``generate`` surface with the pipeline."""

    def build_module_dict(self, model):
        # No in-place module swap -- the pipeline is constructed in make_kv_cache.
        return {}

    def post_register(self, model):
        # Stash where the weights came from (best effort) and install the
        # generate shim. The pipeline itself is built later, in make_kv_cache,
        # once set_device has provided the live mesh device.
        model._tt_dots_model_path = getattr(model.config, "_name_or_path", None)
        model._tt_pipeline = None
        model._tt_pipeline_batch = 1
        model.generate = _make_generate_shim(model)

    def make_kv_cache(self, model, device, **kwargs):
        # Called by set_device with the live mesh device -> build the optimized
        # TTNN pipeline now, reusing the already-loaded HF weights (hf_model=model)
        # so we don't pay a second multi-GB load.
        batch_size = _pipeline_batch_size(device)
        model._tt_pipeline = TTNNDotsOCRPipeline.from_hf_model(
            model_path=model._tt_dots_model_path,
            device=device,
            batch_size=batch_size,
            hf_model=model,
        )
        model._tt_pipeline_batch = batch_size
        model._tt_device = device
        # The pipeline owns its own paged KV cache; nothing to attach here.
        return None
