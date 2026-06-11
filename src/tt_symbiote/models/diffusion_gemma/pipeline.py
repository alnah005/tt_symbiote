# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end TTNN pipeline for DiffusionGemma block-diffusion generation.

Mirrors the dots_ocr pattern (``TTNNDotsOCRPipeline`` + ``PipelineConfig``): a
single object that owns the on-device module graph and orchestrates the full
flow, built once from an HF model at ``set_device`` time and driven by
``model.generate`` via the recipe.

Graph (all sharded TP=4 on P150x4; weights tied encoder<->decoder => one copy):
  encoder:  embed -> TTNNDiffusionGemmaLayerStack (exports per-layer KV cache)
  decoder:  embed -> self-conditioning -> TTNNDiffusionGemmaLayerStack
            (cross-attends to encoder KV) -> norm
  head:     lm_head + final logit softcapping
  loop:     host-orchestrated block-diffusion denoising (HF sampler classes),
            feeding each step's logits back as self-conditioning soft embeddings.

The encoder/decoder layer lists are wrapped in ``TTNNDiffusionGemmaLayerStack``
(trace-enabled) -- the unit Metal Trace captures as one trace across all layers.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@contextmanager
def _force_normal_run():
    """Temporarily bind the eager NORMAL run implementation, restoring on exit.

    ``TTNNModule.call`` reads the process-global ``module.TENSOR_RUN_IMPLEMENTATION``
    (bound once at import from ``TT_SYMBIOTE_RUN_MODE``). Swapping it lets a region run
    eagerly even under ``TRACED`` -- used for the one-shot encoder pass so it does not
    spawn throwaway per-op traces."""
    import tt_symbiote.core.module as _mod
    from tt_symbiote.core.run_config import _RUN_MODE_REGISTRY

    saved = _mod.TENSOR_RUN_IMPLEMENTATION
    _mod.TENSOR_RUN_IMPLEMENTATION = _RUN_MODE_REGISTRY["NORMAL"]
    try:
        yield
    finally:
        _mod.TENSOR_RUN_IMPLEMENTATION = saved


from tt_symbiote.models.diffusion_gemma.generation_diffusion_gemma import _replicate, block_diffusion_generate
from tt_symbiote.models.diffusion_gemma.modeling_diffusion_gemma import (
    TTNNDiffusionGemmaDecoderTextModel,
    TTNNDiffusionGemmaEncoderTextModel,
    TTNNDiffusionGemmaLayerStack,
    TTNNDiffusionGemmaLMHead,
    share_text_weights,
)
from tt_symbiote.utils.device_management import set_device


@dataclass
class PipelineConfig:
    hidden_size: int
    layer_types: list
    vocab_size: int
    final_logit_softcapping: float
    rms_norm_eps: float
    canvas_length: int = 256
    max_denoising_steps: int = 48
    # Sampler/temperature defaults match the model's HF generation_config
    # (diffusiongemma-26B-A4B-it: entropy_bound=0.1, t_min=0.4, t_max=0.8). The
    # block-diffusion entropy-bound sampler is sensitive to these -- a mismatch vs
    # HF skews which low-entropy tokens are accepted/renoised each step.
    entropy_bound: float = 0.1
    t_min: float = 0.4
    t_max: float = 0.8


class TTNNDiffusionGemmaPipeline:
    """Owns the sharded on-device graph and drives block-diffusion generation."""

    def __init__(self, device, encoder, decoder, lm_head, rotary, config, embed_weight, embed_scale):
        self.device = device
        self.encoder = encoder  # TTNNDiffusionGemmaEncoderTextModel (shares decoder weights)
        self.decoder = decoder  # TTNNDiffusionGemmaDecoderTextModel
        self.lm_head = lm_head  # TTNNDiffusionGemmaLMHead
        self.rotary = rotary  # HF DiffusionGemmaTextRotaryEmbedding (host cos/sin)
        self.config = config
        self._embed_weight = embed_weight
        self._embed_scale = embed_scale
        # Trace-enabled stack views over the (already-built) layer objects -- the
        # unit Metal Trace captures as one trace across all 30 layers. The encoder
        # runs ONCE (eager, via forward_with_cache); only the decoder stack is invoked
        # via __call__ in the per-step loop so it traces under TRACED mode. Bind both
        # (idempotent over already-resident layer weights) so set_device's lifecycle
        # flags + .device are set -- TracedRun asserts weights are already on device.
        self.encoder_stack = TTNNDiffusionGemmaLayerStack.from_layers(encoder.layers, config.layer_types)
        self.decoder_stack = TTNNDiffusionGemmaLayerStack.from_layers(decoder.layers, config.layer_types)
        set_device(self.encoder_stack, device)
        set_device(self.decoder_stack, device)
        self._layer_types = list(config.layer_types)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_hf_model(cls, hf_model, device, *, canvas_length=None, max_denoising_steps=48):
        """Build the pipeline from a loaded ``DiffusionGemmaForBlockDiffusion``,
        reusing its (tied) weights -- decoder built first (superset), encoder
        aliases the decoder's on-device handles (one resident copy)."""
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextRotaryEmbedding

        tc = hf_model.config.text_config
        enc_txt = hf_model.model.encoder.language_model
        dec_hf = hf_model.model.decoder

        decoder = TTNNDiffusionGemmaDecoderTextModel.from_torch(dec_hf)
        encoder = TTNNDiffusionGemmaEncoderTextModel.from_torch(enc_txt)
        lm_head = TTNNDiffusionGemmaLMHead.from_torch(hf_model.lm_head, tc.final_logit_softcapping)

        # SHARE: decoder first, alias encoder handles, then set_device (skip-load).
        set_device(decoder, device)
        share_text_weights(encoder, decoder)
        set_device(encoder, device)
        set_device(lm_head, device)

        # Sampler/temperature parameters from the model's HF generation_config
        # (fall back to the PipelineConfig defaults if absent).
        gen_cfg = getattr(hf_model, "generation_config", None)
        _defaults = PipelineConfig(
            hidden_size=0, layer_types=[], vocab_size=0, final_logit_softcapping=0.0, rms_norm_eps=0.0
        )
        t_min = getattr(gen_cfg, "t_min", None)
        t_max = getattr(gen_cfg, "t_max", None)
        eb = _defaults.entropy_bound
        sc = getattr(gen_cfg, "sampler_config", None) if gen_cfg is not None else None
        if isinstance(sc, dict) and sc.get("entropy_bound") is not None:
            eb = float(sc["entropy_bound"])
        cfg = PipelineConfig(
            hidden_size=tc.hidden_size,
            layer_types=list(tc.layer_types),
            vocab_size=tc.vocab_size,
            final_logit_softcapping=tc.final_logit_softcapping,
            rms_norm_eps=tc.rms_norm_eps,
            canvas_length=canvas_length or getattr(hf_model.config, "canvas_length", 256),
            max_denoising_steps=max_denoising_steps,
            entropy_bound=eb,
            t_min=t_min if t_min is not None else _defaults.t_min,
            t_max=t_max if t_max is not None else _defaults.t_max,
        )
        embed_scale = float(getattr(dec_hf.embed_tokens, "scalar_embed_scale", 0.0)) or math.sqrt(tc.hidden_size)
        return cls(
            device,
            encoder,
            decoder,
            lm_head,
            DiffusionGemmaTextRotaryEmbedding(tc),
            cfg,
            dec_hf.embed_tokens.weight.detach().float(),
            embed_scale,
        )

    # --------------------------------------------------------------- rotary
    def _pos_emb(self, positions):
        """Host rotary cos/sin per layer-type -> replicated ttnn dict."""
        dummy = torch.zeros(1, positions.shape[-1], self.config.hidden_size)
        return {
            lt: (
                _replicate(self.rotary(dummy, positions, layer_type=lt)[0], self.device),
                _replicate(self.rotary(dummy, positions, layer_type=lt)[1], self.device),
            )
            for lt in set(self._layer_types)
        }

    def _encode(self, input_ids):
        """Run the encoder once -> per-layer KV cache (ttnn) for cross-attention.

        Forced to NORMAL dispatch: the encoder is a ONE-SHOT pass, and under TRACED its
        submodule linears (called via __call__) would each spin up a throwaway per-op
        warmup+capture (~7 linears x N layers) that the decoder-stack cold compile then
        invalidates -- pure wasted work. NORMAL runs the same math eagerly; the exported
        (K, V) tensors are raw ttnn either way."""
        s_enc = int(input_ids.shape[-1])
        enc_pe = self._pos_emb(torch.arange(s_enc).unsqueeze(0))
        ids = _replicate(
            input_ids.to(torch.int32),
            self.device,
            layout=__import__("ttnn").ROW_MAJOR_LAYOUT,
            dtype=__import__("ttnn").uint32,
        )
        with _force_normal_run():
            _, kv_cache = self.encoder.forward_with_cache(ids, enc_pe)
        return kv_cache, s_enc

    # ------------------------------------------------------------- generate
    def warmup(self, input_ids):
        """Compile kernels: encoder pass + one decoder denoising step."""
        self.generate(input_ids, max_denoising_steps=1)

    def generate(self, input_ids, max_denoising_steps=None, return_logits_step0=False):
        """Block-diffusion denoising. encoder once -> per-step decoder+lm_head with
        self-conditioning feedback + entropy-bound accept/renoise (HF sampler)."""
        from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
            EntropyBoundSampler,
            EntropyBoundSamplerConfig,
            LinearTemperatureScheduleLogitsProcessor,
        )

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        steps = max_denoising_steps or self.config.max_denoising_steps
        canvas = self.config.canvas_length
        kv_cache, s_enc = self._encode(input_ids)
        # decoder positions continue past the encoder sequence
        dec_pe = self._pos_emb((torch.arange(canvas) + s_enc).unsqueeze(0))
        sampler = EntropyBoundSampler(
            EntropyBoundSamplerConfig(entropy_bound=self.config.entropy_bound), canvas, self.config.vocab_size, steps
        )
        temp = LinearTemperatureScheduleLogitsProcessor(self.config.t_min, self.config.t_max, steps)

        import ttnn

        def _decode_step(canvas_tt, sc_signal):
            """One decoder denoising step: embed + self-conditioning + traced 30-layer
            stack + final norm + lm_head. The eager pieces (embed/self-cond/norm/lm_head)
            run per step; the stack is invoked via ``__call__`` so it traces (TRACED) or
            runs eagerly (NORMAL). ``dec_pe``/``kv_cache`` are constant across steps ->
            captured by reference; only ``canvas_tt`` (hidden) varies and is copied in."""
            embeds = self.decoder.embed_tokens.forward(canvas_tt)
            sig = sc_signal if sc_signal is not None else ttnn.multiply(embeds, 0.0)
            h = self.decoder.self_conditioning.forward(embeds, sig)
            h = self.decoder_stack(h, position_embeddings=dec_pe, encoder_kv_cache=kv_cache)
            h = self.decoder.norm.forward(h)
            return self.lm_head.forward(h)

        return block_diffusion_generate(
            _decode_step,
            self._embed_weight,
            self._embed_scale,
            mesh_device=self.device,
            sampler=sampler,
            temperature_processor=temp,
            max_denoising_steps=steps,
            canvas_length=canvas,
            vocab_size=self.config.vocab_size,
            return_logits_step0=return_logits_step0,
        )
