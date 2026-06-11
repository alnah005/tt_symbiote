# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Block-diffusion generation loop for the TTNN DiffusionGemma stack.

The denoising control flow (canvas init, entropy-bound acceptance, renoising,
temperature schedule, stopping) is host-side and reuses HF's tested sampler
classes; only the per-step *model forward* (encoder once -> decoder+lm_head per
step) runs on TTNN hardware. This mirrors HF ``DiffusionGemmaGenerationMixin``:

  1. encoder(prompt) -> per-layer KV cache                       [TTNN, once]
  2. canvas = random tokens; self_conditioning_logits = None
  3. for cur_step in range(max_denoising_steps, 0, -1):
       soft_emb = softmax(sc_logits) @ embed_weight * embed_scale [host, if any]
       hidden   = decoder(canvas, dec_pos, encoder_kv, soft_emb)  [TTNN]
       logits   = lm_head(hidden)                                 [TTNN]
       logits   = temperature_schedule(logits, cur_step)          [host]
       denoiser = multinomial(softmax(logits)); argmax = argmax(logits)
       canvas   = renoise(accept(canvas, denoiser, logits))       [host sampler]
       sc_logits = logits
  4. return final argmax canvas (token ids)

The encoder K,V cache is supplied as a list of (K, V) ttnn tensors (e.g. from
``TTNNDiffusionGemmaEncoderTextModel.forward_with_cache``), and per-layer-type
decoder cos/sin (offset past the encoder sequence) as a dict of ttnn tensors.
"""

from __future__ import annotations

import torch
import ttnn


def _to_torch(t, mesh_device, ref_shape):
    """Readback the V-sharded lm_head logits: each chip holds [B, S, V/TP]; concat on
    the last (vocab) axis reassembles the full [B, S, V], then reshape to ref_shape."""
    g = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
    return g.reshape(ref_shape)


def _replicate(t, mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    if t.dtype == torch.float32 and dtype == ttnn.bfloat16:
        t = t.to(torch.bfloat16)
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def block_diffusion_generate(
    decode_step,
    embed_weight,
    embed_scale,
    *,
    mesh_device,
    sampler,
    temperature_processor=None,
    stopping_criteria=None,
    max_denoising_steps,
    canvas_length,
    vocab_size,
    batch_size=1,
    return_logits_step0=False,
):
    """Drive the block-diffusion denoising loop on the TTNN decoder + lm_head.

    ``decode_step(canvas_ids_uint32_ttnn, sc_signal_ttnn_or_None) -> logits_ttnn``
    runs one decoder+lm_head forward; the caller (pipeline) supplies a closure that
    routes the 30-layer stack through the ``@trace_enabled`` unit so the per-step
    forward is a Metal-Trace replay under ``TT_SYMBIOTE_RUN_MODE=TRACED`` (and a plain
    eager forward under NORMAL). The denoising control flow below stays host-side and
    model-agnostic.

    ``sampler`` is an HF ``EntropyBoundSampler``; ``temperature_processor`` an HF
    ``LinearTemperatureScheduleLogitsProcessor`` (or None); ``stopping_criteria``
    an HF ``StableAndConfidentStoppingCriteria`` (or None). Returns the final
    argmax canvas ``[batch, canvas_length]`` (and, if requested, the step-0
    decoder logits for PCC validation against the HF reference)."""
    device = torch.device("cpu")
    current_canvas = sampler.initialize_canvas(batch_size=batch_size, device=device)
    argmax_canvas = current_canvas.clone()
    sc_logits = None
    finished = torch.zeros(batch_size, dtype=torch.bool)
    step0_logits = None

    for cur_step in range(max_denoising_steps, 0, -1):
        # Self-conditioning soft embeddings from the previous step's logits.
        if sc_logits is not None:
            soft = torch.matmul(torch.softmax(sc_logits, dim=-1), embed_weight) * embed_scale
            sc_signal = _replicate(soft, mesh_device)
        else:
            sc_signal = None

        canvas_tt = _replicate(
            current_canvas.to(torch.int32), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
        )
        logits_tt = decode_step(canvas_tt, sc_signal)
        raw_logits = _to_torch(logits_tt, mesh_device, (batch_size, canvas_length, vocab_size))
        if return_logits_step0 and step0_logits is None:
            step0_logits = raw_logits.clone()

        processed = (
            temperature_processor(None, raw_logits, cur_step=cur_step)
            if temperature_processor is not None
            else raw_logits
        )
        probs = torch.softmax(processed, dim=-1, dtype=torch.float32)
        denoiser = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
        denoiser = denoiser.squeeze(-1).view(batch_size, canvas_length)
        new_argmax = torch.argmax(processed, dim=-1)

        accepted = sampler.accept_canvas(current_canvas, denoiser, processed, cur_step)
        new_canvas = sampler.renoise_canvas(accepted, cur_step)

        if stopping_criteria is not None:
            if finished.any():
                new_argmax = torch.where(finished[:, None], argmax_canvas, new_argmax)
                new_canvas = torch.where(finished[:, None], current_canvas, new_canvas)
            finished = finished | stopping_criteria(new_argmax, processed)

        current_canvas = new_canvas
        argmax_canvas = new_argmax
        sc_logits = processed
        if stopping_criteria is not None and bool(finished.all()):
            break

    if return_logits_step0:
        return argmax_canvas, step0_logits
    return argmax_canvas
