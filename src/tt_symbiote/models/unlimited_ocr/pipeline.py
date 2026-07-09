# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized KV-cache decode pipeline for baidu/Unlimited-OCR on TTNN.

The scaffold TTNN model does prefill-only greedy decode: every step re-runs the
WHOLE VLM forward (vision DeepEncoder + projector + scatter-merge + 12 decoder
layers over the growing sequence + lm_head) -- O(n^2), and it recomputes the
(constant, expensive) vision tower on every single token. This pipeline turns
that into O(n):

  * PREFILL runs the vision tower + scatter-merge + LM ONCE over the whole prompt,
    fills a FIXED pre-allocated block-paged K/V cache (dots_ocr
    ``TTNNPagedAttentionKVCache``, via ``paged_fill_cache``), and emits the first token.
  * DECODE embeds ONE new token, runs the 12 decoder layers writing K/V IN-PLACE at a
    DEVICE-side cur_pos (``paged_update_cache``) and reading via ``paged_sdpa_decode``
    over the fixed buffer, then final-norm -> lm_head -> on-device argmax.

CORRECTNESS CONTRACT: attention is FULL-CAUSAL (the query attends over cached K/V
[0..cur_pos]) -- mathematically identical to re-prefill for causal attention. The
generated token ids match (a) the prefill-greedy demo and (b) the torch
``use_cache=False`` full-causal reference (validated: 18/18 on the sample doc). The
model's real long-form generate() uses a window-128 ring buffer; that long-form-parity
refinement is deliberately OUT OF SCOPE here (noted as future work).

DECODE TRACE: the fixed paged cache is written in-place at a device-side cur_pos
(advanced OUTSIDE the trace) and never changes shape, so the ``@trace_enabled``
``TTNNUnlimitedOcrDecodeGraph`` is trace-capturable. The pipeline ALWAYS traces the
decode step (warm-up double-run -> capture -> replay), scoping the TRACED dispatch to
the decode graph. Prefill (vision DeepEncoder + projector + scatter-merge + LM +
lm_head + on-device argmax) runs eagerly -- it is compute-bound, so tracing it is not
worthwhile. Traced decode tokens are bit-identical to the eager reference.

DATA-PARALLEL (``batched_vision=True`` on an (N,1) mesh): OCRs N images concurrently,
reusing the same traced decode graph across the mesh; ``graph_prefill``
(``TTNNUnlimitedOcrPrefillGraph``) drives the batched prefill.

TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import List, Optional

import torch
import ttnn

import tt_symbiote.core.module as _core_module

from tt_symbiote.core.module import (
    DeviceArch,
    StatefulTTNNModule,
    StatelessTTNNModule,
    run_on_devices,
)
from tt_symbiote.core.run_config import TracedRun, trace_enabled
from tt_symbiote.models.dots_ocr._attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
    dp_batch_shard_tensor_mapper,
)
from tt_symbiote.models.unlimited_ocr.modeling_unlimited_ocr import (
    TTNNUnlimitedOcrForCausalLM,
    UnlimitedOcrKVCache,  # noqa: F401 -- retained export (legacy concat cache)
)
from tt_symbiote.utils.device_management import set_device

TT_METAL_COMMIT = "a0b506c780979538b6d2fc1e57fdbfdfdabc7e31"

_ARCHS = (DeviceArch.P150, DeviceArch.P150x4, DeviceArch.T3K)


def _create_paged_kv_cache(cfg, device, batch_size: int = 1):
    """Build + device the FIXED pre-allocated paged KV cache for the LM decode.

    Reuses the proven dots_ocr ``TTNNPagedAttentionKVCache`` (block-paged, fixed
    device buffers written in-place at a device-side cur_pos). Sized for
    Unlimited-OCR: 12 layers, num_kv_heads=10, head_dim=128. block_size=64,
    64 blocks/stream -> 4096-token capacity/stream (prompt ~300 + max_new_tokens).

    ``batch_size`` == 1 for the single-device path (unchanged). For DATA-PARALLEL
    (batch_size == num_devices on a (N,1) mesh) the page table is batch-sharded
    (one row per device via ``dp_batch_shard_tensor_mapper``) so each device owns
    its own stream's 64 blocks; the K/V cache buffers are replicated per device."""
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    blocks_per_sequence = 64
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=max(64, batch_size * blocks_per_sequence),
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=int(cfg.num_hidden_layers),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        config=config,
        device=None,
    ).to_device(device)


@trace_enabled
class TTNNUnlimitedOcrDecodeGraph(StatefulTTNNModule):
    """Trace unit for one full-causal cached DECODE step.

    Chains embed(1 token) -> 12 paged decoder layers (write K/V in-place at the
    device-side cur_pos + read via paged_sdpa_decode over the fixed buffer) ->
    final RMSNorm -> lm_head -> on-device argmax, producing the next token id
    ``[1, 1]``. Mirrors ``TTNNDotsOCRDecodeGraph``: STATEFUL only because the
    traced subtree contains the stateful per-layer attention that writes the paged
    K/V cache; this graph holds no persistent trace state of its own, so
    ``reset_trace_state`` is a justified no-op (the framework's trace tree-reset
    resets every stateful descendant independently). ``post_trace_execute`` advances
    the paged cache's Python-side seq counters OUTSIDE the trace (once per step).

    This is the live trace unit: with the FIXED pre-allocated paged cache, all decode
    ops are in-place at a device cur_pos with no shape change, so warm-up -> capture ->
    replay is faithful and deterministic (traced tokens are bit-identical to the eager
    decode; validated e2e against the torch reference).
    """

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def reset_trace_state(self) -> None:
        # No persistent trace state on THIS graph; the stateful per-layer attention
        # is reset independently by TracedRun._reset_trace_state_tree.
        return None

    def __init__(self, model: TTNNUnlimitedOcrForCausalLM):
        super().__init__()
        self._m = model

    @run_on_devices(*_ARCHS)
    def forward(self, token_ids, cos, sin, cache_position, past_key_values=None):
        # token_ids / cos / sin / cache_position are POSITIONAL device tensors so
        # TracedRun buffers + refreshes them per replay; past_key_values is a kwarg
        # containing "past_key_value" so module_run passes it through un-transformed
        # (the paged cache is not a tensor). All host->device writes + the cur_pos
        # advance happen OUTSIDE this forward (in the pipeline), so nothing here
        # allocates-to-self or writes from host -> safe under trace capture.
        logits = self._m.forward(
            input_ids=token_ids,
            position_embeddings=(cos, sin),
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        tok = ttnn.argmax(logits, dim=-1, keepdim=True, use_multicore=True)
        return ttnn.reshape(tok, (1, 1))

    def post_trace_execute(self, func_args, func_kwargs, result):
        # Advance the paged cache's Python-side seq counters OUTSIDE the trace, once
        # per decode step (runs on warm-up, capture, AND every replay). The DEVICE-side
        # cur_pos is advanced by the pipeline before each call; this keeps the paged
        # cache's get_seq_length() consistent for observability. Mirrors
        # dots_ocr TTNNDotsOCRDecodeGraph.post_trace_execute.
        past = func_kwargs.get("past_key_values")
        if past is None or not hasattr(past, "update_seq_length"):
            return
        for layer in self._m.model.layers:
            past.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=1)


@trace_enabled
class TTNNUnlimitedOcrPrefillGraph(StatefulTTNNModule):
    """Trace unit for the ONE-TIME VLM prefill (STAGE-3, opt-in).

    Chains the whole prefill in one captured subtree: embed text ids -> vision
    DeepEncoder (SAM on-device 2D + CLIP) -> projector -> build vision block ->
    scatter-merge into the text embeds at the 273 <image> positions -> 12 DeepSeek
    decoder layers (each writing the prompt's rope'd K/V into the FIXED paged cache
    via ``paged_fill_on_device``) -> final RMSNorm -> lm_head -> on-device argmax of
    the LAST position, producing the first token id ``[1, 1]``.

    Mirrors ``TTNNDotsOCRPrefillGraph``: STATEFUL only because the traced subtree
    contains the stateful per-layer attention that writes the paged K/V cache. The
    fill writes positions 0..S-1 with ``batch_idx=0`` (independent of any counter),
    so the warm-up + capture double-run just overwrites the same slots -- idempotent.
    ``reset_trace_state`` is a justified no-op (the framework tree-reset re-baselines
    every stateful descendant independently, and this graph allocates nothing on
    ``self`` inside forward). ``post_trace_execute`` advances the paged cache's
    Python-side seq counters to the prompt length OUTSIDE the trace (observability;
    decode reads its absolute position from the pipeline, not from these counters).

    Trace viability was the key experiment: with the SAM torch-fallback removed the
    forward is pure ttnn + fixed-shape (1024x1024 global view -> 273 vision tokens +
    fixed text prompt), so warm-up -> capture -> replay is faithful and deterministic.
    Used by the DATA-PARALLEL prefill path (prefill_dp).
    """

    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def reset_trace_state(self) -> None:
        # No persistent trace state on THIS graph; the stateful per-layer attention
        # is reset independently by TracedRun._reset_trace_state_tree.
        return None

    def __init__(self, model: TTNNUnlimitedOcrForCausalLM):
        super().__init__()
        self._m = model

    @run_on_devices(*_ARCHS)
    def forward(self, token_ids, cos, sin, pixel_values, vision_idx, vision_mask,
                past_key_values=None):
        # token_ids / cos / sin / pixel_values / vision_idx / vision_mask are POSITIONAL
        # device tensors so TracedRun buffers + refreshes them per replay;
        # past_key_values is a kwarg containing "past_key_value" so module_run passes it
        # through un-transformed (the paged cache is not a tensor). cache_position=0 is a
        # constant here: the paged PREFILL fill uses batch_idx=0 (position-independent) and
        # the S>1 attention path never reads cache_position. All host->device writes + the
        # seq-counter advance happen OUTSIDE this forward -> safe under trace capture.
        logits = self._m.forward(
            input_ids=token_ids,
            pixel_values=pixel_values,
            position_embeddings=(cos, sin),
            vision_idx=vision_idx,
            vision_mask=vision_mask,
            past_key_values=past_key_values,
            cache_position=0,
        )
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        S = int(logits.shape[1])
        V = int(logits.shape[2])
        last = ttnn.slice(logits, [0, S - 1, 0], [1, S, V])
        tok = ttnn.argmax(last, dim=-1, keepdim=True, use_multicore=True)
        return ttnn.reshape(tok, (1, 1))

    def post_trace_execute(self, func_args, func_kwargs, result):
        # Advance the paged cache's Python-side seq counters to the prompt length
        # OUTSIDE the trace (runs on capture + every replay; NOT on warm-up). The fill
        # wrote positions 0..S-1; decode advances from there. Mirrors
        # dots_ocr TTNNDotsOCRPrefillGraph.post_trace_execute.
        past = func_kwargs.get("past_key_values")
        if past is None or not hasattr(past, "update_seq_length"):
            return
        S = int(func_args[0].shape[-1])
        for layer in self._m.model.layers:
            past.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=S)


class TTNNUnlimitedOcrPipeline(StatelessTTNNModule):
    """Standalone KV-cache greedy-decode driver for Unlimited-OCR.

    Not driven by HF ``generate()``; owns its own prefill + decode loop over a
    contiguous per-layer K/V cache. Attention is full-causal (see module docstring).
    """

    # This is a thin host-side orchestrator; it is never itself a compute leaf, so
    # its own forward is unused. Declared stateless (no trace double-run state).
    def reset_trace_state(self):
        return None

    def __init__(self, tt_model, device, cfg, rotary, image_token_id=128815,
                 eos_id=1, batched_vision=False):
        super().__init__()
        self.tt = tt_model
        self._device = device
        self.cfg = cfg
        self._rotary = rotary
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        self.vocab_size = int(cfg.vocab_size)
        self.num_layers = int(cfg.num_hidden_layers)
        self.image_token_id = int(image_token_id)
        self.eos_id = int(eos_id)
        # --- DATA-PARALLEL (DP) state (ADDITIVE; default single-device unchanged) ---
        # On a (N,1) mesh with batched_vision=True the pipeline OCRs N DIFFERENT
        # images CONCURRENTLY (device b -> image b): weights replicate, per-stream
        # inputs (input_ids / pixel_values / per-stream tokens) batch-shard on dim 0,
        # shared inputs (cos/sin/vision_idx/vision_mask/cur_pos) replicate, and
        # outputs gather via ConcatMeshToTensor(dim=0). num_devices == 1 -> every
        # DP branch is skipped and the single-stream path below runs byte-for-byte.
        self.num_devices = int(device.get_num_devices()) if hasattr(device, "get_num_devices") else 1
        self.batched_vision = bool(batched_vision) and self.num_devices > 1
        self._batch_size = self.num_devices if self.batched_vision else 1
        self._batch_input_mapper = (
            dp_batch_shard_tensor_mapper(device, self._batch_size) if self.batched_vision else None
        )
        # Paged KV cache: a FIXED pre-allocated block-paged buffer. Decode writes K/V
        # in-place at a device-side cur_pos (paged_update_cache) and reads via
        # paged_sdpa_decode -- nothing allocates or changes shape inside the traced
        # region, so the decode graph is trace-capturable.
        self.cache = _create_paged_kv_cache(cfg, device, batch_size=self._batch_size)
        # The @trace_enabled decode graph is the trace unit: embed 1 token -> 12 paged
        # decoder layers -> final norm -> lm_head -> on-device argmax. Deviced directly
        # (no weights of its own; wraps the already-deviced tt_model).
        self.graph_decode = TTNNUnlimitedOcrDecodeGraph(tt_model)
        self.graph_decode._device = device
        self.graph_decode._unique_name = "unlimited_ocr_decode_graph"
        self.graph_decode._bypass_tensor_wrapping = True
        # The @trace_enabled prefill graph: wraps the whole VLM prefill -> first-token
        # argmax. Used by the DATA-PARALLEL prefill path (prefill_dp).
        self.graph_prefill = TTNNUnlimitedOcrPrefillGraph(tt_model)
        self.graph_prefill._device = device
        self.graph_prefill._unique_name = "unlimited_ocr_prefill_graph"
        self.graph_prefill._bypass_tensor_wrapping = True
        # host-side cos/sin table, (re)computed lazily to cover the needed length.
        self._cos = None
        self._sin = None
        self._rope_len = 0
        # last measured timings (seconds)
        self.last_prefill_s = 0.0
        self.last_decode_s = 0.0
        self.last_num_decode = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_hf_model(cls, hf_model, cfg, device, batched_vision=False):
        """Build the pipeline from an already-loaded reference HF model + config.

        Reuses the loaded weights (no second load): constructs the TTNN
        ForCausalLM, binds it to ``device``, and captures the model's own rotary
        embedding so the RoPE (cos, sin) exactly matches the torch reference.

        ``batched_vision=True`` (on a (N,1) mesh) enables the ADDITIVE DATA-PARALLEL
        path: N DIFFERENT images OCR'd concurrently (device b -> image b). Default
        False -> the single-device path is byte-for-byte unchanged.
        """
        tt = TTNNUnlimitedOcrForCausalLM.from_torch(hf_model)
        set_device(tt, device)
        rotary = hf_model.model.layers[0].self_attn.rotary_emb
        image_token_id = getattr(getattr(hf_model, "config", None), "image_token_id", 128815)
        eos_id = getattr(getattr(hf_model, "config", None), "eos_token_id", 1) or 1
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]
        pipe = cls(tt, device, cfg, rotary, image_token_id=image_token_id,
                   eos_id=int(eos_id), batched_vision=batched_vision)
        pipe._unique_name = "unlimited_ocr_pipeline"
        return pipe

    # ------------------------------------------------------------------
    # Host-side helpers (torch is permitted: this is orchestration, not a
    # TTNNModule.forward compute path).
    # ------------------------------------------------------------------
    def _ensure_rope(self, seq_len: int) -> None:
        if self._cos is not None and self._rope_len >= seq_len:
            return
        pos = torch.arange(seq_len).unsqueeze(0)
        with torch.no_grad():
            cos, sin = self._rotary(torch.zeros(1, 1, seq_len, self.head_dim), pos)
        self._cos = cos.unsqueeze(1).float()  # [1,1,S,D]
        self._sin = sin.unsqueeze(1).float()
        self._rope_len = seq_len

    def _rope_slice(self, start: int, end: int):
        cos = self._cos[:, :, start:end, :].contiguous()
        sin = self._sin[:, :, start:end, :].contiguous()
        tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._device)
        tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._device)
        return tt_cos, tt_sin

    def _scatter_tensors(self, seq_mask: List[bool]):
        seq = len(seq_mask)
        m = torch.tensor(seq_mask, dtype=torch.bool)
        gather_idx = torch.zeros(1, seq, dtype=torch.int32)
        true_pos = m.nonzero(as_tuple=True)[0]
        gather_idx[0, true_pos] = torch.arange(1, true_pos.numel() + 1, dtype=torch.int32)
        mask_f = m.float().view(1, seq, 1)
        tt_idx = ttnn.from_torch(gather_idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device)
        tt_mask = ttnn.from_torch(mask_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._device)
        return tt_idx, tt_mask

    def _to_logits(self, out, seq: int) -> torch.Tensor:
        from tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(out, TorchTTNNTensor):
            out.elem = None
            return out.to_torch.float().reshape(1, seq, self.vocab_size)
        if isinstance(out, torch.Tensor):
            return out.float().reshape(1, seq, self.vocab_size)
        return ttnn.to_torch(out).float().reshape(1, seq, self.vocab_size)

    def _upload_ids(self, ids: List[int]):
        return ttnn.from_torch(
            torch.tensor([ids], dtype=torch.int32), dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device,
        )

    def _upload_cur_pos(self, position: int):
        """Device-side cur_pos ([batch=1] int32 ROW_MAJOR DRAM) for the paged decode.

        A fresh tensor per step: under TRACED, ``_replay`` copies its contents into
        the trace's persistent cur_pos buffer, so the device-side position advances
        correctly on every replay WITHOUT any allocation inside the traced region."""
        return ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @contextmanager
    def _traced_dispatch(self):
        """Scope the framework TRACED dispatch to a SINGLE ``@trace_enabled`` graph call.

        Patches the module-global run implementation only around the wrapped
        ``graph.__call__`` (``graph_decode``) so exactly one trace unit is
        captured/replayed; the graph's ``forward`` invokes its children via ``.forward``
        (not ``__call__``), so the scope covers exactly the trace unit."""
        saved = _core_module.TENSOR_RUN_IMPLEMENTATION
        _core_module.TENSOR_RUN_IMPLEMENTATION = TracedRun
        try:
            yield
        finally:
            _core_module.TENSOR_RUN_IMPLEMENTATION = saved

    # ------------------------------------------------------------------
    # Prefill / decode
    # ------------------------------------------------------------------
    def prefill(self, input_ids: List[int], seq_mask: Optional[List[bool]] = None,
                tt_px=None) -> int:
        """Run the ONE-TIME prefill: fill the PAGED K/V cache, return the first token.

        ``tt_px`` is an optional NHWC pixel tensor on device (VLM path); when None
        this is a text-only prefill. ``seq_mask`` marks the <image> positions.
        The attention prefill path calls ``paged_fill_on_device`` per layer to
        populate positions 0..seq-1 of the fixed paged buffers.
        """
        self.cache.reset()
        # External seq tracking: paged_fill/paged_update must NOT mutate the Python
        # counters (those advances happen OUTSIDE the trace, in prefill/decode
        # post_trace_execute for the traced paths, or the explicit loop below for the
        # NORMAL paths).
        self.cache._external_seq_tracking = True
        seq = len(input_ids)
        self._ensure_rope(seq + 1)
        tt_cos, tt_sin = self._rope_slice(0, seq)
        tt_ids = self._upload_ids(input_ids)
        t0 = time.perf_counter()
        if tt_px is not None:
            tt_idx, tt_mask = self._scatter_tensors(seq_mask)
            out = self.tt(input_ids=tt_ids, pixel_values=tt_px,
                          position_embeddings=(tt_cos, tt_sin),
                          vision_idx=tt_idx, vision_mask=tt_mask,
                          past_key_values=self.cache, cache_position=0)
        else:
            out = self.tt(input_ids=tt_ids, position_embeddings=(tt_cos, tt_sin),
                          past_key_values=self.cache, cache_position=0)
        logits = self._to_logits(out, seq)
        ttnn.synchronize_device(self._device)
        # Set the paged cache's Python-side seq counters to the prompt length (the
        # fill wrote positions 0..seq-1); decode advances them by 1 per step.
        for i in range(self.num_layers):
            self.cache.update_seq_length(layer_idx=i, seq_len=seq)
        self.last_prefill_s = time.perf_counter() - t0
        return int(logits[0, -1].argmax().item())

    def decode_step(self, prev_token_id: int, position: int) -> int:
        """One paged O(1) decode step via the trace unit ``graph_decode``.

        Embeds the single new token, RoPE at absolute ``position``, writes K/V
        IN-PLACE at the device-side cur_pos (``position``) and attends over the
        paged cache. The decode graph is captured once (warm-up) then replayed.
        cur_pos / cos / sin / token are uploaded here (OUTSIDE the traced region)
        and copied into the trace buffers by ``_replay``."""
        self._ensure_rope(position + 1)
        tt_cos, tt_sin = self._rope_slice(position, position + 1)   # [1,1,1,D]
        tt_ids = self._upload_ids([prev_token_id])                  # [1,1]
        cur_pos = self._upload_cur_pos(position)                    # [1] int32
        with self._traced_dispatch():
            tok = self.graph_decode(tt_ids, tt_cos, tt_sin, cur_pos, past_key_values=self.cache)
        tok_t = tok if isinstance(tok, torch.Tensor) else ttnn.to_torch(tok)
        return int(tok_t.reshape(-1)[0].item())

    def generate(self, input_ids: List[int], seq_mask: Optional[List[bool]] = None,
                 tt_px=None, max_new_tokens: int = 32, stop_on_eos: bool = True) -> List[int]:
        """Full greedy generation: one prefill + a decode loop. Returns new token ids."""
        prompt_len = len(input_ids)
        first = self.prefill(input_ids, seq_mask=seq_mask, tt_px=tt_px)
        generated = [first]
        if first == self.eos_id and stop_on_eos:
            self.last_num_decode = 0
            self.last_decode_s = 0.0
            return generated
        cur = first
        t0 = time.perf_counter()
        n_dec = 0
        for i in range(max_new_tokens - 1):
            pos = prompt_len + i  # absolute position of the token being generated
            nxt = self.decode_step(cur, pos)
            n_dec += 1
            if nxt == self.eos_id and stop_on_eos:
                break
            generated.append(nxt)
            cur = nxt
        ttnn.synchronize_device(self._device)
        self.last_decode_s = time.perf_counter() - t0
        self.last_num_decode = n_dec
        return generated

    # ==================================================================
    # DATA-PARALLEL (DP) path: N DIFFERENT images OCR'd CONCURRENTLY on a
    # (N,1) mesh (device b -> image b). Entirely ADDITIVE: only reachable
    # when batched_vision=True (num_devices>1). The single-stream methods
    # above are untouched.
    # ==================================================================
    def _rep(self):
        return ttnn.ReplicateTensorToMesh(self._device)

    def _rope_slice_dp(self, start: int, end: int):
        # cos/sin are identical across streams (same absolute positions) -> REPLICATE.
        cos = self._cos[:, :, start:end, :].contiguous()
        sin = self._sin[:, :, start:end, :].contiguous()
        tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=self._device, mesh_mapper=self._rep())
        tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=self._device, mesh_mapper=self._rep())
        return tt_cos, tt_sin

    def _scatter_tensors_dp(self, seq_mask: List[bool]):
        # <image> positions are identical across streams (same prompt layout) -> REPLICATE.
        seq = len(seq_mask)
        m = torch.tensor(seq_mask, dtype=torch.bool)
        gather_idx = torch.zeros(1, seq, dtype=torch.int32)
        true_pos = m.nonzero(as_tuple=True)[0]
        gather_idx[0, true_pos] = torch.arange(1, true_pos.numel() + 1, dtype=torch.int32)
        mask_f = m.float().view(1, seq, 1)
        tt_idx = ttnn.from_torch(gather_idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=self._device, mesh_mapper=self._rep())
        tt_mask = ttnn.from_torch(mask_f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=self._device, mesh_mapper=self._rep())
        return tt_idx, tt_mask

    def _upload_ids_dp(self, ids_batch):
        # ids_batch: [B, S] int -> BATCH-SHARDED (device b gets stream b's row).
        t = torch.tensor(ids_batch, dtype=torch.int32)
        return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self._device, mesh_mapper=self._batch_input_mapper)

    def _upload_cur_pos_dp(self, position: int):
        # Single global absolute position (same for every stream) -> REPLICATE [1] int32.
        return ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self._device,
            mesh_mapper=self._rep(), memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _upload_px_dp(self, px_nhwc_batch):
        # px_nhwc_batch: torch [B, H, W, 3] -> BATCH-SHARDED (device b gets image b).
        return ttnn.from_torch(px_nhwc_batch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self._device, mesh_mapper=self._batch_input_mapper)

    def _gather_tokens(self, tok) -> List[int]:
        t = tok if isinstance(tok, torch.Tensor) else ttnn.to_torch(
            tok, mesh_composer=ttnn.ConcatMeshToTensor(self._device, dim=0))
        return [int(x) for x in t.reshape(-1).tolist()]

    def prefill_dp(self, input_ids_batch, seq_mask: List[bool], px_nhwc_batch) -> List[int]:
        """DP prefill: fill each device's paged stream for its own image; return B first tokens.

        ``input_ids_batch`` is ``[B, S]`` (all streams same length S), ``seq_mask`` the
        SHARED <image> layout, ``px_nhwc_batch`` a host ``[B, H, W, 3]`` tensor (image b
        -> device b). Runs the WHOLE VLM prefill (vision + scatter + LM + argmax) once
        across the mesh -- B images in one device pass."""
        self.cache.reset()
        self.cache._external_seq_tracking = True
        seq = len(seq_mask)
        self._ensure_rope(seq + 1)
        tt_cos, tt_sin = self._rope_slice_dp(0, seq)
        tt_ids = self._upload_ids_dp(input_ids_batch)
        tt_idx, tt_mask = self._scatter_tensors_dp(seq_mask)
        tt_px = self._upload_px_dp(px_nhwc_batch)
        t0 = time.perf_counter()
        tok = self.graph_prefill(tt_ids, tt_cos, tt_sin, tt_px, tt_idx, tt_mask,
                                 past_key_values=self.cache)
        ttnn.synchronize_device(self._device)
        firsts = self._gather_tokens(tok)
        for i in range(self.num_layers):
            self.cache.update_seq_length(layer_idx=i, seq_len=seq)
        self.last_prefill_s = time.perf_counter() - t0
        return firsts

    def decode_step_dp(self, prev_tokens: List[int], position: int) -> List[int]:
        """One DP paged decode step across the mesh: B tokens in, B next tokens out.

        The decode graph call is scoped by ``_traced_dispatch`` EXACTLY as in the
        single-device ``decode_step`` -- the ONLY difference is the mesh + batch-B paged
        cache + batch-sharded per-stream inputs. The ``@trace_enabled`` ``graph_decode``
        is captured ONCE (warm-up double-run via ``warmup_dp``) and REPLAYED per step;
        the per-step token/cos/sin/cur_pos device tensors are refreshed into the trace's
        persistent buffers by ``TracedRun._replay`` (``ttnn.copy`` is per-device on the
        mesh, so batch-sharded ids and the replicated cur_pos copy correctly)."""
        self._ensure_rope(position + 1)
        tt_cos, tt_sin = self._rope_slice_dp(position, position + 1)
        tt_ids = self._upload_ids_dp([[t] for t in prev_tokens])   # [B,1] batch-sharded
        cur_pos = self._upload_cur_pos_dp(position)                # [1] replicated
        with self._traced_dispatch():
            tok = self.graph_decode(tt_ids, tt_cos, tt_sin, cur_pos, past_key_values=self.cache)
        return self._gather_tokens(tok)

    def generate_dp(self, input_ids_batch, seq_mask: List[bool], px_nhwc_batch,
                    max_new_tokens: int = 32, stop_on_eos: bool = True) -> List[List[int]]:
        """Full DP greedy generation: one concurrent prefill + concurrent decode loop.

        Returns one token-id list per stream (prompt excluded). All B streams step in
        lockstep (same decode depth); each stops APPENDING on EOS unless disabled."""
        B = len(input_ids_batch)
        prompt_len = len(input_ids_batch[0])
        firsts = self.prefill_dp(input_ids_batch, seq_mask, px_nhwc_batch)
        generated: List[List[int]] = [[t] for t in firsts]
        active = [not (stop_on_eos and t == self.eos_id) for t in firsts]
        currents = list(firsts)
        t0 = time.perf_counter()
        n_dec = 0
        for i in range(max_new_tokens - 1):
            if stop_on_eos and not any(active):
                break
            nxts = self.decode_step_dp(currents, prompt_len + i)
            n_dec += 1
            for b in range(B):
                if stop_on_eos and not active[b]:
                    continue
                if nxts[b] == self.eos_id and stop_on_eos:
                    active[b] = False
                    continue
                generated[b].append(nxts[b])
            currents = list(nxts)
        ttnn.synchronize_device(self._device)
        self.last_decode_s = time.perf_counter() - t0
        self.last_num_decode = n_dec
        return generated

    def warmup_dp(self, input_ids_batch, seq_mask: List[bool], px_nhwc_batch) -> None:
        """Prime JIT + capture the DP decode trace on the (N,1) mesh (trace double-run).

        The DP mirror of the single-device ``warmup``: prefill runs eagerly (creates no
        traced cold-compiles), so two short DP generates warm the kernels then -- after
        ``TracedRun.release_all`` -- capture+replay the batch-B ``graph_decode`` trace on
        the mesh. ``px_nhwc_batch`` is a HOST torch tensor re-uploaded per prefill
        (``_upload_px_dp`` does a fresh ``from_torch``), so the same host batch is safely
        reused across both warm-up generates. Call once before the timed ``generate_dp``."""
        self.generate_dp(input_ids_batch, seq_mask, px_nhwc_batch, max_new_tokens=2,
                         stop_on_eos=False)
        TracedRun.release_all()
        self.generate_dp(input_ids_batch, seq_mask, px_nhwc_batch, max_new_tokens=4,
                         stop_on_eos=False)
        self.cache.reset()

    def warmup(self, input_ids: List[int], seq_mask: Optional[List[bool]] = None,
               tt_px=None) -> None:
        """Prime JIT + capture the decode trace (trace double-run). Prefill stays eager,
        so it creates no traced cold-compiles: two short generates warm the kernels,
        then (after ``release_all``) the second capture+replays the decode graph."""
        self.generate(input_ids, seq_mask=seq_mask, tt_px=tt_px, max_new_tokens=2,
                      stop_on_eos=False)
        TracedRun.release_all()
        self.generate(input_ids, seq_mask=seq_mask, tt_px=tt_px, max_new_tokens=4,
                      stop_on_eos=False)
        self.cache.reset()

    def release(self) -> None:
        TracedRun.release_all()
        self.cache.reset()


__all__ = [
    "TT_METAL_COMMIT",
    "TTNNUnlimitedOcrPipeline",
    "TTNNUnlimitedOcrDecodeGraph",
    "TTNNUnlimitedOcrPrefillGraph",
]
