# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# This file was assembled during the Phase 2 mechanical migration
# (see scripts/merge_model_files.py). It concatenates the following
# original sources in order, deduplicating top-of-file imports:
#   - models/experimental/tt_symbiote/models/bailing_moe_v2.py
#   - models/experimental/tt_symbiote/modules/decoder_layer.py

from typing import List, Optional

import torch
import ttnn
from torch import nn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import MoeModelOutputWithPast

from tt_symbiote.core.module import TTNNModule
from tt_symbiote.core.run_config import trace_enabled
from tt_symbiote.models.auto.auto_mappings import register_recipe
from tt_symbiote.modules.ttnn_attention import PagedAttentionConfig, TTNNBailingMoEAttention, TTNNPagedAttentionKVCache
from tt_symbiote.modules.ttnn_embedding import TTNNBailingPaddedEmbedding, TTNNBailingRotaryEmbedding
from tt_symbiote.modules.ttnn_linear import TTNNLinearIColShardedWRowSharded
from tt_symbiote.modules.ttnn_moe import TTNNBailingMoE
from tt_symbiote.modules.ttnn_normalization import TTNNDistributedRMSNorm
from tt_symbiote.utils.module_replacement import register_modules

# === content from models/experimental/tt_symbiote/models/bailing_moe_v2.py ===
"""TTNN BailingMoeV2 Model implementation."""


class MoeV2ModelOutputWithPast(MoeModelOutputWithPast):
    def __init__(self, mtp_hidden_states=None, **kwargs):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states


class TTNNBailingMoeV2Model(TTNNModule):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`BailingMoeV2DecoderLayer`]

    Args:
        config: BailingMoeV2Config
    """

    @classmethod
    def from_torch(cls, hf_model):
        """Wrap a HF ``BailingMoeV2Model`` and convert its subtree to TTNN.

        Phase 5 / Option 1 contract: each TTNN wrapper class owns the
        conversion of its own subtree. Previously the test harness ran
        :func:`register_module_replacement_dict` three separate times
        (decoder/norm/embed/rotary, then nn.Linear/nn.SiLU at the root,
        then this outer wrapper). With Option 1 the wrapper itself
        rewrites its children before adopting the HF model, so the recipe
        can describe the swap as a single dict.
        """
        register_modules(
            hf_model,
            {
                type(hf_model.layers[0]): TTNNBailingMoEDecoderLayerPadded,
                type(hf_model.norm): TTNNDistributedRMSNorm,
                nn.Embedding: TTNNBailingPaddedEmbedding,
                type(hf_model.rotary_emb): TTNNBailingRotaryEmbedding,
            },
            model_config=None,
        )

        new_model = cls()
        new_model._fallback_torch_layer = hf_model
        new_model.model = hf_model

        # Bypass tensor wrapping/unwrapping for decoder layers.
        # These sit under the HF BailingMoeV2Model (nn.Module), so
        # set_device() would give them _bypass_tensor_wrapping=False.
        # Bypassing is safe: no PyTorch ops touch hidden_states between
        # layer calls, and each layer's forward already works with raw
        # ttnn.Tensor objects.
        for layer in hf_model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        if isinstance(hf_model.norm, TTNNModule):
            hf_model.norm._bypass_tensor_wrapping = True

        return new_model

    def call(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        ttnn_object = self
        self = self.model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = list(input_ids.shape)[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = list(inputs_embeds.shape)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            input_ids = ttnn.from_torch(
                input_ids.cpu().to(torch.int32),
                device=ttnn_object.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
            )
            inputs_embeds = self.word_embeddings(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = ttnn.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])
            position_ids = ttnn.unsqueeze(position_ids, 0)
        else:
            position_ids = ttnn.from_torch(
                position_ids.cpu().to(torch.int32),
                device=ttnn_object.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_object.device),
            )

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
            )

        # Pre-convert attention_mask to ttnn.Tensor for bypass-enabled decoder layers.
        # With _bypass_tensor_wrapping=True, fast_unwrap_to_device passes torch.Tensor
        # unchanged, but TTNNSDPAAttention needs ttnn.Tensor for on-device SDPA.
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            mesh_mapper = (
                ttnn.ReplicateTensorToMesh(ttnn_object.device) if ttnn_object.device.get_num_devices() > 1 else None
            )
            attention_mask = ttnn.from_torch(
                attention_mask,
                device=ttnn_object.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        for layer_idx, decoder_layer in enumerate(layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            # Update KV cache Python-side counters OUTSIDE the trace boundary.
            # During trace replay, execute_trace only replays device ops;
            # _seq_lengths increments inside paged_update_on_device / paged_fill_on_device
            # do NOT execute. By updating here, the counters advance correctly
            # in all phases (warmup, capture, replay).
            if past_key_values is not None and hasattr(past_key_values, "update_seq_length"):
                seq_len = inputs_embeds.shape[1]  # prefill: SEQ_LEN, decode: 1
                past_key_values.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        main_hidden_states = hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (main_hidden_states,)

        mtp_hidden_states = None

        if mtp_layers:
            for decoder_layer in mtp_layers:
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
                inputs_embeds = self.word_embeddings(input_ids)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        inputs_embeds,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        output_router_logits,
                        use_cache,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                if mtp_hidden_states is None:
                    mtp_hidden_states = []
                hidden_states = layer_outputs[0]
                mtp_hidden_states.append(hidden_states)

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [main_hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeV2ModelOutputWithPast(
            last_hidden_state=main_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            mtp_hidden_states=mtp_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


# === content from models/experimental/tt_symbiote/modules/decoder_layer.py ===
"""TTNN Decoder Layer for BailingMoeV2 (Ling-mini-2.0).

Replaces BailingMoeV2DecoderLayer to perform residual adds on-device using ttnn.add,
eliminating host round-trips that force device synchronization.
"""


@trace_enabled
class TTNNBailingMoEDecoderLayer(TTNNModule):
    """Replaces BailingMoeV2DecoderLayer to keep residual adds on-device.

    Eliminates 2 host round-trips per layer (one for attention residual,
    one for MoE/MLP residual) by using ttnn.add instead of aten::add.
    """

    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.attention = None
        self.mlp = None
        self._is_dense_layer = False

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from BailingMoeV2DecoderLayer.

        Args:
            torch_layer: HuggingFace BailingMoeV2DecoderLayer instance
        """
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer

        new_layer.input_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.attention = TTNNBailingMoEAttention.from_torch(torch_layer.attention)

        config = torch_layer.attention.config
        layer_idx = torch_layer.attention.layer_idx
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        is_dense = getattr(config, "num_experts", None) is None or layer_idx < first_k_dense
        new_layer._is_dense_layer = is_dense

        if is_dense:
            from tt_symbiote.modules.ttnn_moe import TTNNBailingMoeV2MLP

            new_layer.mlp = TTNNBailingMoeV2MLP.from_torch(torch_layer.mlp)
        else:
            new_layer.mlp = TTNNBailingMoE.from_torch(torch_layer.mlp)

        return new_layer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        position_embeddings=None,
        cache_position=None,
        **kwargs,
    ):
        hs = hidden_states

        # Ensure TILE layout and bfloat16 for TTNN ops
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        # Save residual (stays on device as TTNN tensor)
        residual = hs

        # Input layernorm
        hs = self.input_layernorm(hs)

        # Attention — use cache_position (explicit kwarg) if provided,
        # otherwise fall back to position_ids for backward compatibility.
        # Making cache_position a named parameter allows TracedRun to
        # pre-allocate a device buffer for it, so the paged-attention
        # decode path receives a device tensor and avoids host→device
        # writes during trace capture.
        attn_cache_position = cache_position if cache_position is not None else position_ids
        attn_out, self_attn_weights, present_key_value = self.attention(
            hidden_states=hs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=attn_cache_position,
        )

        # Residual add ON DEVICE (replaces aten::add on CPU)
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)
        # NOTE: Do NOT deallocate residual here — it is the pre-allocated trace
        # input buffer.  ttnn.deallocate inside a traced forward would be
        # replayed by execute_trace, freeing the buffer that
        # _copy_inputs_to_trace_buffer needs on the next replay iteration.

        # Save new residual
        residual = hs

        # Post-attention layernorm
        hs_normed = self.post_attention_layernorm(hs)

        # MLP / MoE
        # MLP (layer 0) or MoE (layers 1-19)
        mlp_out = self.mlp(hs_normed)
        router_logits = None
        if isinstance(mlp_out, tuple):
            mlp_out, router_logits = mlp_out

        # Residual add ON DEVICE (replaces aten::add on CPU)
        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        outputs = (hs,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# Re-exported for back-compat with code that imported this helper from
# the modeling module before Phase 5 moved it to ``utils/math_utils.py``
# (where it now lives to break the circular import between this file
# and ``integrations/ttnn_embedding.py``).
from tt_symbiote.utils.math_utils import next_power_of_2 as _next_power_of_2  # noqa: E402


class TTNNBailingMoEDecoderLayerPadded(TTNNModule):
    """Decoder layer that pads the input sequence length to the next power of 2.

    Padding to a power-of-2 sequence length reduces the number of unique trace
    cache keys during prefill, since many different prompt lengths map to the
    same padded length. This improves trace reuse across turns.

    The pad is applied before the forward pass and the output is sliced back
    to the original sequence length afterward.
    """

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from BailingMoeV2DecoderLayer.

        Args:
            torch_layer: HuggingFace BailingMoeV2DecoderLayer instance
        """
        new_layer = cls()
        new_layer.layer = TTNNBailingMoEDecoderLayer.from_torch(torch_layer)
        return new_layer

    @staticmethod
    def _pad_dim(tensor, dim, pad_amount, value=0.0):
        """Pad a single dimension of a tensor by ``pad_amount``."""
        rank = len(tensor.shape)
        padding = tuple((0, pad_amount if i == dim else 0) for i in range(rank))
        return ttnn.pad(tensor, padding=padding, value=value)

    @staticmethod
    def _slice_dim(tensor, dim, length):
        """Slice a tensor along ``dim`` to ``length``."""
        starts = [0] * len(tensor.shape)
        ends = list(tensor.shape)
        ends[dim] = length
        return ttnn.slice(tensor, starts, ends)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        position_embeddings=None,
        cache_position=None,
        **kwargs,
    ):
        rank = len(hidden_states.shape)
        seq_dim = rank - 2  # sequence length is always second-to-last
        seq_len = hidden_states.shape[seq_dim]
        padded_seq_len = _next_power_of_2(seq_len)
        pad_amount = padded_seq_len - seq_len

        if pad_amount > 0:
            hidden_states = self._pad_dim(hidden_states, seq_dim, pad_amount, value=0.0)

            # attention_mask: [..., seq_len, seq_len] — pad last two dims
            if attention_mask is not None:
                mask_rank = len(attention_mask.shape)
                attention_mask = self._pad_dim(attention_mask, mask_rank - 2, pad_amount, value=float("-inf"))
                attention_mask = self._pad_dim(attention_mask, mask_rank - 1, pad_amount, value=float("-inf"))

            # position_ids: [batch, seq_len] — pad seq dim with 0
            if position_ids is not None:
                pid_seq_dim = len(position_ids.shape) - 1
                position_ids = self._pad_dim(position_ids, pid_seq_dim, pad_amount, value=0)

            # position_embeddings (cos, sin): [batch, seq_len, head_dim]
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cos_seq_dim = len(cos.shape) - 2
                cos = self._pad_dim(cos, cos_seq_dim, pad_amount, value=0.0)
                sin = self._pad_dim(sin, cos_seq_dim, pad_amount, value=0.0)
                position_embeddings = (cos, sin)

        # Pass cache_position explicitly (not padded) so the inner layer's
        # trace infrastructure can pre-allocate a device buffer for it.
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            **kwargs,
        )

        if pad_amount > 0:
            hs = self._slice_dim(outputs[0], seq_dim, seq_len)
            if isinstance(outputs, tuple):
                outputs = (hs,) + outputs[1:]
            else:
                outputs = [hs] + list(outputs[1:])

        return outputs


# ---------------------------------------------------------------------------
# Recipe (Phase 5 / Option 1)
# ---------------------------------------------------------------------------
#
# Single-dict module-replacement recipe for ``BailingMoeV2ForCausalLM``.
# Per the Option 1 contract:
#
#   * ``build_module_dict`` returns one flat ``{torch_class: ttnn_class}``
#     dict that the auto factory hands to ``register_modules`` in a single
#     pass.
#   * Each TTNN wrapper class is responsible for converting its own
#     subtree in ``from_torch``. ``TTNNBailingMoeV2Model.from_torch`` (above)
#     internally swaps decoder layers, the final norm, ``nn.Embedding``,
#     and the rotary embedding, so the recipe only needs to describe the
#     two top-level swaps: the outer model wrapper and the bare
#     ``lm_head`` linear.
#   * ``post_register`` patches ``model.device`` to ``cpu`` so HF's
#     generation loop has a valid device for the token tensor it places
#     itself (after replacement, no ``nn.Module`` params remain and HF's
#     default implementation would call ``next(self.parameters())`` and
#     raise ``StopIteration``).
#   * ``make_kv_cache`` allocates the paged-attention KV cache; it is
#     invoked by ``set_device`` once a device is bound and the result is
#     attached as ``model._tt_kv_cache`` (resolves docs/internal/PROJECT_PROPOSAL.md Q9).
#
# Design-time coverage manifests (backfilled post-Phase 8 to keep
# ``tt_symbiote.compatibility.report`` informative for the reference
# port). Names match the HF upstream module file
# (``transformers_modules/inclusionAI/Ling.../modeling_bailing_moe_v2.py``)
# so the runtime hook in :mod:`tt_symbiote.core.run_config` can
# cross-reference observed classes against the recipe's declared intent.
#
# Membership rules (mirrors the Gemma-4 / Qwen3-VL recipes):
#   * ``tt_implemented``: this commit ships a TTNN wrapper that runs on
#     device. For Ling-mini-2.0 every compute-bearing class is wrapped
#     — either by the outer ``build_module_dict`` or recursively by
#     ``TTNNBailingMoeV2Model.from_torch`` /
#     ``TTNNBailingMoEDecoderLayer.from_torch``.
#   * ``cpu_fallback``: HF class is exercised by the demo but stays on
#     PyTorch. Ling-mini-2.0 is a *full* TTNN port — this list is
#     intentionally empty.
#   * ``host_glue``: HF class is intentionally host-only by policy
#     (orchestration / ``GenerationMixin`` shim / pure container with
#     no FLOPs). Glue has no compute to accelerate.
#   * ``out_of_scope``: HF class exists in the model file but is *not
#     touched* by the verified ``examples/e2e/run_ling_mini_2_0.py`` run
#     (alternate attention implementations, MTP head when
#     ``num_nextn_predict_layers == 0``, output dataclasses).


_TT_IMPLEMENTED: list[str] = [
    "BailingMoeV2RMSNorm",  # -> TTNNDistributedRMSNorm
    "BailingMoeV2RotaryEmbedding",  # -> TTNNBailingRotaryEmbedding
    "BailingMoeV2MLP",  # -> TTNNBailingMoeV2MLP (dense layer 0)
    "BailingMoeV2Gate",  # gate weight folded into TTNNBailingMoE
    "BailingMoeV2SparseMoeBlock",  # -> TTNNBailingMoE (layers 1..N-1)
    "BailingMoeV2SdpaAttention",  # -> TTNNBailingMoEAttention (SDPA is the demo default)
    "BailingMoeV2DecoderLayer",  # -> TTNNBailingMoEDecoderLayerPadded
    "BailingMoeV2Model",  # -> TTNNBailingMoeV2Model
]


# Ling-mini-2.0 is a full TTNN port. Every HF class on the demo path is
# either accelerated (``tt_implemented``) or pure orchestration with no
# FLOPs (``host_glue``). A clean run should leave
# ``compatibility.report(model)["regressions"]`` empty — any class
# showing up there is a real regression signal.
_CPU_FALLBACK: list[str] = []


_HOST_GLUE: list[str] = [
    # Abstract HF base class; provides config / generation mixins only.
    "BailingMoeV2PreTrainedModel",
    # Top-level causal-LM head. ``forward`` delegates to the swapped
    # ``BailingMoeV2Model`` and applies the (also-swapped) ``nn.Linear``
    # ``lm_head``; the rest is ``GenerationMixin`` orchestration.
    "BailingMoeV2ForCausalLM",
]


_OUT_OF_SCOPE: list[str] = [
    # ----- Output dataclasses (not torch modules) -----
    "MoEV2CausalLMOutputWithPast",
    "MoeV2ModelOutputWithPast",
    # ----- Alternate attention implementations not selected by
    # ----- the verified demo (which falls through to SDPA). The
    # ----- swap inside ``TTNNBailingMoEDecoderLayer.from_torch`` is
    # ----- attribute-based, so users who explicitly opt into
    # ----- ``attn_implementation="eager"`` or ``"flash_attention_2"``
    # ----- would still get the same TTNN wrapper; the demo path
    # ----- itself never instantiates these classes.
    "BailingMoeV2Attention",  # eager / manual SDPA
    "BailingMoeV2FlashAttention2",  # flash-attn-2 backend
    # ----- Multi-Token Prediction head — only constructed when
    # ----- ``config.num_nextn_predict_layers > 0``. Ling-mini-2.0
    # ----- ships with ``num_nextn_predict_layers=0`` so MTP layers
    # ----- are never built and the model.layers list contains only
    # ----- ``BailingMoeV2DecoderLayer`` instances.
    "BailingMoeV2MTPLayer",
]


@register_recipe(hf_class_name="BailingMoeV2ForCausalLM")
class BailingMoEV2Recipe:
    tt_implemented: list[str] = _TT_IMPLEMENTED
    cpu_fallback: list[str] = _CPU_FALLBACK
    host_glue: list[str] = _HOST_GLUE
    out_of_scope: list[str] = _OUT_OF_SCOPE

    def build_module_dict(self, model):
        return {
            type(model.model): TTNNBailingMoeV2Model,
            nn.Linear: TTNNLinearIColShardedWRowSharded,
        }

    def post_register(self, model):
        type(model).device = property(lambda self: torch.device("cpu"))

    def make_kv_cache(self, model, device, batch_size: int = 1, **kwargs):
        config = PagedAttentionConfig(
            block_size=kwargs.get("block_size", 64),
            max_num_blocks=kwargs.get("max_num_blocks", 32),
            batch_size=batch_size,
        )
        return TTNNPagedAttentionKVCache(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            config=config,
            device=None,
        ).to_device(device)
