# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN SigLIP vision tower for the pi0.5 dual-expert backbone.

Ported into the :class:`~tt_symbiote.core.module.TTNNModule` lifecycle from the
tt-metal reference ``models/experimental/pi0_5/tt/ttnn_siglip.py`` and validated
against the PyTorch golden ``reference/torch_siglip.py`` (branch ``tt/pi0.5_bh``).

Classes
-------
* :class:`TTNNPi05SigLIPPatchEmbedding` -- conv2d-as-linear patch extraction via
  the device-side 6D-permute unfold (channel-last im2col) + ``ttnn.linear``.
* :class:`TTNNPi05SigLIPAttention` -- standard MHA (16 heads, head_dim 72 padded
  to 96 for tile alignment) with fused QKV, SDPA and native head split/concat.
* :class:`TTNNPi05SigLIPMLP` -- ``fc2(gelu_tanh(fc1(x)))``.
* :class:`TTNNPi05SigLIPBlock` -- pre-LN encoder block (LayerNorm, NOT RMSNorm).
* :class:`TTNNPi05SigLIPVisionTower` -- patch-embed + pos-embed + 27 blocks + post-LN.
* :class:`TTNNPi05MultiModalProjector` -- single linear (1152 -> 2048).

Conventions
-----------
The golden ``reference/torch_siglip.py`` classes are *plain Python objects* (not
``nn.Module``) holding raw weight tensors as attributes, so ``from_torch`` plucks
tensors directly:

* ``PatchEmbedding``        -> ``.conv_weight`` [out, in_c, kH, kW], ``.conv_bias``
* ``SigLIPAttention``       -> ``.q_proj`` / ``.k_proj`` / ``.v_proj`` / ``.out_proj``
                               [out, in] weights + optional ``.q_bias`` / ...
                               / ``.out_bias``; ``.num_heads`` / ``.head_dim`` / ``.scale``
* ``SigLIPMLP``             -> ``.fc1_weight`` / ``.fc1_bias`` / ``.fc2_weight`` / ``.fc2_bias``
* ``SigLIPBlock``           -> ``.ln1_weight`` / ``.ln1_bias`` / ``.ln2_weight`` / ``.ln2_bias``;
                               ``.attention`` / ``.mlp``
* ``SigLIPVisionTower``     -> ``.patch_embed`` / ``.position_embedding`` (tensor)
                               / ``.blocks`` (list) / ``.post_layernorm_weight`` / ``.post_layernorm_bias``
* ``MultiModalProjector``   -> ``.weight`` [out, in] / ``.bias``

``ttnn.linear`` computes ``x @ w`` (w is ``[in, out]``); torch weights are
``[out, in]``, so linear weights are transposed on host before upload.

This is the functionality-first (interleaved, NON block-sharded) baseline. The
PI0_SIGLIP_BS block-sharded encoder data path from the reference is deliberately
NOT ported here -- that is for the later op-sweep / config-optimize stage.

VALIDATION STATUS: attention (head-dim padding + fused-QKV split + SDPA + concat)
and the unfold patch-embedding require on-device PCC validation against the torch
reference (Tier 1/2 of the pcc-test-gen stage). The MLP / LayerNorm / projection
paths are low-risk.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import ttnn

from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices
from tt_symbiote.core.run_config import trace_enabled

from .configuration_pi05 import SigLIPConfig

# Runtime tt-metal commit the model executes against (installed/built ttnn, main).
# Reference patterns ported from branch tt/pi0.5_bh.
TT_METAL_COMMIT = "b2af0cd67b4e92dafeb2d0254e1c5b43c3ec5a25"

__all__ = [
    "TTNNPi05SigLIPPatchEmbedding",
    "TTNNPi05SigLIPAttention",
    "TTNNPi05SigLIPMLP",
    "TTNNPi05SigLIPBlock",
    "TTNNPi05SigLIPVisionTower",
    "TTNNPi05MultiModalProjector",
]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


# ---------------------------------------------------------------------------
# Weight upload helpers (host-side; allowed to use torch).
# ---------------------------------------------------------------------------
def _nearest_32(x: int) -> int:
    """Round up to nearest multiple of 32 for TTNN tile alignment."""
    return ((x + 31) // 32) * 32


def _linear_weight_to_tt(
    w: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16
) -> ttnn.Tensor:
    """Transpose a torch ``[out, in]`` linear weight to ttnn ``[in, out]`` host tensor."""
    return ttnn.from_torch(w.t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _bias_to_tt(b: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
    """1D bias ``[dim]`` -> ttnn host tensor shape ``[1, dim]`` for fused-bias linear."""
    return ttnn.from_torch(b.reshape(1, -1).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _norm_weight_to_tt(w: torch.Tensor) -> ttnn.Tensor:
    """LayerNorm gamma/beta ``[dim]`` -> ttnn host tensor shape ``[1, dim]``, bf16 TILE."""
    return ttnn.from_torch(w.reshape(1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _to_dev(t: Optional[ttnn.Tensor], device) -> Optional[ttnn.Tensor]:
    if t is None:
        return None
    return ttnn.to_device(t, device, memory_config=_DRAM)


# ---------------------------------------------------------------------------
# Patch Embedding (conv2d-as-linear via device unfold)
# ---------------------------------------------------------------------------
class TTNNPi05SigLIPPatchEmbedding(TTNNModule):
    """Patch embedding: device-side 6D-permute unfold + ``ttnn.linear``.

    Reconstructs the conv2d (kernel = stride = patch_size, non-overlapping
    patches) as a linear over flattened channel-last patch features. The conv
    weight ``[out, in_c, kH, kW]`` is permuted to ``[out, kH, kW, in_c]`` and
    flattened to ``[out, kH*kW*in_c]`` so it matches the channel-last (h, w, c)
    ordering produced by the unfold, then transposed to ``[in_feat, out]`` for
    ``ttnn.linear``. ``in_features`` (588 = 14*14*3) is padded to a tile multiple
    (608); the matching activation pad keeps the contraction dim tile-aligned.

    Reference: ``tt/ttnn_siglip.py::PatchEmbeddingTTNN`` (L163-334), torch golden
    ``reference/torch_siglip.py::PatchEmbedding`` (L31-84).
    """

    @classmethod
    def from_torch(cls, patch_embed, config: SigLIPConfig) -> "TTNNPi05SigLIPPatchEmbedding":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = patch_embed
        new._config = config
        new.patch_size = config.patch_size
        new.hidden_size = config.hidden_size
        # Plain-object reference holds raw conv tensors.
        new._conv_weight = patch_embed.conv_weight  # [out, in_c, kH, kW]
        new._conv_bias = getattr(patch_embed, "conv_bias", None)
        return new

    def preprocess_weights_impl(self):
        w = self._conv_weight  # [hidden, in_c, kH, kW]
        out_channels = w.shape[0]
        in_c = w.shape[1]
        in_features = in_c * w.shape[2] * w.shape[3]  # 3 * 14 * 14 = 588
        self.in_features = in_features
        self.in_features_padded = _nearest_32(in_features)  # 588 -> 608

        # Reorder to channel-last (out, h, w, c) to match the unfold output order,
        # flatten, then transpose to [in_features, out] for ttnn.linear.
        lin_w = w.permute(0, 2, 3, 1).contiguous().view(out_channels, -1)  # [out, h*w*c]
        lin_w = lin_w.t().contiguous()  # [in_features, out]
        pad_len = self.in_features_padded - in_features
        if pad_len > 0:
            lin_w = torch.nn.functional.pad(lin_w, (0, 0, 0, pad_len))  # pad rows (in dim)
        self.tt_weight = ttnn.from_torch(lin_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        self.tt_bias = _bias_to_tt(self._conv_bias) if self._conv_bias is not None else None

    def move_weights_to_device_impl(self):
        self.tt_weight = _to_dev(self.tt_weight, self.device)
        self.tt_bias = _to_dev(self.tt_bias, self.device)

    @run_on_devices(DeviceArch.P150)
    def forward(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        # Accept channels-first (B, C, H, W) and convert to channel-last (B, H, W, C).
        x = pixel_values
        if int(x.shape[1]) == 3 and int(x.shape[-1]) != 3:
            x = ttnn.permute(x, (0, 2, 3, 1))

        b = int(x.shape[0])
        img_h = int(x.shape[1])
        img_w = int(x.shape[2])
        img_c = int(x.shape[3])
        ph = img_h // self.patch_size
        pw = img_w // self.patch_size

        # Unfold via 6D reshape + permute (MultiCoreTileInvariant pattern):
        # (B,H,W,C) -> (B,ph,ps,pw,ps,C) -> (B,ph,pw,ps,ps,C) -> (B,num_patches,ps*ps*C)
        x = ttnn.reshape(x, (b, ph, self.patch_size, pw, self.patch_size, img_c))
        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))
        x = ttnn.reshape(x, (b, ph * pw, self.patch_size * self.patch_size * img_c))

        # Pad the contraction (patch-feature) dim to tile-aligned (588 -> 608).
        cur = int(x.shape[-1])
        if cur < self.in_features_padded:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, self.in_features_padded - cur)], value=0.0)

        out = ttnn.linear(
            x,
            self.tt_weight,
            bias=self.tt_bias,
            dtype=ttnn.bfloat16,
            memory_config=_L1,
        )
        ttnn.deallocate(x)
        return out  # (B, num_patches, hidden_size)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (16 heads, head_dim 72 padded to 96)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05SigLIPAttention(TTNNModule):
    """Standard MHA: fused QKV -> head split -> SDPA -> head concat -> O-proj.

    head_dim 72 is not tile-aligned (32). Following the reference, Q/K/V/O
    weights (and biases) are padded on host so the per-head dim becomes 96
    (3 tiles); ``nlp_create_qkv_heads`` / SDPA / ``nlp_concat_heads`` then run on
    the padded head dim, and the O-proj input dim is correspondingly padded.
    The padding columns carry zeros so they contribute nothing to the output.

    Reference: ``tt/ttnn_siglip.py::SigLIPAttentionTTNN`` (L342-728), torch golden
    ``reference/torch_siglip.py::SigLIPAttention`` (L92-164).
    """

    @classmethod
    def from_torch(cls, attn, config: SigLIPConfig) -> "TTNNPi05SigLIPAttention":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = attn
        new._config = config
        new.num_heads = config.num_attention_heads
        new.head_dim = config.head_dim
        new.hidden_size = config.hidden_size
        new.padded_head_dim = _nearest_32(new.head_dim)  # 72 -> 96
        new.scale = 1.0 / math.sqrt(new.head_dim)
        new._q_w = attn.q_proj
        new._k_w = attn.k_proj
        new._v_w = attn.v_proj
        new._o_w = attn.out_proj
        new._q_b = getattr(attn, "q_bias", None)
        new._k_b = getattr(attn, "k_bias", None)
        new._v_b = getattr(attn, "v_bias", None)
        new._o_b = getattr(attn, "out_bias", None)
        return new

    # --- host-side head-dim padding helpers -------------------------------
    def _pad_proj_weight(self, w: torch.Tensor, heads_out: bool = True) -> torch.Tensor:
        """Pad the per-head dim of a torch ``[out, in]`` projection weight.

        ``heads_out=True``: the OUT dim is laid out as heads*head_dim (Q/K/V);
        pad each head's slice from head_dim to padded_head_dim.
        ``heads_out=False``: the IN dim is heads*head_dim (O-proj); pad the input.
        """
        pad = self.padded_head_dim - self.head_dim
        if pad <= 0:
            return w
        dim = w.shape[0]  # out dim for q/k/v, out (hidden) for o
        if heads_out:
            w = w.t()  # -> [in, out] so the trailing dim is heads*head_dim
            w = w.reshape(w.shape[0], self.num_heads, self.head_dim)
            w = torch.nn.functional.pad(w, (0, pad))
            w = w.reshape(w.shape[0], self.num_heads * self.padded_head_dim)
            w = w.t()
        else:
            # O-proj weight is [hidden_out, heads*head_dim]; pad the input dim.
            w = w.reshape(dim, self.num_heads, self.head_dim)
            w = torch.nn.functional.pad(w, (0, pad))
            w = w.reshape(dim, self.num_heads * self.padded_head_dim)
        return w.contiguous()

    def _pad_bias(self, b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if b is None:
            return None
        pad = self.padded_head_dim - self.head_dim
        if pad <= 0:
            return b
        b = b.view(self.num_heads, self.head_dim)
        b = torch.nn.functional.pad(b, (0, pad))
        return b.reshape(self.num_heads * self.padded_head_dim).contiguous()

    def preprocess_weights_impl(self):
        # Fuse Q/K/V on host along the (padded) output dim.
        wq = self._pad_proj_weight(self._q_w, heads_out=True).t().contiguous()  # [in, head_out]
        wk = self._pad_proj_weight(self._k_w, heads_out=True).t().contiguous()
        wv = self._pad_proj_weight(self._v_w, heads_out=True).t().contiguous()
        wqkv = torch.cat([wq, wk, wv], dim=-1)  # [in, 3*heads*padded_head_dim]
        self.tt_wqkv = ttnn.from_torch(wqkv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        bq = self._pad_bias(self._q_b)
        bk = self._pad_bias(self._k_b)
        bv = self._pad_bias(self._v_b)
        if bq is not None and bk is not None and bv is not None:
            self.tt_bqkv = _bias_to_tt(torch.cat([bq, bk, bv], dim=-1))
        else:
            self.tt_bqkv = None

        # O-proj: input dim padded (heads*padded_head_dim), output is hidden.
        wo = self._pad_proj_weight(self._o_w, heads_out=False)  # [hidden, heads*padded_head_dim]
        self.tt_o = ttnn.from_torch(wo.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bo = _bias_to_tt(self._o_b) if self._o_b is not None else None

    def move_weights_to_device_impl(self):
        self.tt_wqkv = _to_dev(self.tt_wqkv, self.device)
        self.tt_bqkv = _to_dev(self.tt_bqkv, self.device)
        self.tt_o = _to_dev(self.tt_o, self.device)
        self.tt_bo = _to_dev(self.tt_bo, self.device)

    @run_on_devices(DeviceArch.P150)
    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        from tt_symbiote.models.pi05.modeling_pi05_common import get_sdpa_compute_kernel_config

        b = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[-2])

        # Reshape to 4D [b, 1, seq, hidden] for nlp_create_qkv_heads.
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(hidden_states, (b, 1, seq_len, hidden_states.shape[-1]))

        qkv = ttnn.linear(
            hidden_states,
            self.tt_wqkv,
            bias=self.tt_bqkv,
            dtype=ttnn.bfloat16,
            memory_config=_L1,
        )
        # MHA: num_kv_heads == num_heads.
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=_L1,
        )
        ttnn.deallocate(qkv)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=get_sdpa_compute_kernel_config(),
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=_L1)
        out = ttnn.linear(
            attn_out,
            self.tt_o,
            bias=self.tt_bo,
            dtype=ttnn.bfloat16,
            memory_config=_L1,
        )
        ttnn.deallocate(attn_out)
        out = ttnn.reshape(out, (b, seq_len, self.hidden_size))
        return out


# ---------------------------------------------------------------------------
# MLP: fc2(gelu_tanh(fc1(x)))
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05SigLIPMLP(TTNNModule):
    """SigLIP MLP: ``fc2(gelu_tanh(fc1(x)))``.

    Reference: ``tt/ttnn_siglip.py::SigLIPMLPTTNN`` (L731-966), torch golden
    ``reference/torch_siglip.py::SigLIPMLP`` (L172-212). The big fc1/fc2 weights
    upload as bfloat8_b (per the reference dtype map).
    """

    @classmethod
    def from_torch(cls, mlp, config: Optional[SigLIPConfig] = None) -> "TTNNPi05SigLIPMLP":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = mlp
        new._config = config
        new._fc1_w = mlp.fc1_weight
        new._fc1_b = getattr(mlp, "fc1_bias", None)
        new._fc2_w = mlp.fc2_weight
        new._fc2_b = getattr(mlp, "fc2_bias", None)
        return new

    def preprocess_weights_impl(self):
        self.tt_fc1 = _linear_weight_to_tt(self._fc1_w, dtype=ttnn.bfloat8_b)
        self.tt_fc2 = _linear_weight_to_tt(self._fc2_w, dtype=ttnn.bfloat8_b)
        self.tt_fc1_b = _bias_to_tt(self._fc1_b) if self._fc1_b is not None else None
        self.tt_fc2_b = _bias_to_tt(self._fc2_b) if self._fc2_b is not None else None

    def move_weights_to_device_impl(self):
        self.tt_fc1 = _to_dev(self.tt_fc1, self.device)
        self.tt_fc2 = _to_dev(self.tt_fc2, self.device)
        self.tt_fc1_b = _to_dev(self.tt_fc1_b, self.device)
        self.tt_fc2_b = _to_dev(self.tt_fc2_b, self.device)
        # fc1 (256x1152x4320) is best on the default grid; fc2 (256x4320x1152,
        # large-K) wins ~3x on the full 2D grid (op-sweep validated, PCC>=0.9998).
        _g = self.device.compute_with_storage_grid_size()
        self._full_cg = ttnn.CoreGrid(y=_g.y, x=_g.x)

    @run_on_devices(DeviceArch.P150)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = ttnn.linear(x, self.tt_fc1, bias=self.tt_fc1_b, dtype=ttnn.bfloat16, memory_config=_L1)
        h = ttnn.gelu(h, fast_and_approximate_mode=True, memory_config=_L1)  # tanh-approx (matches reference; faster)
        out = ttnn.linear(h, self.tt_fc2, bias=self.tt_fc2_b, dtype=ttnn.bfloat16, memory_config=_L1, core_grid=self._full_cg)
        ttnn.deallocate(h)
        return out


# ---------------------------------------------------------------------------
# Pre-LN encoder block
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05SigLIPBlock(TTNNModule):
    """Pre-LN encoder block: LN1 -> attn -> +res -> LN2 -> mlp -> +res.

    LayerNorm (NOT RMSNorm). Reference: ``tt/ttnn_siglip.py::SigLIPBlockTTNN``
    (L969-1198), torch golden ``reference/torch_siglip.py::SigLIPBlock`` (L220-288).
    """

    @classmethod
    def from_torch(cls, block, config: SigLIPConfig) -> "TTNNPi05SigLIPBlock":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = block
        new._config = config
        new._eps = config.layer_norm_eps
        new._ln1_w = block.ln1_weight
        new._ln1_b = getattr(block, "ln1_bias", None)
        new._ln2_w = block.ln2_weight
        new._ln2_b = getattr(block, "ln2_bias", None)
        new.attention = TTNNPi05SigLIPAttention.from_torch(block.attention, config)
        new.mlp = TTNNPi05SigLIPMLP.from_torch(block.mlp, config)
        return new

    def preprocess_weights_impl(self):
        self.tt_ln1_w = _norm_weight_to_tt(self._ln1_w)
        self.tt_ln1_b = _norm_weight_to_tt(self._ln1_b) if self._ln1_b is not None else None
        self.tt_ln2_w = _norm_weight_to_tt(self._ln2_w)
        self.tt_ln2_b = _norm_weight_to_tt(self._ln2_b) if self._ln2_b is not None else None
        self.attention.preprocess_weights()
        self.mlp.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.tt_ln1_w = _to_dev(self.tt_ln1_w, self.device)
        self.tt_ln1_b = _to_dev(self.tt_ln1_b, self.device)
        self.tt_ln2_w = _to_dev(self.tt_ln2_w, self.device)
        self.tt_ln2_b = _to_dev(self.tt_ln2_b, self.device)
        self.attention.move_weights_to_device()
        self.mlp.move_weights_to_device()

    @run_on_devices(DeviceArch.P150)
    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.tt_ln1_w,
            bias=self.tt_ln1_b,
            epsilon=self._eps,
            memory_config=_L1,
        )
        attn_out = self.attention(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=_L1)
        ttnn.deallocate(attn_out)

        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.tt_ln2_w,
            bias=self.tt_ln2_b,
            epsilon=self._eps,
            memory_config=_L1,
        )
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_out, memory_config=_L1)
        ttnn.deallocate(mlp_out)
        return hidden_states


# ---------------------------------------------------------------------------
# Full vision tower
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05SigLIPVisionTower(TTNNModule):
    """SigLIP vision tower: patch-embed + pos-embed + 27 blocks + post-LN.

    ``@trace_enabled`` at the TOWER level: the 27 blocks each run once per
    forward (so per-block auto-trace never captures), but the tower is invoked
    repeatedly (once per inference), so capturing the whole 27-block graph as a
    single trace at this level is what makes the encoder traceable. Child blocks
    run normally inside the tower's capture (``_TRACE_RUNNING`` guard).

    Reference: ``tt/ttnn_siglip.py::SigLIPVisionTowerTTNN`` (L1206-1452), torch
    golden ``reference/torch_siglip.py::SigLIPVisionTower`` (L296-410).

    Position embedding: for the canonical pi0.5 config (224/14 -> 256 patches)
    the learned position embedding already matches num_patches, so no bicubic
    interpolation is needed (the reference only interpolates when checkpoint
    resolution differs). The position table is uploaded as ``[1, num_patches,
    hidden]`` and added (broadcasting over batch).
    """

    @classmethod
    def from_torch(cls, vision_tower, config: SigLIPConfig) -> "TTNNPi05SigLIPVisionTower":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = vision_tower
        new._config = config
        new._eps = config.layer_norm_eps
        new.hidden_size = config.hidden_size
        new._num_patches = config.num_patches
        new.patch_embed = TTNNPi05SigLIPPatchEmbedding.from_torch(vision_tower.patch_embed, config)
        new._position_embedding = vision_tower.position_embedding  # tensor [num_pos, hidden] or None
        new.blocks: List[TTNNPi05SigLIPBlock] = [
            TTNNPi05SigLIPBlock.from_torch(block, config) for block in vision_tower.blocks
        ]
        new._post_ln_w = getattr(vision_tower, "post_layernorm_weight", None)
        new._post_ln_b = getattr(vision_tower, "post_layernorm_bias", None)
        return new

    def preprocess_weights_impl(self):
        if self._position_embedding is not None:
            pos = self._position_embedding
            if pos.shape[0] != self._num_patches:
                raise NotImplementedError(
                    "Position-embedding bicubic interpolation (num_pos != num_patches) is "
                    "not ported; ref tt/ttnn_siglip.py:L1242-L1280 (fallback_ops.interpolate). "
                    f"Got num_pos={pos.shape[0]} vs num_patches={self._num_patches}."
                )
            # Upload as [1, num_patches, hidden] so ttnn.add broadcasts over batch.
            self.tt_pos_emb = ttnn.from_torch(
                pos.reshape(1, pos.shape[0], pos.shape[1]).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            self.tt_pos_emb = None

        if self._post_ln_w is not None:
            self.tt_post_ln_w = _norm_weight_to_tt(self._post_ln_w)
            self.tt_post_ln_b = _norm_weight_to_tt(self._post_ln_b) if self._post_ln_b is not None else None
        else:
            self.tt_post_ln_w = None
            self.tt_post_ln_b = None

        self.patch_embed.preprocess_weights()
        for block in self.blocks:
            block.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.tt_pos_emb = _to_dev(self.tt_pos_emb, self.device)
        self.tt_post_ln_w = _to_dev(self.tt_post_ln_w, self.device)
        self.tt_post_ln_b = _to_dev(self.tt_post_ln_b, self.device)
        self.patch_embed.move_weights_to_device()
        for block in self.blocks:
            block.move_weights_to_device()

    @run_on_devices(DeviceArch.P150)
    def forward(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = self.patch_embed(pixel_values)

        if self.tt_pos_emb is not None:
            hidden_states = ttnn.add(hidden_states, self.tt_pos_emb, memory_config=_L1)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        if self.tt_post_ln_w is not None:
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.tt_post_ln_w,
                bias=self.tt_post_ln_b,
                epsilon=self._eps,
                memory_config=_L1,
            )
        return hidden_states  # (B, num_patches, hidden_size)


# ---------------------------------------------------------------------------
# Multi-modal projector (1152 -> 2048)
# ---------------------------------------------------------------------------
@trace_enabled
class TTNNPi05MultiModalProjector(TTNNModule):
    """Single linear projecting vision features to the VLM hidden size.

    Reference: ``tt/ttnn_siglip.py::MultiModalProjectorTTNN`` (L1460-1524), torch
    golden ``reference/torch_siglip.py::MultiModalProjector`` (L418-449).
    """

    @classmethod
    def from_torch(cls, projector) -> "TTNNPi05MultiModalProjector":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = projector
        new._w = projector.weight  # [out, in] = [2048, 1152]
        new._b = getattr(projector, "bias", None)
        return new

    def preprocess_weights_impl(self):
        self.tt_weight = _linear_weight_to_tt(self._w, dtype=ttnn.bfloat8_b)
        self.tt_bias = _bias_to_tt(self._b) if self._b is not None else None

    def move_weights_to_device_impl(self):
        self.tt_weight = _to_dev(self.tt_weight, self.device)
        self.tt_bias = _to_dev(self.tt_bias, self.device)

    @run_on_devices(DeviceArch.P150)
    def forward(self, vision_features: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            vision_features,
            self.tt_weight,
            bias=self.tt_bias,
            dtype=ttnn.bfloat16,
            memory_config=_L1,
        )
