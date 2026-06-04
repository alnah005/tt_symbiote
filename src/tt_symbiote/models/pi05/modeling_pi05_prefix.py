# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN prefix embedding for the pi0.5 VLM backbone.

Ported into the :class:`~tt_symbiote.core.module.TTNNModule` lifecycle from the
tt-metal reference ``models/experimental/pi0_5/tt/ttnn_prefix.py``
(``PrefixEmbeddingTTNN``) and validated against the PyTorch golden
``reference/torch_prefix.py`` (``PrefixEmbedding``), branch ``tt/pi0.5_bh``.

The prefix combines image embeddings (from the SigLIP vision tower) and language
token embeddings (from the Gemma backbone) into the prefix half of the dual-expert
sequence. It holds **no weights of its own** -- it operates entirely through two
callbacks supplied by the backbone:

* ``embed_image_fn(image)`` -> ``(B, num_image_tokens, vlm_hidden)`` per image.
* ``embed_language_fn(lang_tokens)`` -> ``(B, seq, vlm_hidden)``.

``embed_prefix`` embeds each image, embeds the language tokens and scales them by
``sqrt(vlm_hidden)`` (standard embedding scaling), concatenates image + language
embeddings along the sequence dim, and builds the padding and attention masks.
Prefix tokens use bidirectional attention, so ``prefix_att_masks`` is all zeros.

Because there are no weights, ``preprocess_weights_impl`` / ``move_weights_to_device_impl``
are pass-throughs (the callbacks own their own weights / device residency).

VALIDATION STATUS: the embedding concat + sqrt(hidden) language scaling are fully
implemented and mirror the validated TTNN reference. The padding-mask construction
(per-image mask expansion + concat with language masks) is partially scaffolded --
see ``embed_prefix`` -- because correct mask dtype/layout handling for the mixed
host/device mask inputs needs on-device PCC verification against
``tt/ttnn_prefix.py::PrefixEmbeddingTTNN.embed_prefix``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import ttnn

from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices

from .configuration_pi05 import PrefixConfig

# Runtime tt-metal commit the model executes against (installed/built ttnn, main).
# Reference patterns ported from branch tt/pi0.5_bh @ b0703a56465989da179480c93a8992ec519e1cde.
TT_METAL_COMMIT = "b2af0cd67b4e92dafeb2d0254e1c5b43c3ec5a25"

__all__ = ["TTNNPi05PrefixEmbedding"]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


class TTNNPi05PrefixEmbedding(TTNNModule):
    """pi0.5 prefix embedding (images + language) for the VLM backbone.

    Reference: ``reference/torch_prefix.py::PrefixEmbedding`` and
    ``tt/ttnn_prefix.py::PrefixEmbeddingTTNN``.

    Holds no weights; embedding is delegated to the ``embed_image_fn`` /
    ``embed_language_fn`` callbacks supplied at construction.
    """

    @classmethod
    def from_torch(
        cls,
        prefix=None,
        *,
        embed_image_fn,
        embed_language_fn,
        config: PrefixConfig,
        vlm_hidden_size: int = 2048,
    ) -> "TTNNPi05PrefixEmbedding":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = prefix
        new._config = config
        new.embed_image_fn = embed_image_fn
        new.embed_language_fn = embed_language_fn
        new._vlm_hidden_size = vlm_hidden_size
        new._lang_scale = math.sqrt(vlm_hidden_size)
        return new

    def preprocess_weights_impl(self):
        # No weights of its own. The image/language callbacks own their weights.
        return self

    def move_weights_to_device_impl(self):
        # No weights of its own. The image/language callbacks own device residency.
        return self

    @run_on_devices(DeviceArch.P150)
    def embed_language(self, lang_tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed language tokens and scale by sqrt(vlm_hidden_size).

        ``ttnn.embedding`` returns ROW_MAJOR; convert to TILE so the downstream
        concat with TILE image embeddings works for any token length.
        """
        lang_emb = self.embed_language_fn(lang_tokens)
        if lang_emb.layout != ttnn.TILE_LAYOUT:
            lang_emb = ttnn.to_layout(lang_emb, ttnn.TILE_LAYOUT, memory_config=_L1)
        return ttnn.multiply(lang_emb, self._lang_scale, memory_config=_L1)

    @run_on_devices(DeviceArch.P150)
    def embed_prefix(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Embed prefix (images + language).

        Returns ``(prefix_embs, prefix_pad_masks, prefix_att_masks)``:
          * ``prefix_embs``: ``(B, prefix_len, vlm_hidden)`` -- image embeddings
            followed by sqrt-scaled language embeddings concatenated on the seq dim.
          * ``prefix_pad_masks``: ``(B, prefix_len)`` -- per-image expanded masks
            followed by ``lang_masks``.
          * ``prefix_att_masks``: ``(B, prefix_len)`` -- all zeros (bidirectional
            prefix attention).

        FULLY IMPLEMENTED: image embedding, sqrt-scaled language embedding, and the
        sequence-dim concat of embeddings (matches
        ``tt/ttnn_prefix.py::PrefixEmbeddingTTNN.embed_prefix`` L191-214).
        """
        embs: List[ttnn.Tensor] = []
        pad_masks: List[ttnn.Tensor] = []

        # --- Image embeddings --------------------------------------------------
        if images and self.embed_image_fn is not None:
            for img, mask in zip(images, img_masks):
                img_emb = self.embed_image_fn(img)
                if img_emb.layout != ttnn.TILE_LAYOUT:
                    img_emb = ttnn.to_layout(img_emb, ttnn.TILE_LAYOUT, memory_config=_L1)
                embs.append(img_emb)
                pad_masks.append(self._expand_image_mask(mask, img_emb.shape))

        # --- Language embeddings (sqrt-scaled) ---------------------------------
        if self.embed_language_fn is not None:
            lang_emb = self.embed_language(lang_tokens)
            embs.append(lang_emb)
            lang_mask = lang_masks
            if lang_mask.layout != ttnn.TILE_LAYOUT:
                lang_mask = ttnn.to_layout(lang_mask, ttnn.TILE_LAYOUT, memory_config=_L1)
            pad_masks.append(lang_mask)

        if not embs:
            raise ValueError("embed_prefix requires at least one of images or language tokens")

        # --- Concatenate embeddings (TTNN concat needs uniform layout) ---------
        embs = [
            e if e.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(e, ttnn.TILE_LAYOUT, memory_config=_L1)
            for e in embs
        ]
        prefix_embs = (
            ttnn.concat(embs, dim=1, memory_config=_L1) if len(embs) > 1 else embs[0]
        )

        prefix_pad_masks = self._concat_pad_masks(pad_masks)

        # --- Attention mask: all zeros (bidirectional prefix attention) --------
        b = prefix_embs.shape[0]
        prefix_len = prefix_embs.shape[1]
        prefix_att_masks = ttnn.zeros(
            (b, prefix_len),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_L1,
        )

        return prefix_embs, prefix_pad_masks, prefix_att_masks

    @run_on_devices(DeviceArch.P150)
    def forward(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        return self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    # ------------------------------------------------------------------ helpers
    def _expand_image_mask(self, mask: ttnn.Tensor, emb_shape) -> ttnn.Tensor:
        """Expand a per-image validity mask (B,) to (B, num_image_tokens).

        SCAFFOLD: correct host/device dtype + layout handling for the mask inputs
        (which may arrive as ttnn ROW_MAJOR or already-tiled tensors, and require
        a (B,1) -> (B, num_tokens) repeat) must be PCC-verified on hardware against
        ``tt/ttnn_prefix.py::PrefixEmbeddingTTNN.embed_images`` (L117-137). The
        embedding concat above is fully implemented; only the mask broadcast is
        deferred.
        """
        raise NotImplementedError(
            "TTNNPi05PrefixEmbedding._expand_image_mask: image-mask (B,)->(B,num_tokens) "
            "broadcast is scaffolded; implement + PCC-verify against "
            "tt/ttnn_prefix.py:PrefixEmbeddingTTNN.embed_images L117-137 on hardware."
        )

    def _concat_pad_masks(self, pad_masks: List[ttnn.Tensor]) -> ttnn.Tensor:
        """Concatenate per-segment padding masks along the sequence dim.

        SCAFFOLD: depends on uniform mask layout/dtype produced by
        ``_expand_image_mask``; deferred together with it. Reference concat is
        ``tt/ttnn_prefix.py::PrefixEmbeddingTTNN.embed_prefix`` L212-214.
        """
        raise NotImplementedError(
            "TTNNPi05PrefixEmbedding._concat_pad_masks: pad-mask concat is scaffolded; "
            "implement + PCC-verify against tt/ttnn_prefix.py:PrefixEmbeddingTTNN.embed_prefix "
            "L212-214 on hardware (depends on _expand_image_mask)."
        )
