# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 (composite) PCC tests for the pi0.5 TTNN port.

Validates multi-layer composites against the pi0_5 PyTorch golden on identical
random weights -- runnable on a single Blackhole (P150) WITHOUT the gated
``pi05_base`` checkpoint. The full SigLIP vision tower exercises patch-embed
(conv-as-unfold) + position embedding + 27 transformer blocks + post-LN end to
end. 27 layers of bf16 accumulation compound drift, so the tower threshold is
looser than the per-block 0.99 (the reference E2E PCC is ~0.991).
"""

from __future__ import annotations

import torch

import ttnn
from tt_symbiote.models.pi05.configuration_pi05 import SigLIPConfig
from tt_symbiote.utils.device_management import set_device

from ..pi05_helpers import assert_pcc
from ..pi05_helpers import SEED, require_reference


def _siglip_tower_weights(cfg: SigLIPConfig) -> dict:
    h, inter = cfg.hidden_size, cfg.intermediate_size
    # Use the checkpoint ("vision_model.*") key form: the reference uses a
    # `weights.get(a) or weights.get(b)` idiom that raises on a multi-element
    # tensor when the first key hits, so the first form must be absent.
    w = {
        "vision_model.embeddings.patch_embedding.weight": torch.randn(
            h, cfg.num_channels, cfg.patch_size, cfg.patch_size
        )
        * 0.02,
        "vision_model.embeddings.patch_embedding.bias": torch.randn(h) * 0.02,
        "vision_model.embeddings.position_embedding.weight": torch.randn(cfg.num_patches, h) * 0.02,
        "vision_model.post_layernorm.weight": torch.randn(h) * 0.02 + 1.0,
        "vision_model.post_layernorm.bias": torch.randn(h) * 0.02,
    }
    for i in range(cfg.num_hidden_layers):
        p = f"vision_model.encoder.layers.{i}."
        w.update(
            {
                p + "layer_norm1.weight": torch.randn(h) * 0.02 + 1.0,
                p + "layer_norm1.bias": torch.randn(h) * 0.02,
                p + "layer_norm2.weight": torch.randn(h) * 0.02 + 1.0,
                p + "layer_norm2.bias": torch.randn(h) * 0.02,
                p + "self_attn.q_proj.weight": torch.randn(h, h) * 0.02,
                p + "self_attn.q_proj.bias": torch.randn(h) * 0.02,
                p + "self_attn.k_proj.weight": torch.randn(h, h) * 0.02,
                p + "self_attn.k_proj.bias": torch.randn(h) * 0.02,
                p + "self_attn.v_proj.weight": torch.randn(h, h) * 0.02,
                p + "self_attn.v_proj.bias": torch.randn(h) * 0.02,
                p + "self_attn.out_proj.weight": torch.randn(h, h) * 0.02,
                p + "self_attn.out_proj.bias": torch.randn(h) * 0.02,
                p + "mlp.fc1.weight": torch.randn(inter, h) * 0.02,
                p + "mlp.fc1.bias": torch.randn(inter) * 0.02,
                p + "mlp.fc2.weight": torch.randn(h, inter) * 0.02,
                p + "mlp.fc2.bias": torch.randn(h) * 0.02,
            }
        )
    return w


# The pi0.5 deployment feeds 3 camera views (agentview + wrist + placeholder);
# the SigLIP tower runs ONCE PER CAMERA (each 224x224 -> 256 patches), so this
# composite drives the tower over all 3 distinct camera images.
_N_CAMERAS = 3


def test_siglip_vision_tower(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower
    from tt_symbiote.models.pi05.modeling_pi05_siglip import TTNNPi05SigLIPVisionTower

    torch.manual_seed(SEED)
    cfg = SigLIPConfig()
    w = _siglip_tower_weights(cfg)
    ref = SigLIPVisionTower(cfg, w)
    tt = TTNNPi05SigLIPVisionTower.from_torch(ref, cfg)
    set_device(tt, dev)

    # Drive the tower over all 3 camera inputs (a distinct image per camera).
    for cam in range(_N_CAMERAS):
        torch.manual_seed(SEED + 100 + cam)
        pixel_values = torch.randn(1, cfg.num_channels, cfg.image_size, cfg.image_size) * 0.5
        out_ref = ref.forward(pixel_values)  # (1, 256, 1152)
        px_tt = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out_tt = tt.forward(px_tt)
        # 27 bf16 layers compound; reference E2E PCC ~0.991. Gate the composite at 0.95.
        assert_pcc(out_tt, out_ref, threshold=0.95, msg=f"SigLIPVisionTower(27L) cam{cam}")
