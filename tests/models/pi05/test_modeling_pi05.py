# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 3/4 (sub-model + end-to-end) PCC + semantic tests for pi0.5.

These exercise the full PaliGemma backbone and the end-to-end ``sample_actions``
flow-matching policy against the pi0_5 PyTorch golden using REAL ``pi05_base``
weights (gated, ~14.5GB), so they SKIP automatically when the checkpoint is
absent. Component correctness (Tier 1/2) is covered without weights in
``test_ops_pi05.py`` / ``test_composites_pi05.py``.

Tier 4 additionally implements the semantic-coherence gate required by the
model-bringup decision profile: the produced action chunk must be finite, the
correct shape (1, action_horizon, action_dim), and correlate with the torch
reference (the reference reports mean E2E PCC ~0.991 over 10 seeds).

NOTE: ``TTNNPi05Model.sample_actions`` is scaffolded (the host-side mask building
and Euler denoise loop are completed during this stage once weights are
available). Until then these tests xfail/skip with a clear reason.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
from tt_symbiote.utils.device_management import set_device

from .pi05_helpers import compute_pcc
from .pi05_helpers import SEED, require_checkpoint, require_reference


@pytest.fixture(scope="module")
def ref_model():
    require_reference()
    ckpt = require_checkpoint()
    from tt_symbiote.models.pi05.modeling_pi05 import load_reference_pi05_model

    return load_reference_pi05_model(ckpt)


def test_backbone_forward_vlm_pcc(dev, ref_model):
    """Tier 3: 18-layer Gemma-2B VLM stack (real weights), TTNN vs torch."""
    from tt_symbiote.models.pi05.configuration_pi05 import PaliGemmaConfig
    from tt_symbiote.models.pi05.modeling_pi05_paligemma import TTNNPi05PaliGemmaBackbone

    torch.manual_seed(SEED)
    cfg = Pi0_5ModelConfig()
    pg = PaliGemmaConfig(vlm_config=cfg.vlm_config, expert_config=cfg.expert_config, siglip_config=cfg.siglip_config)
    seq = 64
    hidden = torch.randn(1, seq, cfg.vlm_config.width) * 0.5
    out_ref, _ = ref_model.backbone.forward_vlm(hidden, use_cache=False)

    tt = TTNNPi05PaliGemmaBackbone.from_torch(ref_model.backbone, pg)
    set_device(tt, dev)
    h_tt = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_tt, _ = tt.forward_vlm(h_tt, use_cache=False)
    pcc = compute_pcc(out_tt, out_ref)
    assert pcc >= 0.95, f"forward_vlm PCC {pcc:.4f} < 0.95"


def _reference_sample_actions(ref_model, images_t, img_masks_t, lang_tokens_t, lang_masks_t, noise):
    """Run the torch reference E2E with injected noise (so TTNN and torch share it)."""
    from models.experimental.pi0_5.reference.torch_pi0_5_model import _build_prefix_mask_and_pos

    prefix_embs, prefix_pad_masks, prefix_att_masks = ref_model.embed_prefix(
        images_t, img_masks_t, lang_tokens_t, lang_masks_t
    )
    pos, mask4d = _build_prefix_mask_and_pos(prefix_pad_masks, prefix_att_masks, prefix_embs.dtype)
    _, vlm_cache = ref_model.backbone.forward_vlm(
        prefix_embs, attention_mask=mask4d, position_ids=pos, use_cache=True
    )
    x = noise.clone()
    n = ref_model.config.num_denoising_steps
    for i in range(n):
        t = torch.tensor([1.0 - i / n])
        dt = -1.0 / n
        v = ref_model._denoise_forward(x, t, vlm_cache, prefix_pad_masks=prefix_pad_masks)
        x = x + dt * v
    return x  # (1, action_horizon, action_dim)


def test_sample_actions_e2e_and_semantic(dev, ref_model):
    """Tier 4: end-to-end action chunk -- TTNN vs torch (shared noise) + semantic gate.

    Uses unpadded, tile-aligned inputs (1 image -> 256 patches + 32 language
    tokens = 288 prefix) so the baseline (full attention, sequential/offset RoPE,
    phantom-suffix mask) matches the reference. Flow-matching over random noise is
    chaotic and amplifies bf16 drift, so the PCC gate is lenient (0.90); the hard
    requirement is the semantic-coherence gate (finite, correct shape,
    non-degenerate) per the decision profile.
    """
    from tt_symbiote.models.pi05.modeling_pi05 import TTNNPi05Model

    torch.manual_seed(SEED)
    cfg = Pi0_5ModelConfig()
    lang_len = 32  # 256 (1 image) + 32 = 288 = tile-aligned prefix
    images_t = [torch.randn(1, 3, 224, 224)]
    img_masks_t = [torch.ones(1, dtype=torch.bool)]
    lang_tokens_t = torch.randint(0, 257152, (1, lang_len))
    lang_masks_t = torch.ones(1, lang_len, dtype=torch.bool)
    noise = torch.randn(1, cfg.action_horizon, cfg.action_dim)

    # Reference E2E (fp32) with the same noise.
    actions_ref = _reference_sample_actions(ref_model, images_t, img_masks_t, lang_tokens_t, lang_masks_t, noise)

    # TTNN E2E on P150.
    tt = TTNNPi05Model.from_torch(ref_model, cfg)
    set_device(tt, dev)
    images = [ttnn.from_torch(images_t[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
    img_masks = [ttnn.from_torch(torch.ones(1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
    lang_tokens = ttnn.from_torch(lang_tokens_t.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    lang_masks = ttnn.from_torch(torch.ones(1, lang_len), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    actions = tt.sample_actions(images, img_masks, lang_tokens, lang_masks, noise=noise)
    actions_torch = ttnn.to_torch(actions)

    # Semantic-coherence gate (the hard requirement).
    assert actions_torch.shape == (1, cfg.action_horizon, cfg.action_dim), actions_torch.shape
    assert torch.isfinite(actions_torch).all(), "actions contain NaN/inf"
    assert actions_torch.float().std() > 1e-4, "actions are degenerate (constant)"

    # E2E correlation vs the torch reference (lenient; chaotic flow-matching).
    pcc = compute_pcc(actions_torch, actions_ref)
    print(f"\npi0.5 E2E action PCC (TTNN vs torch, shared noise): {pcc:.4f}")
    assert pcc >= 0.90, f"E2E action PCC {pcc:.4f} < 0.90"
