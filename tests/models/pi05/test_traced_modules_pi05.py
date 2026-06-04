# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-submodule TRACED validation (bottom-up Tier 1).

Each ``@trace_enabled`` pi0.5 submodule is driven via ``__call__`` under
``TT_SYMBIOTE_RUN_MODE=TRACED`` so the framework runs warmup -> capture -> replay.
We assert (a) a trace was actually captured (``TracedRun.cache_size`` grows) and
(b) the replayed output matches the torch golden. This validates each module's
traced execution path directly -- fast, and independent of the full model (where
run-once prefix modules never reach capture).

Random weights -> runs on a single Blackhole (P150) WITHOUT the gated checkpoint.
Each test has a timeout so a capture deadlock fails fast instead of holding the
device.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from tt_symbiote.core.run_config import TracedRun
from tt_symbiote.models.pi05.configuration_pi05 import GemmaConfig, SigLIPConfig
from tt_symbiote.models.pi05.modeling_pi05_common import precompute_freqs_cis_meta
from tt_symbiote.utils.device_management import set_device

from .pi05_helpers import assert_pcc, compute_pcc  # noqa: F401
from .pi05_helpers import SEED, require_reference

PCC = 0.99

# The run implementation is bound at import from TT_SYMBIOTE_RUN_MODE, so this
# file must be launched with TT_SYMBIOTE_RUN_MODE=TRACED (dispatch -> TracedRun).
# Skip cleanly otherwise so it never silently runs in NORMAL.
pytestmark = pytest.mark.skipif(
    os.environ.get("TT_SYMBIOTE_RUN_MODE") != "TRACED",
    reason="run with TT_SYMBIOTE_RUN_MODE=TRACED to validate the framework trace path",
)


def _rope(dev, head_dim, seq, base=10000.0):
    cos, sin = precompute_freqs_cis_meta(head_dim, max(seq, 64), dev, base)
    return ttnn.slice(cos, [0, 0, 0, 0], [1, 1, seq, head_dim]), ttnn.slice(sin, [0, 0, 0, 0], [1, 1, seq, head_dim])


def _drive_traced(dev, module, call_args, golden, thr=PCC, msg=""):
    """Call `module(*call_args)` 3x via __call__ (dispatch -> TracedRun, since the
    process is launched with TT_SYMBIOTE_RUN_MODE=TRACED); assert a trace was
    captured and the replayed (3rd) output matches the golden."""
    before = TracedRun.cache_size()
    out = None
    for _ in range(3):  # warmup -> capture -> replay
        r = module(*call_args)
        ttnn.synchronize_device(dev)
        out = r[0] if isinstance(r, tuple) else r
    captured = TracedRun.cache_size() - before
    assert captured >= 1, f"{msg}: no trace captured (cache grew by {captured})"
    pcc = compute_pcc(out, golden)
    assert pcc >= thr, f"{msg}: replayed PCC {pcc:.4f} < {thr}"
    return pcc


def _attn_w(cfg):
    return {
        "self_attn.q_proj.weight": torch.randn(cfg.num_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.k_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.v_proj.weight": torch.randn(cfg.num_kv_heads * cfg.head_dim, cfg.width) * 0.02,
        "self_attn.o_proj.weight": torch.randn(cfg.width, cfg.num_heads * cfg.head_dim) * 0.02,
    }


def _mlp_w(cfg):
    return {
        "mlp.gate_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.up_proj.weight": torch.randn(cfg.mlp_dim, cfg.width) * 0.02,
        "mlp.down_proj.weight": torch.randn(cfg.width, cfg.mlp_dim) * 0.02,
    }


@pytest.mark.timeout(180)
@pytest.mark.parametrize("config_fn,name", [(GemmaConfig.gemma_2b, "vlm"), (GemmaConfig.gemma_300m, "expert")], ids=["vlm", "expert"])
def test_traced_gemma_mlp(dev, config_fn, name):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import GemmaMLP
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05GemmaMLP

    torch.manual_seed(SEED)
    cfg = config_fn()
    w = _mlp_w(cfg)
    ref = GemmaMLP(cfg, w)
    x = torch.randn(1, 64, cfg.width) * 0.5
    golden = ref.forward(x)
    tt = TTNNPi05GemmaMLP.from_torch(ref, cfg)
    set_device(tt, dev)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    _drive_traced(dev, tt, (x_tt,), golden, msg=f"GemmaMLP[{name}]")


@pytest.mark.timeout(180)
def test_traced_gemma_attention(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import GemmaAttention, precompute_freqs_cis
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05GemmaAttention

    torch.manual_seed(SEED)
    cfg = GemmaConfig.gemma_2b()
    w = _attn_w(cfg)
    ref = GemmaAttention(cfg, w, 0)
    x = torch.randn(1, 64, cfg.width) * 0.5
    cos_t, sin_t = precompute_freqs_cis(cfg.head_dim, 2048, cfg.rope_base)
    golden, _ = ref.forward(x, cos_t, sin_t)
    tt = TTNNPi05GemmaAttention.from_torch(ref, cfg)
    set_device(tt, dev)
    cos, sin = _rope(dev, cfg.head_dim, 64, cfg.rope_base)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    _drive_traced(dev, tt, (x_tt, cos, sin), golden, msg="GemmaAttention")


@pytest.mark.timeout(180)
def test_traced_gemma_block(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import GemmaBlock, precompute_freqs_cis
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05GemmaBlock

    torch.manual_seed(SEED)
    cfg = GemmaConfig.gemma_2b()
    w = {"input_layernorm.weight": torch.randn(cfg.width) * 0.02, "post_attention_layernorm.weight": torch.randn(cfg.width) * 0.02, **_attn_w(cfg), **_mlp_w(cfg)}
    ref = GemmaBlock(cfg, w, 0)
    x = torch.randn(1, 64, cfg.width) * 0.5
    cos_t, sin_t = precompute_freqs_cis(cfg.head_dim, 2048, cfg.rope_base)
    golden, _ = ref.forward(x, cos_t, sin_t)
    tt = TTNNPi05GemmaBlock.from_torch(ref, cfg)
    set_device(tt, dev)
    cos, sin = _rope(dev, cfg.head_dim, 64, cfg.rope_base)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    _drive_traced(dev, tt, (x_tt, cos, sin), golden, msg="GemmaBlock")


@pytest.mark.timeout(180)
def test_traced_adarms_block(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock, precompute_freqs_cis
    from tt_symbiote.models.pi05.modeling_pi05_gemma import TTNNPi05AdaRMSGemmaBlock

    torch.manual_seed(SEED)
    cfg = GemmaConfig.gemma_300m()
    w = {
        "input_layernorm.dense.weight": torch.randn(3 * cfg.width, cfg.width) * 0.02,
        "input_layernorm.dense.bias": torch.randn(3 * cfg.width) * 0.02,
        "post_attention_layernorm.dense.weight": torch.randn(3 * cfg.width, cfg.width) * 0.02,
        "post_attention_layernorm.dense.bias": torch.randn(3 * cfg.width) * 0.02,
        **_attn_w(cfg), **_mlp_w(cfg),
    }
    ref = AdaRMSGemmaBlock(cfg, w, 0)
    x = torch.randn(1, 64, cfg.width) * 0.5
    cond = torch.randn(1, cfg.width) * 0.5
    cos_t, sin_t = precompute_freqs_cis(cfg.head_dim, 2048, cfg.rope_base)
    golden, _ = ref.forward(x, cos_t, sin_t, cond)
    tt = TTNNPi05AdaRMSGemmaBlock.from_torch(ref, cfg)
    set_device(tt, dev)
    cos, sin = _rope(dev, cfg.head_dim, 64, cfg.rope_base)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cond_tt = ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    _drive_traced(dev, tt, (x_tt, cos, sin, cond_tt), golden, msg="AdaRMSGemmaBlock")


def _siglip_w(cfg):
    h, inter = cfg.hidden_size, cfg.intermediate_size
    return {
        "layer_norm1.weight": torch.randn(h) * 0.02 + 1, "layer_norm1.bias": torch.randn(h) * 0.02,
        "layer_norm2.weight": torch.randn(h) * 0.02 + 1, "layer_norm2.bias": torch.randn(h) * 0.02,
        "self_attn.q_proj.weight": torch.randn(h, h) * 0.02, "self_attn.q_proj.bias": torch.randn(h) * 0.02,
        "self_attn.k_proj.weight": torch.randn(h, h) * 0.02, "self_attn.k_proj.bias": torch.randn(h) * 0.02,
        "self_attn.v_proj.weight": torch.randn(h, h) * 0.02, "self_attn.v_proj.bias": torch.randn(h) * 0.02,
        "self_attn.out_proj.weight": torch.randn(h, h) * 0.02, "self_attn.out_proj.bias": torch.randn(h) * 0.02,
        "mlp.fc1.weight": torch.randn(inter, h) * 0.02, "mlp.fc1.bias": torch.randn(inter) * 0.02,
        "mlp.fc2.weight": torch.randn(h, inter) * 0.02, "mlp.fc2.bias": torch.randn(h) * 0.02,
    }


@pytest.mark.timeout(180)
@pytest.mark.parametrize("cls_name", ["SigLIPMLP", "SigLIPAttention", "SigLIPBlock"])
def test_traced_siglip(dev, cls_name):
    require_reference()
    import models.experimental.pi0_5.reference.torch_siglip as RS
    import tt_symbiote.models.pi05.modeling_pi05_siglip as MS

    torch.manual_seed(SEED)
    cfg = SigLIPConfig()
    w = _siglip_w(cfg)
    ref = getattr(RS, cls_name)(cfg, w)
    x = torch.randn(1, cfg.num_patches, cfg.hidden_size) * 0.5
    golden = ref.forward(x)
    tt = getattr(MS, f"TTNNPi05{cls_name}").from_torch(ref, cfg)
    set_device(tt, dev)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    _drive_traced(dev, tt, (x_tt,), golden, msg=cls_name)


@pytest.mark.timeout(180)
def test_traced_mm_projector(dev):
    require_reference()
    from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector
    from tt_symbiote.models.pi05.modeling_pi05_siglip import TTNNPi05MultiModalProjector

    torch.manual_seed(SEED)
    cfg = SigLIPConfig()
    pw = {"linear.weight": torch.randn(2048, cfg.hidden_size) * 0.02, "linear.bias": torch.randn(2048) * 0.02}
    ref = MultiModalProjector(pw)
    x = torch.randn(1, cfg.num_patches, cfg.hidden_size) * 0.5
    golden = ref.forward(x)
    tt = TTNNPi05MultiModalProjector.from_torch(ref)
    set_device(tt, dev)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    _drive_traced(dev, tt, (x_tt,), golden, msg="MultiModalProjector")


# --------------------------------------------------------------------------- #
# Tier 2: full SigLIP vision tower traced as ONE graph (27 blocks captured
# together; per-block auto-trace can't capture run-once blocks).
# --------------------------------------------------------------------------- #
def _siglip_tower_w(cfg):
    h, inter = cfg.hidden_size, cfg.intermediate_size
    # checkpoint ("vision_model.*") key form -- the reference's `get(a) or get(b)`
    # idiom raises on a multi-element tensor when the first (legacy) key hits.
    w = {
        "vision_model.embeddings.patch_embedding.weight": torch.randn(h, cfg.num_channels, cfg.patch_size, cfg.patch_size) * 0.02,
        "vision_model.embeddings.patch_embedding.bias": torch.randn(h) * 0.02,
        "vision_model.embeddings.position_embedding.weight": torch.randn(cfg.num_patches, h) * 0.02,
        "vision_model.post_layernorm.weight": torch.randn(h) * 0.02 + 1.0,
        "vision_model.post_layernorm.bias": torch.randn(h) * 0.02,
    }
    for i in range(cfg.num_hidden_layers):
        p = f"vision_model.encoder.layers.{i}."
        for k, v in _siglip_w(cfg).items():
            w[p + k] = v
    return w


@pytest.mark.timeout(300)
def test_traced_siglip_vision_tower(dev):
    """Tier 2: capture+replay the whole SigLIP-27 tower under framework TRACED."""
    require_reference()
    from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower
    from tt_symbiote.models.pi05.modeling_pi05_siglip import TTNNPi05SigLIPVisionTower

    torch.manual_seed(SEED)
    cfg = SigLIPConfig()
    w = _siglip_tower_w(cfg)
    ref = SigLIPVisionTower(cfg, w)
    pixel_values = torch.randn(1, cfg.num_channels, cfg.image_size, cfg.image_size) * 0.5
    golden = ref.forward(pixel_values)  # (1, 256, 1152)

    tt = TTNNPi05SigLIPVisionTower.from_torch(ref, cfg)
    set_device(tt, dev)
    px = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    # 27 bf16 layers compound drift -> 0.95 gate (matches the NORMAL Tier-2 test).
    pcc = _drive_traced(dev, tt, (px,), golden, thr=0.95, msg="SigLIPVisionTower(27L)")
    print(f"\n[tier2-traced] SigLIP tower replay PCC={pcc:.4f}")


# --------------------------------------------------------------------------- #
# Tier 3: PaliGemma backbone VLM prefill (forward_vlm, use_cache=False) traced
# as ONE graph via a single-forward @trace_enabled adapter (the backbone itself
# is multi-method, so it can't be @trace_enabled directly). Real weights.
# --------------------------------------------------------------------------- #
from tt_symbiote.core.module import DeviceArch, TTNNModule, run_on_devices  # noqa: E402
from tt_symbiote.core.run_config import trace_enabled  # noqa: E402


@trace_enabled
class _VLMPrefillTraceAdapter(TTNNModule):
    """Single-`forward` trace unit wrapping backbone.forward_vlm(use_cache=False)."""

    @classmethod
    def wrap(cls, backbone, device):
        m = cls()
        m._bypass_tensor_wrapping = True
        m._bk = backbone  # already preprocessed + on device
        m._device = device
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    @run_on_devices(DeviceArch.P150)
    def forward(self, hidden):
        out, _ = self._bk.forward_vlm(hidden, attention_mask=None, use_cache=False)
        return out


@pytest.mark.timeout(300)
def test_traced_backbone_forward_vlm(dev):
    """Tier 3: capture+replay the 18-layer Gemma-2B VLM prefill under TRACED (real weights)."""
    require_reference()
    from .pi05_helpers import require_checkpoint

    ckpt = require_checkpoint()
    from tt_symbiote.models.pi05.configuration_pi05 import PaliGemmaConfig, Pi0_5ModelConfig
    from tt_symbiote.models.pi05.modeling_pi05 import load_reference_pi05_model
    from tt_symbiote.models.pi05.modeling_pi05_paligemma import TTNNPi05PaliGemmaBackbone

    torch.manual_seed(SEED)
    cfg = Pi0_5ModelConfig()
    ref_model = load_reference_pi05_model(ckpt)
    pg = PaliGemmaConfig(vlm_config=cfg.vlm_config, expert_config=cfg.expert_config, siglip_config=cfg.siglip_config)
    seq = 64
    hidden = torch.randn(1, seq, cfg.vlm_config.width) * 0.5
    golden, _ = ref_model.backbone.forward_vlm(hidden, use_cache=False)

    tt = TTNNPi05PaliGemmaBackbone.from_torch(ref_model.backbone, pg)
    set_device(tt, dev)
    adapter = _VLMPrefillTraceAdapter.wrap(tt, dev)
    h_tt = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    pcc = _drive_traced(dev, adapter, (h_tt,), golden, thr=0.95, msg="backbone.forward_vlm(18L)")
    print(f"\n[tier3-traced] backbone forward_vlm replay PCC={pcc:.4f}")


@trace_enabled
class _VLMPrefillStoreTraceAdapter(TTNNModule):
    """Single-`forward` trace unit wrapping backbone.forward_vlm WITH the VLM
    prefix store active (use_cache=True). This is the cache-building prefill path
    that previously could NOT be captured (each layer returned freshly-allocated
    persistent K/V). With the pre-allocated prefix store the per-layer K/V are
    mirrored in-place, so the whole 18-layer prefill now traces end-to-end."""

    @classmethod
    def wrap(cls, backbone, device):
        m = cls()
        m._bypass_tensor_wrapping = True
        m._bk = backbone
        m._device = device
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    @run_on_devices(DeviceArch.P150)
    def forward(self, hidden):
        out, _ = self._bk.forward_vlm(hidden, attention_mask=None, use_cache=True)
        return out


@pytest.mark.timeout(300)
def test_traced_backbone_forward_vlm_static_kv(dev):
    """Tier 3: capture+replay the cache-building VLM prefill (forward_vlm with the
    prefix store active) under TRACED. Proves the prefill traces end-to-end -- the
    in-place ttnn.fill_cache store write replaces the per-layer persistent-cache
    allocation that previously blocked capture. Asserts replay PCC vs torch AND
    that the prefix stores were actually populated."""
    require_reference()
    from .pi05_helpers import require_checkpoint

    ckpt = require_checkpoint()
    from tt_symbiote.models.pi05.configuration_pi05 import PaliGemmaConfig, Pi0_5ModelConfig
    from tt_symbiote.models.pi05.modeling_pi05 import load_reference_pi05_model
    from tt_symbiote.models.pi05.modeling_pi05_paligemma import TTNNPi05PaliGemmaBackbone

    torch.manual_seed(SEED)
    cfg = Pi0_5ModelConfig()
    ref_model = load_reference_pi05_model(ckpt)
    pg = PaliGemmaConfig(vlm_config=cfg.vlm_config, expert_config=cfg.expert_config, siglip_config=cfg.siglip_config)
    seq = 64  # tile-aligned prefix length
    hidden = torch.randn(1, seq, cfg.vlm_config.width) * 0.5
    golden, _ = ref_model.backbone.forward_vlm(hidden, use_cache=False)

    tt = TTNNPi05PaliGemmaBackbone.from_torch(ref_model.backbone, pg)
    set_device(tt, dev)
    tt.init_vlm_static_kv(seq)  # allocate the prefix stores OUTSIDE the trace region
    adapter = _VLMPrefillStoreTraceAdapter.wrap(tt, dev)
    h_tt = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    pcc = _drive_traced(dev, adapter, (h_tt,), golden, thr=0.95, msg="forward_vlm(store,18L)")

    # The prefix stores must have been populated in-place during the (traced) prefill.
    k0 = ttnn.to_torch(tt.vlm_blocks[0].attention._vlm_prefix_k)
    populated = bool(torch.isfinite(k0).all() and k0.abs().sum() > 0)
    print(f"\n[tier3-traced] forward_vlm(store) replay PCC={pcc:.4f} stores_populated={populated}")
    assert populated, "VLM prefix store was not populated during the traced prefill"


# --------------------------------------------------------------------------- #
# Tier 4: full sample_actions denoise loop under framework TRACED. The per-step
# velocity (embed_actions -> forward_expert[static-KV] -> project_output) is one
# @trace_enabled adapter, driven 10x: step1 warmup, step2 captures the WHOLE step
# as one trace (rms_norm compiles in warmup -> no per-block load_binaries), steps
# 3-10 replay with per-step (x_t, mods) copied to trace buffers. E2E vs torch.
# --------------------------------------------------------------------------- #


def _reference_sample_actions(ref_model, images_t, img_masks_t, lang_tokens_t, lang_masks_t, noise):
    """Torch reference E2E with injected noise (shared with the TTNN run)."""
    from models.experimental.pi0_5.reference.torch_pi0_5_model import _build_prefix_mask_and_pos
    prefix_embs, ppm, pam = ref_model.embed_prefix(images_t, img_masks_t, lang_tokens_t, lang_masks_t)
    pos, mask4d = _build_prefix_mask_and_pos(ppm, pam, prefix_embs.dtype)
    _, vlm_cache = ref_model.backbone.forward_vlm(prefix_embs, attention_mask=mask4d, position_ids=pos, use_cache=True)
    x = noise.clone()
    n = ref_model.config.num_denoising_steps
    for i in range(n):
        t = torch.tensor([1.0 - i / n])
        v = ref_model._denoise_forward(x, t, vlm_cache, prefix_pad_masks=ppm)
        x = x + (-1.0 / n) * v
    return x


@trace_enabled
class _DenoiseStepAdapter(TTNNModule):
    @classmethod
    def wrap(cls, model, prefix_len, suffix_mask, device):
        m = cls()
        m._bypass_tensor_wrapping = True
        m._m = model
        m._pl = prefix_len
        m._mask = suffix_mask
        m._nblocks = len(model.backbone.expert_blocks)
        m._device = device
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    @run_on_devices(DeviceArch.P150)
    def forward(self, x_t, *flat_mods):
        # Per-step mods are passed FLAT (top-level tensor args) so the framework's
        # trace input-copy (_copy_inputs_to_trace_buffer, which only copies
        # top-level ttnn.Tensors, not nested lists) updates every one on replay.
        # Reconstruct the [18 x (scale1,shift,gate) x2] + final (scale1,shift).
        nb = self._nblocks
        block_mods = [tuple(flat_mods[i * 6:(i + 1) * 6]) for i in range(nb)]
        final_mod = (flat_mods[nb * 6], flat_mods[nb * 6 + 1])
        se = self._m.suffix_embedding.embed_actions(x_t)
        eo = self._m.backbone.forward_expert(
            se, past_key_values=None, attention_mask=self._mask, position_offset=self._pl,
            precomputed_block_mods=block_mods, precomputed_final_mod=final_mod,
        )
        return self._m.suffix_embedding.project_output(eo)


def _flatten_step_mods(block_mods, final_mod):
    flat = []
    for six in block_mods:
        flat.extend(six)
    flat.extend(final_mod)
    return flat


@pytest.mark.timeout(360)
def test_traced_sample_actions_e2e(dev):
    """Tier 4: full 10-step denoise under framework TRACED (per-step adapter
    warmup->capture->replay); actions finite/correct-shape + E2E PCC vs torch."""
    require_reference()
    import math

    from .pi05_helpers import require_checkpoint

    ckpt = require_checkpoint()
    from tt_symbiote.models.pi05.configuration_pi05 import Pi0_5ModelConfig
    from tt_symbiote.models.pi05.modeling_pi05 import TTNNPi05Model, load_reference_pi05_model

    torch.manual_seed(SEED)
    cfg = Pi0_5ModelConfig()
    ref_model = load_reference_pi05_model(ckpt)
    tt = TTNNPi05Model.from_torch(ref_model, cfg)
    set_device(tt, dev)

    lang_len = 32
    images_t = [torch.randn(1, 3, 224, 224)]
    img_masks_t = [torch.ones(1, dtype=torch.bool)]
    lang_tokens_t = torch.randint(0, 257152, (1, lang_len))
    lang_masks_t = torch.ones(1, lang_len, dtype=torch.bool)
    noise = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    actions_ref = _reference_sample_actions(ref_model, images_t, img_masks_t, lang_tokens_t, lang_masks_t, noise)

    # Prefill + static-KV init (eager, outside trace).
    images = [ttnn.from_torch(images_t[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)]
    lang_tokens = ttnn.from_torch(lang_tokens_t.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    img_embeds = tt.backbone.embed_image(images[0])
    lang_embeds = ttnn.multiply(tt.backbone.embed_language_tokens(lang_tokens), math.sqrt(cfg.vlm_config.width))
    prefix = ttnn.concat([img_embeds, lang_embeds], dim=1)
    prefix_len = prefix.shape[1]
    ah, ad = cfg.action_horizon, cfg.action_dim
    ahp = tt._tile_pad(ah)
    # Prefill via the trace-safe VLM prefix store, then fill the expert cross-attn
    # KV buffers from those stores (no intermediate persistent cache list).
    tt.backbone.init_vlm_static_kv(prefix_len)
    tt.backbone.forward_vlm(prefix, attention_mask=None, use_cache=True)
    tt.backbone.init_expert_static_kv_from_vlm(prefix_len, ahp)
    suffix_mask = tt._suffix_phantom_mask(prefix_len, ah, ahp)
    step_mods = [tt.backbone.precompute_step_mods(tt.suffix_embedding.embed_adarms_cond(tt._ts(1.0 - i / cfg.num_denoising_steps))) for i in range(cfg.num_denoising_steps)]

    # Framework-traced denoise loop.
    adapter = _DenoiseStepAdapter.wrap(tt, prefix_len, suffix_mask, dev)
    x_t = tt._noise_to_device(noise, ahp)
    n = cfg.num_denoising_steps
    before = TracedRun.cache_size()
    for i in range(n):
        bmods, fmod = step_mods[i]
        v = adapter(x_t, *_flatten_step_mods(bmods, fmod))  # warmup(0)/capture(1)/replay(2+)
        v_dt = ttnn.multiply(v, -1.0 / n, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_t = ttnn.add(x_t, v_dt, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)
    captured = TracedRun.cache_size() - before
    out = ttnn.to_torch(ttnn.slice(x_t, [0, 0, 0], [x_t.shape[0], ah, ad]))

    print(f"\n[tier4-traced] captured={captured} actions_shape={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
    assert captured >= 1, "no denoise-step trace captured"
    assert out.shape == (1, ah, ad) and torch.isfinite(out).all() and out.float().std() > 1e-4
    pcc = compute_pcc(out, actions_ref)
    print(f"[tier4-traced] E2E action PCC (framework-TRACED vs torch) = {pcc:.4f}")
    assert pcc >= 0.90, f"E2E PCC {pcc:.4f} < 0.90"
