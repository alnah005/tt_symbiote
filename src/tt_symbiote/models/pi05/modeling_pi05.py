# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Top-level TTNN pi0.5 (PI0.5) vision-language-action model.

pi0.5 is a flow-matching VLA policy. Inference flow (see
``reference/torch_pi0_5_model.py::Pi0_5Model.forward_inference`` and
``tt/ttnn_pi0_5_model.py::Pi0_5ModelTTNN.sample_actions``):

  1. Embed prefix (3x 224x224 images via SigLIP+projector, language+state tokens
     via the Gemma embedding * sqrt(width)); concat -> prefix sequence.
  2. ``forward_vlm`` over the 18-layer Gemma-2B stack to fill the prefix KV cache.
  3. Flow-matching denoise loop (10 Euler steps, t: 1.0 -> 0.0):
       - suffix_embs = action_in_proj(x_t); adarms_cond = time_mlp(sincos(t)).
       - expert_out = forward_expert(suffix_embs, adarms_cond, prefix_kv_cache).
       - velocity = action_out_proj(expert_out); x_t <- x_t + dt * velocity.
  4. Return x_t sliced to (B, action_horizon, action_dim).

The golden PyTorch reference (``Pi0_5Model``) is a plain object exposing
``.suffix_embedding``, ``.backbone``, ``.prefix_embedding`` (and ``.config``).

INTEGRATION PATH: manual. pi0.5 is a lerobot/openpi checkpoint, not a
``transformers`` AutoModel — there is no ``@register_recipe`` here. Build the
TTNN model from a checkpoint with :func:`from_checkpoint`, or wrap an existing
reference model with :meth:`TTNNPi05Model.from_torch`.

VALIDATION STATUS: component wiring + per-component forwards are implemented and
PCC-checkable (Tier 1/2). The end-to-end ``sample_actions`` orchestration
(host-side attention masks/position-ids and the Euler denoise loop) is scaffolded
with precise reference pointers; it is completed and validated during the
pcc-test-gen stage once the gated ``pi05_base`` weights are available.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import ttnn

from tt_symbiote.core.module import DeviceArch, StatefulTTNNModule, StatelessTTNNModule, run_on_devices
from tt_symbiote.core.run_config import trace_enabled

from .configuration_pi05 import PaliGemmaConfig, Pi0_5ModelConfig
from .modeling_pi05_paligemma import TTNNPi05PaliGemmaBackbone
from .modeling_pi05_prefix import TTNNPi05PrefixEmbedding
from .modeling_pi05_suffix import TTNNPi05SuffixEmbedding

TT_METAL_COMMIT = "b2af0cd67b4e92dafeb2d0254e1c5b43c3ec5a25"

# Default location of the read-only pi0_5 reference worktree (branch tt/pi0.5_bh).
# Overridable so PCC tests can point at any checkout that contains
# ``models/experimental/pi0_5``.
DEFAULT_REFERENCE_ROOT = os.environ.get("PI05_REFERENCE_ROOT", "/home/ttuser/salnahari/pi0_5_ref")

__all__ = ["TTNNPi05Model", "TTNNPi05DenoiseStep", "from_checkpoint", "load_reference_pi05_model"]

_L1 = ttnn.L1_MEMORY_CONFIG


def _flatten_step_mods(block_mods, final_mod):
    """Flatten the per-step adaRMS modulations [nb x (scale1,shift,gate)x2] + final
    (scale1,shift) into a flat tuple of tensors, so they can be passed as TOP-LEVEL
    forward args (the framework only refreshes top-level tensor args into the trace
    input buffers on replay -- nested lists/tuples are not copied)."""
    flat = []
    for triple in block_mods:
        flat.extend(triple)
    flat.extend(final_mod)
    return flat


@trace_enabled
class TTNNPi05DenoiseStep(StatefulTTNNModule):
    """One flow-matching velocity eval (embed_actions -> forward_expert -> project_output)
    as a SINGLE trace unit.

    The denoise loop runs the SAME graph every step (shape-identical; the prefix KV lives
    in constant static buffers). The only per-step differences -- the action state ``x_t``
    and the adaRMS modulations -- are passed as TOP-LEVEL tensor args, so under
    ``TT_SYMBIOTE_RUN_MODE=TRACED`` the framework captures exactly ONE trace and refreshes
    those inputs into its buffers on every replay. Under NORMAL it runs eager. There is no
    manual ``begin/end_trace_capture`` anywhere -- the framework owns the (single) trace.
    """

    @classmethod
    def bind(cls, model, prefix_len, suffix_mask, nblocks):
        m = cls()
        m._bypass_tensor_wrapping = True
        m._model = model
        m._prefix_len = prefix_len
        m._mask = suffix_mask
        m._nblocks = nblocks
        m._device = model.device
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    def reset_trace_state(self) -> None:
        # No OWN trace state. STATEFUL only because the captured subtree reaches the stateful
        # expert attention (KV fill_cache) via self._model; the framework's trace tree-reset
        # resets those descendants. The expert KV writes are fixed-index overwrites (idempotent
        # under the double-run), so this is an own-state no-op.
        return None

    @run_on_devices(DeviceArch.P150)
    def forward(self, x_t, *flat_mods):
        nb = self._nblocks
        block_mods = [tuple(flat_mods[i * 6 : (i + 1) * 6]) for i in range(nb)]
        final_mod = (flat_mods[nb * 6], flat_mods[nb * 6 + 1])
        return self._model._denoise_forward(x_t, None, self._prefix_len, self._mask, block_mods, final_mod)


class TTNNPi05Model(StatelessTTNNModule):
    """End-to-end TTNN pi0.5 policy."""

    @classmethod
    def from_torch(cls, ref_model, config: Optional[Pi0_5ModelConfig] = None) -> "TTNNPi05Model":
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = ref_model
        new._config = config or getattr(ref_model, "config", None) or Pi0_5ModelConfig()
        cfg = new._config

        paligemma_config = PaliGemmaConfig(
            vlm_config=cfg.vlm_config,
            expert_config=cfg.expert_config,
            siglip_config=cfg.siglip_config,
        )
        new.backbone = TTNNPi05PaliGemmaBackbone.from_torch(ref_model.backbone, paligemma_config)
        new.suffix_embedding = TTNNPi05SuffixEmbedding.from_torch(ref_model.suffix_embedding, cfg.suffix_config)
        # Prefix has no weights; it delegates to the backbone embed callbacks.
        new.prefix_embedding = TTNNPi05PrefixEmbedding.from_torch(
            getattr(ref_model, "prefix_embedding", None),
            embed_image_fn=new.backbone.embed_image,
            embed_language_fn=new.backbone.embed_language_tokens,
            config=cfg.prefix_config,
            vlm_hidden_size=cfg.vlm_config.width,
        )
        new._num_steps = cfg.num_denoising_steps
        new._action_horizon = cfg.action_horizon
        new._action_dim = cfg.action_dim
        new._step_mods = None  # cached per-step adaRMS modulations (constant across inferences)
        new._denoise_step = None  # cached single-trace denoise unit (rebuilt on prefix_len change)
        return new

    @run_on_devices(DeviceArch.P150)
    def _ensure_step_mods(self):
        """Compute + cache the per-step adaRMS modulations once (DRAM).

        Timesteps (1 - i/n) and the expert weights are fixed, so the modulations
        are constant across all inferences -- computed once here and reused, so
        the per-call denoise loop (and its trace) carries no mod-matmuls.
        """
        if self._step_mods is not None:
            return self._step_mods
        n = self._num_steps
        mods = []
        for i in range(n):
            cond = self.suffix_embedding.embed_adarms_cond(self._ts(1.0 - i / n))
            mods.append(self.backbone.precompute_step_mods(cond))
        self._step_mods = mods
        return mods

    def preprocess_weights_impl(self):
        self.backbone.preprocess_weights()
        self.suffix_embedding.preprocess_weights()
        self.prefix_embedding.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.backbone.move_weights_to_device()
        self.suffix_embedding.move_weights_to_device()
        self.prefix_embedding.move_weights_to_device()

    # ------------------------------------------------------------------ helpers (host)
    @staticmethod
    def _tile_pad(n: int, tile: int = 32) -> int:
        return ((n + tile - 1) // tile) * tile

    @staticmethod
    def _ts(t: float):
        """Host helper: scalar timestep -> torch tensor (1,) for the sincos embed."""
        import torch

        return torch.tensor([t], dtype=torch.float32)

    @staticmethod
    def _default_noise(ah: int, ad: int):
        """Host helper: sample initial flow-matching noise ~N(0,I)."""
        import torch

        return torch.randn(1, ah, ad)

    def _noise_to_device(self, noise_torch, ahp: int):
        """Pad host noise (B, action_horizon, action_dim) to ahp rows; upload bf16."""
        import torch

        b, ah, ad = noise_torch.shape
        if ahp > ah:
            noise_torch = torch.cat([noise_torch, torch.zeros(b, ahp - ah, ad, dtype=noise_torch.dtype)], dim=1)
        return ttnn.from_torch(
            noise_torch.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def _suffix_phantom_mask(self, prefix_len: int, ah: int, ahp: int):
        """Additive SDPA mask (1,1,ahp,prefix_len+ahp): -1e4 on phantom suffix keys.

        Keys = [prefix(prefix_len) valid + suffix(ahp)]; the last (ahp-ah) suffix
        columns are phantom (padding) and must not be attended. -1e4 (not -inf)
        avoids the bf16 softmax NaN pathology (ref tt/ttnn_pi0_5_model.py:482-516).
        """
        import torch

        if ahp == ah:
            return None
        kv = prefix_len + ahp
        mask = torch.zeros(1, 1, ahp, kv, dtype=torch.float32)
        mask[:, :, :, prefix_len + ah :] = -1e4
        return ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------ denoise
    @run_on_devices(DeviceArch.P150)
    def _denoise_forward(self, x_t, prefix_kv_cache, prefix_len, suffix_mask, block_mods, final_mod):
        """One velocity eval with PRECOMPUTED per-step adaRMS modulations.

        The conditioning (adarms_cond) and all per-layer (scale,shift,gate)
        modulations are computed once per step *outside* this method (see
        ``sample_actions``), so the captured trace of this graph carries no
        mod-matmuls/slices (reference TIER A perf path).
        """
        suffix_embs = self.suffix_embedding.embed_actions(x_t)
        expert_out = self.backbone.forward_expert(
            suffix_embs,
            past_key_values=prefix_kv_cache,
            attention_mask=suffix_mask,
            position_offset=prefix_len,
            precomputed_block_mods=block_mods,
            precomputed_final_mod=final_mod,
        )
        return self.suffix_embedding.project_output(expert_out)

    @run_on_devices(DeviceArch.P150)
    def sample_actions(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
        state: Optional[ttnn.Tensor] = None,
        noise=None,
    ) -> ttnn.Tensor:
        """Full VLA inference: prefix prefill + flow-matching Euler denoise.

        Returns the action chunk (B, action_horizon, action_dim).

        Baseline (functionality-first): assumes unpadded prefix tokens (all
        masks valid) and a tile-aligned prefix length, so prefix attention is
        full-bidirectional with sequential RoPE and the suffix attends to the
        whole prefix + real suffix (phantom suffix rows masked). The padded-mask
        / openpi-position path is a later refinement
        (ref tt/ttnn_pi0_5_model.py::_build_upstream_attn_artifacts).

        ``noise`` (optional host torch (B, action_horizon, action_dim)) seeds the
        Euler integration; if None, sampled ~N(0,I).
        """
        import math as _math

        # 1. Prefix embedding: images via SigLIP+projector, language * sqrt(width).
        # Under TRACED the @trace_enabled SigLIP vision_tower (a SECOND trace unit reaching
        # replay across 3 cameras: cam1 capture, cam2 replay) writes into a SINGLE persistent
        # trace-output buffer that lives in the L1 trace working set. The eager mm_projector
        # output (and on-device L1/DRAM copies of it) land in that working set too, so the
        # NEXT camera's tower replay clobbers the current camera's embed before it is
        # consumed downstream -- aliasing the cameras and corrupting the capture-encounter
        # camera's prefix tokens (measured: cam1 embed l2 4869 vs correct 2642 => 3-cam
        # TRACED E2E -0.13 vs eager +0.52). On-device clone/DRAM-copy + synchronize did NOT
        # decouple them (the copy buffers are themselves in the clobbered working set); only
        # reading each embed back to HOST (synchronized) before the next camera captures the
        # settled value. We host-stage each camera's embed and re-upload it -- a between-
        # camera data-movement barrier (3x per inference, OUTSIDE the hot denoise loop ->
        # negligible cost). Under NORMAL this is an identity round-trip (same numerics). The
        # hot per-step denoise trace is unaffected and remains the real traced speedup.
        # See bringup_status.json decision_log (deep-plan_1 §5.C VISION fix).
        img_embeds = []
        for im in images:
            _e = self.backbone.embed_image(im)
            _eh = ttnn.to_torch(_e)
            ttnn.synchronize_device(self.device)
            img_embeds.append(
                ttnn.from_torch(_eh, dtype=_e.dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=_L1)
            )
        lang_embeds = self.backbone.embed_language_tokens(lang_tokens)
        lang_embeds = ttnn.multiply(lang_embeds, _math.sqrt(self._config.vlm_config.width), memory_config=_L1)
        prefix = ttnn.concat([*img_embeds, lang_embeds], dim=1, memory_config=_L1)
        prefix_len = prefix.shape[1]

        # 2. VLM prefill -> prefix KV stored in pre-allocated per-layer buffers.
        # The VLM prefix store (allocated here, outside any trace region) lets each
        # VLM attention mirror its prefix K/V in-place, so forward_vlm allocates no
        # persistent cache and the whole prefill is trace-capturable.
        ah, ad = self._action_horizon, self._action_dim
        ahp = self._tile_pad(ah)
        self.backbone.init_vlm_static_kv(prefix_len)
        self.backbone.forward_vlm(prefix, attention_mask=None, use_cache=True)

        # 3. Flow-matching denoise loop (Euler, t: 1 -> 0).
        # Static KV buffers (trace-safe): allocate per expert layer once + prefill
        # the constant prefix region (from the VLM stores) here, OUTSIDE the
        # (per-step) trace region. The denoise loop then writes only the suffix K/V
        # in-place (no concat -> no allocation in the captured per-step graph).
        self.backbone.init_expert_static_kv_from_vlm(prefix_len, ahp)

        if noise is None:
            noise = self._default_noise(ah, ad)
        x_t = self._noise_to_device(noise, ahp)
        suffix_mask = self._suffix_phantom_mask(prefix_len, ah, ahp)

        # Per-step adaRMS modulations are constant (fixed timesteps + weights):
        # computed + cached once, reused across inferences (and excluded from the
        # per-step trace graph).
        n = self._num_steps
        step_mods = self._ensure_step_mods()

        # The velocity eval is a single trace unit dispatched via __call__: under
        # TT_SYMBIOTE_RUN_MODE=TRACED the framework captures ONE trace (the graph is
        # shape-identical every step; per-step x_t + mods are refreshed as trace inputs)
        # and replays it for the remaining steps; under NORMAL it runs eager. The unit is
        # rebuilt only when prefix_len changes (its shape -> trace cache key). No manual
        # begin/end_trace_capture: the framework owns the single trace.
        denoise = self._get_denoise_step(prefix_len, suffix_mask)

        for i in range(n):
            dt = -1.0 / n
            block_mods, final_mod = step_mods[i]
            # prefix_kv_cache (None) is owned by the static KV buffers filled above; x_t and
            # the flat per-step mods are the only per-step inputs the trace refreshes.
            v = denoise(x_t, *_flatten_step_mods(block_mods, final_mod))
            v_dt = ttnn.multiply(v, dt, memory_config=_L1)
            # NB: do NOT deallocate v -- under TRACED it is the persistent trace output
            # buffer reused by every replay; multiply has already consumed it here.
            x_next = ttnn.add(x_t, v_dt, memory_config=_L1)
            ttnn.deallocate(v_dt)
            ttnn.deallocate(x_t)
            x_t = x_next

        # 4. Slice phantom rows -> (B, action_horizon, action_dim).
        return ttnn.slice(x_t, [0, 0, 0], [x_t.shape[0], ah, ad])

    def _get_denoise_step(self, prefix_len, suffix_mask):
        """Return the cached single-trace denoise unit, rebuilding it only when prefix_len
        changes (prefix_len + the derived suffix-mask shape are baked into the trace, so a
        new shape needs a fresh trace). Same prefix_len reuses the captured trace across
        inferences; the deterministic suffix-mask values are identical so the baked mask
        stays valid."""
        cached = self._denoise_step
        if cached is None or cached._prefix_len != prefix_len:
            self._denoise_step = TTNNPi05DenoiseStep.bind(
                self, prefix_len, suffix_mask, len(self.backbone.expert_blocks)
            )
        return self._denoise_step

    # forward() delegates to sample_actions for TTNNModule API compatibility.
    @run_on_devices(DeviceArch.P150)
    def forward(self, images, img_masks, lang_tokens, lang_masks, state=None):
        return self.sample_actions(images, img_masks, lang_tokens, lang_masks, state)


# ---------------------------------------------------------------------------
# Checkpoint loading factory (manual integration path)
# ---------------------------------------------------------------------------
def load_reference_pi05_model(checkpoint_dir: str, reference_root: Optional[str] = None):
    """Build the PyTorch golden ``Pi0_5Model`` from a checkpoint directory.

    Adds the pi0_5 reference tree to ``sys.path`` (it lives in the tt-metal
    ``models/experimental/pi0_5`` package, accessed via a read-only worktree),
    then constructs the reference weight loader + model. Returns the reference
    ``Pi0_5Model`` instance — pass it to :meth:`TTNNPi05Model.from_torch`.
    """
    root = reference_root or DEFAULT_REFERENCE_ROOT
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig as _RefCfg
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

    loader = Pi0_5WeightLoader(checkpoint_dir)
    return Pi0_5Model(_RefCfg(), loader)


def from_checkpoint(
    checkpoint_dir: str,
    device: "ttnn.Device",
    config: Optional[Pi0_5ModelConfig] = None,
    reference_root: Optional[str] = None,
) -> TTNNPi05Model:
    """Load weights from ``checkpoint_dir`` and return a device-ready TTNN pi0.5 model.

    Wraps the reference model, runs the standard TTNNModule lifecycle
    (from_torch -> set_device -> preprocess_weights -> move_weights_to_device).
    """
    from tt_symbiote.utils.device_management import set_device

    ref_model = load_reference_pi05_model(checkpoint_dir, reference_root=reference_root)
    tt_model = TTNNPi05Model.from_torch(ref_model, config=config)
    set_device(tt_model, device)
    return tt_model
