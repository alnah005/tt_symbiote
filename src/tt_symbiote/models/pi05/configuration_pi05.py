# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for the pi0.5 (PI0.5) vision-language-action model.

Ported from the tt-metal reference at
``models/experimental/pi0_5/common/configs.py`` (branch ``tt/pi0.5_bh``,
commit ``b0703a5``). pi0.5 is a lerobot/openpi checkpoint and is *not* a
``transformers`` model, so these configs are self-contained dataclasses
rather than a ``transformers.PretrainedConfig`` subclass.

Architecture summary (see ``modeling_pi05.py`` module docstring for the
full data-flow):

* SigLIP-27 vision tower (1152 hidden, 27 layers, 16 heads, 224/14 patches).
* Gemma-2B VLM backbone (2048 width, 18 layers, 8 heads / 1 KV head, head_dim 256).
* Gemma-300M action expert (1024 width, 18 layers, adaRMS modulation) sharing
  K/V with the VLM.
* Flow-matching denoiser (10 Euler steps) producing a (B, action_horizon,
  action_dim) action chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "GemmaConfig",
    "SigLIPConfig",
    "SuffixConfig",
    "PrefixConfig",
    "PaliGemmaConfig",
    "DenoiseConfig",
    "Pi0_5ModelConfig",
]


@dataclass
class GemmaConfig:
    """Configuration for a Gemma transformer stack (VLM backbone or action expert)."""

    width: int = 2048
    depth: int = 18
    mlp_dim: int = 16384
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0

    @classmethod
    def gemma_2b(cls) -> "GemmaConfig":
        """Gemma 2B configuration (VLM backbone)."""
        return cls(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)

    @classmethod
    def gemma_300m(cls) -> "GemmaConfig":
        """Gemma 300M configuration (action expert)."""
        return cls(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)


@dataclass
class SigLIPConfig:
    """Configuration for the SigLIP vision encoder."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class SuffixConfig:
    """Configuration for the suffix (action + time) embedding."""

    action_dim: int = 32
    action_horizon: int = 50
    expert_width: int = 1024
    state_dim: int = 32
    time_emb_dim: int = 1024
    pi05: bool = True  # pi0.5 uses adaRMS time conditioning (no fused action-time MLP)


@dataclass
class PrefixConfig:
    """Configuration for the prefix (image + language) embedding."""

    vlm_hidden_size: int = 2048
    num_image_tokens: int = 256  # tokens per image from SigLIP
    max_lang_tokens: int = 512


@dataclass
class PaliGemmaConfig:
    """Configuration for the PaliGemma dual-expert backbone."""

    vlm_config: GemmaConfig = None
    expert_config: GemmaConfig = None
    siglip_config: SigLIPConfig = None
    max_seq_len: int = 2048

    def __post_init__(self):
        if self.vlm_config is None:
            self.vlm_config = GemmaConfig.gemma_2b()
        if self.expert_config is None:
            self.expert_config = GemmaConfig.gemma_300m()
        if self.siglip_config is None:
            self.siglip_config = SigLIPConfig()


@dataclass
class DenoiseConfig:
    """Configuration for the flow-matching denoiser."""

    num_steps: int = 10
    noise_scale: float = 1.0
    action_dim: int = 32
    action_horizon: int = 50


@dataclass
class Pi0_5ModelConfig:
    """Complete configuration for the pi0.5 model.

    Differences vs pi0:
      * ``pi05=True`` drives the adaRMS path in the suffix + action expert.
      * ``max_token_len=200`` (pi0.5 default in openpi).
      * ``discrete_state_input=True`` (robot state encoded as language tokens).
    """

    # Core dimensions
    action_dim: int = 32
    action_horizon: int = 50
    state_dim: int = 32

    # Processing
    precision: str = "bfloat16"
    num_denoising_steps: int = 10
    max_seq_len: int = 2048

    # pi0.5 specifics
    pi05: bool = True
    max_token_len: int = 200
    discrete_state_input: bool = True

    # Component configs
    vlm_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_2b)
    expert_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_300m)
    siglip_config: SigLIPConfig = field(default_factory=SigLIPConfig)

    def __post_init__(self):
        # Always pin component configs to the canonical pi0.5 variants.
        self.vlm_config = GemmaConfig.gemma_2b()
        self.expert_config = GemmaConfig.gemma_300m()
        self.siglip_config = SigLIPConfig()

    @property
    def suffix_config(self) -> SuffixConfig:
        return SuffixConfig(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            expert_width=self.expert_config.width,
            state_dim=self.state_dim,
            time_emb_dim=self.expert_config.width,
            pi05=self.pi05,
        )

    @property
    def prefix_config(self) -> PrefixConfig:
        return PrefixConfig(
            vlm_hidden_size=self.vlm_config.width,
            num_image_tokens=self.siglip_config.num_patches,
            max_lang_tokens=self.max_token_len,
        )

    @property
    def denoise_config(self) -> DenoiseConfig:
        return DenoiseConfig(
            num_steps=self.num_denoising_steps,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )
