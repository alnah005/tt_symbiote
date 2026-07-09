# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference torch loader for baidu/Unlimited-OCR under transformers 5.9.0.

Encapsulates every config backfill + monkeypatch needed to instantiate the
custom remote-code DeepSeek-OCR-style VLM on CPU and run text-only LM +
vision-encoder forwards. This is HOST/torch reference code (NOT a TTNNModule),
so torch is permitted throughout. Both the Auto recipe and the PCC tests import
this so there is a single source of truth for the load procedure.

Procedure (see deep-work/unlimited_ocr/torch_compat_recipe.md):
  1. install_transformers_shims()  (is_torch_fx_available, ROPE default init)
  2. AutoConfig.from_pretrained(..., trust_remote_code=True)
  3. backfill DeepseekV2Config defaults dropped by 5.9.0 deserialization
  4. cfg.standardize_rope_params() + force eager attention
  5. AutoModel.from_pretrained(..., dtype=..., attn_implementation="eager")
  6. rematerialize non-persistent buffers (rotary inv_freq, CLIP position_ids)

adapted from deep-work/unlimited_ocr/ref_loader.py (validated: 3,336,106,240
params; text logits (1,8,129280) finite; vision (1,256,1280) finite).
"""

from __future__ import annotations

import warnings

import torch

MID = "baidu/Unlimited-OCR"

# DeepseekV2Config.__init__ defaults dropped when the legacy config is rebuilt
# from dict under transformers 5.9.0; backfill any that are missing/None.
_DEEPSEEK_DEFAULTS = dict(
    attention_dropout=0.0,
    attention_bias=False,
    rope_theta=10000.0,
    rope_scaling=None,
    pretraining_tp=1,
    initializer_range=0.02,
    ep_size=1,
    routed_scaling_factor=1.0,
    moe_layer_freq=1,
    norm_topk_prob=False,
    scoring_func="softmax",
    aux_loss_alpha=0.001,
    seq_aux=True,
    hidden_act="silu",
    use_cache=True,
    pad_token_id=None,
    bos_token_id=0,
    eos_token_id=1,
    mlp_bias=False,
    tie_word_embeddings=False,
    rms_norm_eps=1e-6,
    kv_lora_rank=0,          # use_mla=False path: qk_rope/qk_nope/kv_lora all 0
    q_lora_rank=None,
    initializer_factor=1.0,
)


def _suppress_noise() -> None:
    warnings.filterwarnings("ignore")


def load_reference_config(model_id: str = MID):
    """Build the compat-corrected HF config for Unlimited-OCR (steps 1-4).

    Applies the transformers shims, loads the remote-code config, backfills the
    DeepseekV2Config defaults, standardizes RoPE parameters and forces eager
    attention. Returns the ready-to-use config (no weights loaded).
    """
    _suppress_noise()

    # 1. tt_symbiote compat shims (is_torch_fx_available, ROPE default init).
    from tt_symbiote.utils.hf_compat import install_transformers_shims

    install_transformers_shims()

    from transformers import AutoConfig

    # 2. Load config (custom remote code).
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # 3. Backfill DeepseekV2Config defaults lost in 5.9.0 deserialization. Only
    #    fill genuinely missing / None attrs; never clobber real config values.
    for k, v in _DEEPSEEK_DEFAULTS.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
        elif getattr(cfg, k) is None and v is not None:
            setattr(cfg, k, v)
    # head_dim = hidden_size / num_attention_heads = 1280 / 10 = 128.
    if not getattr(cfg, "head_dim", None):
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads

    # 4. RoPE standardization: LlamaRotaryEmbedding(config=config) reads
    #    config.rope_parameters["rope_type"]; standardize_rope_params() moves
    #    rope_theta into rope_parameters and sets rope_type="default".
    if getattr(cfg, "rope_parameters", None) is None:
        cfg.rope_parameters = None  # ensure attr exists for standardize
    cfg.standardize_rope_params()

    #    Force eager attention: DeepseekV2DecoderLayer selects
    #    ATTENTION_CLASSES["mha_" + config._attn_implementation]; only "mha_eager"
    #    (SlidingWindowLlamaAttention, use_mla=False) exists.
    cfg._attn_implementation = "eager"
    cfg._attn_implementation_internal = "eager"

    return cfg


def load_reference_model(model_id: str = MID, dtype=torch.float32):
    """Instantiate baidu/Unlimited-OCR on CPU. Returns ``(model, config)``.

    Runs :func:`load_reference_config` (steps 1-4), loads the weights (step 5),
    then rematerializes non-persistent buffers left as meta-device garbage by the
    5.9.0 streaming init (step 6, mandatory for correctness).
    """
    _suppress_noise()

    cfg = load_reference_config(model_id)

    from transformers import AutoModel

    # 5. Load weights on CPU (dtype=, NOT torch_dtype=).
    model = AutoModel.from_pretrained(
        model_id,
        config=cfg,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
    )

    # 6. Rematerialize non-persistent buffers computed inside custom submodule
    #    __init__ (LlamaRotaryEmbedding.inv_freq, CLIP position_ids): 5.9.0 builds
    #    on the meta device then streams weights in, leaving these as garbage.
    _rematerialize_buffers(model)

    model.eval()
    return model, cfg


def _rematerialize_buffers(model) -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    for m in model.modules():
        # LlamaRotaryEmbedding.inv_freq (non-persistent, meta-garbage -> NaN cos/sin).
        if isinstance(m, LlamaRotaryEmbedding):
            if m.rope_type == "default":
                inv_freq, scaling = m.compute_default_rope_parameters(m.config)
            else:
                inv_freq, scaling = ROPE_INIT_FUNCTIONS[m.rope_type](m.config)
            m.inv_freq = inv_freq
            m.original_inv_freq = inv_freq.clone()
            m.attention_scaling = scaling
        # CLIPVisionEmbeddings.position_ids buffer (missing from checkpoint).
        if hasattr(m, "position_ids") and hasattr(m, "num_positions"):
            m.position_ids = torch.arange(m.num_positions).expand((1, -1))


__all__ = ["MID", "load_reference_config", "load_reference_model"]
