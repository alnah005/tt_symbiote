# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims for upstream ``transformers`` API drift.

Hub modeling files loaded via ``trust_remote_code=True`` are pinned to
whatever ``transformers`` release their author was using when the file
was uploaded. When ``tt_symbiote`` pins ``transformers`` to a *newer*
release (per ``docs/development/PROJECT_PROPOSAL.md`` §10), older Hub files may import
symbols that have since been removed or moved.

Each shim below documents exactly one such removal and installs a
replacement when the symbol is missing. :func:`install_transformers_shims`
is called from
:meth:`tt_symbiote.models.auto.auto_factory._BaseAutoModelClass.from_pretrained`
before the HF auto factory triggers the dynamic remote-code import, so
the shims are in place when the Hub file's top-level ``from
transformers...import...`` lines execute.

The shims are idempotent: installing them twice is a no-op.
"""

from __future__ import annotations

__all__ = ["install_transformers_shims"]


_INSTALLED = False


def install_transformers_shims() -> None:
    """Install all known transformers compatibility shims, once per process."""
    global _INSTALLED
    if _INSTALLED:
        return

    import torch
    import transformers.utils.import_utils as iu

    # Shim 1 — ``is_torch_fx_available``.
    #
    # Removed from ``transformers.utils.import_utils`` between 4.x and
    # 5.x. Hub modeling files (e.g.
    # ``inclusionAI/Ling-mini-2.0/modeling_bailing_moe_v2.py``) import
    # it as a feature gate before calling ``torch.fx.wrap`` on
    # ``_prepare_4d_causal_attention_mask`` so the symbolic tracer
    # doesn't unfold it. Re-installing it as a function returning
    # ``hasattr(torch, "fx")`` preserves the original semantics:
    # ``torch.fx`` is part of upstream PyTorch and is always available
    # in modern installs.
    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: hasattr(torch, "fx")

    # Shim 2 — ``ROPE_INIT_FUNCTIONS["default"]``.
    #
    # Between 4.x and 5.x the legacy ``"default"`` entry was dropped
    # from ``transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS``
    # (the unscaled RoPE init now lives inside
    # ``RotaryEmbeddingConfigMixin``). However the docstring at
    # ``modeling_rope_utils.py:648`` still advertises ``"default"`` as a
    # valid key, and older Hub modeling files (e.g.
    # ``inclusionAI/Ling-mini-2.0/modeling_bailing_moe_v2.py:204``)
    # look it up directly: ``ROPE_INIT_FUNCTIONS[self.rope_type]`` with
    # ``self.rope_type == "default"`` whenever ``config.rope_scaling is
    # None``. Re-installing a plain unscaled RoPE entry restores the
    # advertised contract for those files.
    #
    # The implementation is the canonical RoPE formula —
    # ``inv_freq = 1 / base ** (arange(0, dim, 2) / dim)`` — and matches
    # the structure of upstream ``_compute_linear_scaling_rope_parameters``
    # minus the linear ``factor`` step. We read ``rope_theta``,
    # ``head_dim`` and ``partial_rotary_factor`` from the config via
    # ``getattr`` rather than from ``config.rope_parameters`` so this
    # shim works with the *legacy* config shape that the Hub modeling
    # files were authored against (modern transformers funnels everything
    # through ``config.standardize_rope_params()`` which the legacy
    # configs do not satisfy).
    import transformers.modeling_rope_utils as mru

    if "default" not in mru.ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(
            config=None,
            device=None,
            seq_len=None,
            layer_type=None,
        ):
            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            attention_factor = 1.0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            return inv_freq, attention_factor

        mru.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    _INSTALLED = True
