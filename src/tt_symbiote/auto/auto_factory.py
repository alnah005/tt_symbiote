# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Base ``Auto*`` factory shared by every ``tt_symbiote.AutoModel*`` class.

Per ``PROJECT_PROPOSAL.md`` §4.2 each ``Auto*`` class is a thin wrapper over
the corresponding ``transformers.Auto*`` class that, after HF loading,
applies the registered tt_symbiote recipe (if any) and returns the
TTNN-augmented model. When no recipe is registered for the loaded model
class we warn and return the unmodified HF model so that the user-facing
import path never raises ``AttributeError`` for missing classes.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Type

from tt_symbiote.auto.auto_mappings import TT_MODEL_REGISTRY
from tt_symbiote.utils.module_replacement import register_modules

__all__ = ["_BaseAutoModelClass", "_BaseAutoBackboneClass"]


class _BaseAutoModelClass:
    """Common ``Auto*`` plumbing.

    Subclasses set ``_HF_AUTO_CLASS`` to the matching ``transformers.AutoModel*``
    class (resolved lazily so importing :mod:`tt_symbiote.auto` does not
    require ``transformers`` to be installed at module-import time — only at
    ``from_pretrained`` time).
    """

    _HF_AUTO_CLASS: Optional[Type[Any]] = None

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the HF model, apply the tt_symbiote recipe if one is registered.

        tt_symbiote-specific keyword arguments (popped before the call
        reaches HF's ``transformers.Auto*.from_pretrained``):

        - ``kv_cache_kwargs`` (default ``None``): mapping that the recipe's
          ``make_kv_cache`` hook receives at :func:`set_device` time. The
          cache shape is a model-config decision (capacity, block size,
          batch budget), so it pairs naturally with ``from_pretrained``
          rather than the device-binding call. Stored on the returned
          model as ``model._tt_kv_cache_kwargs``; ``set_device`` reads it
          and applies any per-key override that may also be passed at
          the bind site.

          For Ling-mini-2.0 the recipe consumes
          ``{"block_size": 64, "max_num_blocks": 512, "batch_size": 1}``.
          For Gemma-4 / Qwen3-VL / ResNet the recipe's ``make_kv_cache``
          is a no-op (HF ``DynamicCache`` is sufficient), so the kwarg
          is silently ignored.
        """
        if cls._HF_AUTO_CLASS is None:
            raise NotImplementedError(f"{cls.__name__} has no HF counterpart configured (set _HF_AUTO_CLASS).")

        # Pop tt_symbiote-only kwargs before they reach HF.
        kv_cache_kwargs = kwargs.pop("kv_cache_kwargs", None)

        # Install compat shims before HF's dynamic remote-code loader runs:
        # Hub modeling files authored against older transformers releases
        # frequently import symbols (e.g. ``is_torch_fx_available``) that
        # have since been removed. See ``tt_symbiote/_hf_compat.py``.
        from tt_symbiote._hf_compat import install_transformers_shims

        install_transformers_shims()

        model = cls._HF_AUTO_CLASS.from_pretrained(pretrained_name_or_path, *args, **kwargs)

        hf_class_name = type(model).__name__
        recipe = TT_MODEL_REGISTRY.get(hf_class_name)
        if recipe is None:
            warnings.warn(
                f"No tt_symbiote recipe for {hf_class_name!r}; returning unmodified HF model. "
                f"set_device() will be a no-op for this model.",
                stacklevel=2,
            )
            # Still attach the kv_cache_kwargs in case the user later
            # re-registers a recipe and calls set_device.
            model._tt_kv_cache_kwargs = dict(kv_cache_kwargs) if kv_cache_kwargs else {}
            return model

        module_dict = recipe.build_module_dict(model)
        register_modules(model, module_dict)
        recipe.post_register(model)
        # Marker read by tt_symbiote.set_device for the hard-error contract.
        model._tt_symbiote_has_recipe = True
        # Stash the cache-shape intent for set_device to consume; default
        # to an empty dict so set_device can always splat it unconditionally.
        model._tt_kv_cache_kwargs = dict(kv_cache_kwargs) if kv_cache_kwargs else {}
        return model


class _BaseAutoBackboneClass(_BaseAutoModelClass):
    """Backbone-style Autos share the model-class plumbing in v0.1."""
