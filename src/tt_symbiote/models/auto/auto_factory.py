# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Base ``Auto*`` factory shared by every ``tt_symbiote.AutoModel*`` class.

Each ``Auto*`` class is a thin wrapper over the corresponding
``transformers.Auto*`` class that, after HF loading,
applies the registered tt_symbiote recipe (if any) and returns the
TTNN-augmented model. When no recipe is registered for the loaded model
class we warn and return the unmodified HF model so that the user-facing
import path never raises ``AttributeError`` for missing classes.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Type

from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY
from tt_symbiote.utils.module_replacement import register_modules
from tt_symbiote.utils.runtime_compat import check_ttnn_compat

__all__ = ["_BaseAutoModelClass", "_BaseAutoBackboneClass"]


class _BaseAutoModelClass:
    """Common ``Auto*`` plumbing.

    Subclasses set ``_HF_AUTO_CLASS`` to the matching ``transformers.AutoModel*``
    class (resolved lazily so importing :mod:`tt_symbiote.models.auto` does not
    require ``transformers`` to be installed at module-import time — only at
    ``from_pretrained`` time).
    """

    _HF_AUTO_CLASS: Optional[Type[Any]] = None

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the HF model, apply the tt_symbiote recipe if one is registered.

        tt_symbiote-specific keyword arguments (popped before the call
        reaches HF's ``transformers.Auto*.from_pretrained``). All of
        these are *configuration decisions* about the loaded model;
        they pair naturally with construction. The downstream
        :func:`tt_symbiote.set_device` call is strictly a binding step
        and accepts only ``(model, mesh_device)``.

        - ``kv_cache_kwargs`` (default ``None``): mapping that the recipe's
          ``make_kv_cache`` hook receives at :func:`set_device` time. The
          cache shape is a model-config decision (capacity, block size,
          batch budget). Stored on the returned model as
          ``model._tt_kv_cache_kwargs``.

          For Ling-mini-2.0 the recipe consumes
          ``{"block_size": 64, "max_num_blocks": 512, "batch_size": 1}``.
          For Gemma-4 / Qwen3-VL / ResNet the recipe's ``make_kv_cache``
          is a no-op (HF ``DynamicCache`` is sufficient), so the kwarg
          is silently ignored.

        - ``dump_visualization`` (default ``False``): if ``True``,
          :func:`set_device` writes ``model_graph.png`` to the cwd at
          the end of binding. Diagnostic feature for recipe authors;
          production users leave this off.

        - ``register_forward_hook`` (default ``False``): if ``True``,
          :func:`set_device` wraps each module's ``forward`` / ``call``
          with timing instrumentation. Diagnostic feature; production
          users leave this off.
        """
        if cls._HF_AUTO_CLASS is None:
            raise NotImplementedError(f"{cls.__name__} has no HF counterpart configured (set _HF_AUTO_CLASS).")

        # Pop tt_symbiote-only kwargs before they reach HF.
        kv_cache_kwargs = kwargs.pop("kv_cache_kwargs", None)
        dump_visualization = bool(kwargs.pop("dump_visualization", False))
        register_forward_hook = bool(kwargs.pop("register_forward_hook", False))

        # Install compat shims before HF's dynamic remote-code loader runs:
        # Hub modeling files authored against older transformers releases
        # frequently import symbols (e.g. ``is_torch_fx_available``) that
        # have since been removed. See ``tt_symbiote/utils/hf_compat.py``.
        from tt_symbiote.utils.hf_compat import install_transformers_shims

        install_transformers_shims()

        model = cls._HF_AUTO_CLASS.from_pretrained(pretrained_name_or_path, *args, **kwargs)

        def _attach_tt_runtime_config(m: Any) -> None:
            # Set on every loaded model regardless of recipe registration —
            # set_device reads these unconditionally and the defaults are
            # the production-safe values.
            m._tt_kv_cache_kwargs = dict(kv_cache_kwargs) if kv_cache_kwargs else {}
            m._tt_dump_visualization = dump_visualization
            m._tt_register_forward_hook = register_forward_hook

        hf_class_name = type(model).__name__
        # Warn (or, under TT_SYMBIOTE_STRICT_TTNN=1, raise) if the installed ttnn's
        # tt-metal commit != the one this recipe was verified against. No-op for
        # models absent from RUNTIME_PINS.
        check_ttnn_compat(hf_class_name)
        recipe = TT_MODEL_REGISTRY.get(hf_class_name)
        if recipe is None:
            warnings.warn(
                f"No tt_symbiote recipe for {hf_class_name!r}; returning unmodified HF model. "
                f"set_device() will be a no-op for this model.",
                stacklevel=2,
            )
            _attach_tt_runtime_config(model)
            return model

        module_dict = recipe.build_module_dict(model)
        register_modules(model, module_dict)
        recipe.post_register(model)
        # Marker read by tt_symbiote.set_device for the hard-error contract.
        model._tt_symbiote_has_recipe = True
        _attach_tt_runtime_config(model)
        return model


class _BaseAutoBackboneClass(_BaseAutoModelClass):
    """Backbone-style Autos share the model-class plumbing in v0.1."""
