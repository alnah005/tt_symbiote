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
        """Load the HF model, apply the tt_symbiote recipe if one is registered."""
        if cls._HF_AUTO_CLASS is None:
            raise NotImplementedError(f"{cls.__name__} has no HF counterpart configured (set _HF_AUTO_CLASS).")

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
            return model

        module_dict = recipe.build_module_dict(model)
        register_modules(model, module_dict)
        recipe.post_register(model)
        # Marker read by tt_symbiote.set_device for the hard-error contract.
        model._tt_symbiote_has_recipe = True
        return model


class _BaseAutoBackboneClass(_BaseAutoModelClass):
    """Backbone-style Autos share the model-class plumbing in v0.1."""
