# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Recipe registry that powers :mod:`tt_symbiote.auto`.

A *Recipe* is a small object that knows, given a loaded HF model instance,
how to produce the PyTorch -> TTNN module-replacement dict (and any
optional post-replacement model-specific patches). Recipes are registered
at import time using the :func:`register_recipe` decorator and looked up
by Auto* factories via the HF model class name (e.g. ``"BailingMoeV2ForCausalLM"``).

Per ``PROJECT_PROPOSAL.md`` §4.2 this module imports **nothing** from
``transformers`` so model recipes can populate the registry without a
runtime HF dependency.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Protocol, runtime_checkable

__all__ = ["Recipe", "TT_MODEL_REGISTRY", "register_recipe"]


@runtime_checkable
class Recipe(Protocol):
    """Protocol satisfied by every model recipe.

    Recipe instances expose ``build_module_dict(model)`` which returns a
    ``{torch_class: ttnn_class}`` mapping suitable for
    :func:`tt_symbiote.utils.module_replacement.register_modules`. They may
    optionally implement ``post_register(model)`` to perform any
    model-specific post-replacement patches. :func:`register_recipe`
    installs a no-op ``post_register`` if the recipe class does not provide
    one.
    """

    def build_module_dict(self, model: Any) -> Dict[type, type]:  # pragma: no cover - protocol
        ...

    def post_register(self, model: Any) -> None:  # pragma: no cover - protocol
        ...


TT_MODEL_REGISTRY: Dict[str, Recipe] = {}
"""Global ``{HF class name: Recipe}`` registry consumed by Auto* factories."""


def register_recipe(hf_class_name: str) -> Callable[[type], type]:
    """Class decorator: register a Recipe under the given HF class name.

    Usage::

        @register_recipe(hf_class_name="BailingMoeV2ForCausalLM")
        class BailingMoEV2Recipe:
            def build_module_dict(self, model): ...
            def post_register(self, model): ...   # optional

    The decorator instantiates the class (with no arguments), ensures it
    has a ``post_register`` attribute (installing a no-op if missing),
    and inserts the instance into :data:`TT_MODEL_REGISTRY`. Re-registering
    the same name emits a ``UserWarning`` and overrides the previous entry.
    """

    if not isinstance(hf_class_name, str) or not hf_class_name:
        raise ValueError("register_recipe requires a non-empty hf_class_name string")

    def decorator(cls):
        instance = cls()
        if not hasattr(instance, "build_module_dict"):
            raise TypeError(
                f"Recipe {cls.__name__} is missing required method build_module_dict(model)"
            )
        if not hasattr(instance, "post_register"):
            instance.post_register = lambda model: None  # noqa: E731
        if hf_class_name in TT_MODEL_REGISTRY:
            warnings.warn(
                f"Overriding existing tt_symbiote recipe for {hf_class_name!r}",
                stacklevel=2,
            )
        TT_MODEL_REGISTRY[hf_class_name] = instance
        return cls

    return decorator
