# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Recipe registry that powers :mod:`tt_symbiote.models.auto`.

A *Recipe* is a small object that knows, given a loaded HF model instance,
how to produce the PyTorch -> TTNN module-replacement dict (and any
optional post-replacement model-specific patches). Recipes are registered
at import time using the :func:`register_recipe` decorator and looked up
by Auto* factories via the HF model class name (e.g. ``"BailingMoeV2ForCausalLM"``).

Per ``docs/internal/PROJECT_PROPOSAL.md`` §4.2 this module imports **nothing** from
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
    optionally implement:

    - ``post_register(model)``: any model-specific post-replacement patches
      (runs immediately after ``register_modules`` inside
      :class:`~tt_symbiote.models.auto.auto_factory._BaseAutoModelClass.from_pretrained`).
    - ``make_kv_cache(model, device, **kwargs)``: build and return the
      model-specific KV cache object (e.g. a paged attention cache). Called
      by :func:`tt_symbiote.utils.device_management.set_device` once every
      module has been bound to ``device`` and weights have been moved.
      Resolves Q9 from ``docs/internal/PROJECT_PROPOSAL.md`` (the model owns its KV
      cache, mirroring the ``tt_transformers`` constructor pattern but
      delayed to ``set_device`` time so the device is known).

    :func:`register_recipe` installs a no-op default for any optional hook
    the recipe class does not provide.
    """

    def build_module_dict(self, model: Any) -> Dict[type, type]:  # pragma: no cover - protocol
        ...

    def post_register(self, model: Any) -> None:  # pragma: no cover - protocol
        ...

    # ``make_kv_cache`` is intentionally *not* part of the Protocol body so
    # that the ``runtime_checkable`` ``isinstance`` check stays permissive
    # for the (common) case of a recipe that has no KV cache. The decorator
    # installs a no-op default when the recipe class doesn't define it; the
    # actual call site in :func:`tt_symbiote.utils.device_management.set_device`
    # uses ``hasattr`` rather than ``isinstance`` for the same reason.


TT_MODEL_REGISTRY: Dict[str, Recipe] = {}
"""Global ``{HF class name: Recipe}`` registry consumed by Auto* factories."""


def register_recipe(hf_class_name: str) -> Callable[[type], type]:
    """Class decorator: register a Recipe under the given HF class name.

    Usage::

        @register_recipe(hf_class_name="BailingMoeV2ForCausalLM")
        class BailingMoEV2Recipe:
            def build_module_dict(self, model): ...
            def post_register(self, model): ...                       # optional
            def make_kv_cache(self, model, device, **kwargs): ...     # optional

    The decorator instantiates the class (with no arguments), ensures it
    has both ``post_register`` and ``make_kv_cache`` attributes
    (installing no-op defaults if missing), and inserts the instance into
    :data:`TT_MODEL_REGISTRY`. Re-registering the same name emits a
    ``UserWarning`` and overrides the previous entry.
    """

    if not isinstance(hf_class_name, str) or not hf_class_name:
        raise ValueError("register_recipe requires a non-empty hf_class_name string")

    def decorator(cls):
        instance = cls()
        if not hasattr(instance, "build_module_dict"):
            raise TypeError(f"Recipe {cls.__name__} is missing required method build_module_dict(model)")
        if not hasattr(instance, "post_register"):
            instance.post_register = lambda model: None  # noqa: E731
        if not hasattr(instance, "make_kv_cache"):
            instance.make_kv_cache = lambda model, device, **kwargs: None  # noqa: E731
        if hf_class_name in TT_MODEL_REGISTRY:
            warnings.warn(
                f"Overriding existing tt_symbiote recipe for {hf_class_name!r}",
                stacklevel=2,
            )
        TT_MODEL_REGISTRY[hf_class_name] = instance
        return cls

    return decorator
