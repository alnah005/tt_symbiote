# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device management utilities for TTNN modules.

The single public entry point is :func:`set_device`. It is the mandatory
final step of the ``tt_symbiote`` loading flow and does six things in order:

1. Walks the model graph.
2. For every ``TTNNModule``, reads ``forward.__tt_allowed_archs__``. If the
   active device architecture (resolved from ``MESH_DEVICE``) is not in the
   allowed set, the module is swapped in place with its
   ``_fallback_torch_layer`` and a warning is logged.
3. Calls ``to_device(device)`` and (for multi-device meshes)
   ``set_device_state(...)`` on every remaining TTNN module.
4. Calls ``preprocess_weights()`` then ``move_weights_to_device()`` on every
   visited TTNN module (subsumes the explicit per-test loop that callers
   previously wrote by hand).
5. If a recipe is registered for ``type(obj).__name__`` and exposes
   ``make_kv_cache``, builds the model-specific KV cache and attaches it as
   ``obj._tt_kv_cache``.
   The kwargs passed to ``make_kv_cache`` come from
   ``obj._tt_kv_cache_kwargs`` (set by ``AutoModel*.from_pretrained``'s
   ``kv_cache_kwargs=``). There is no bind-site override: cache shape
   is a model-construction decision and pairs with ``from_pretrained``.
6. Sets ``_tt_symbiote_device_set = True`` on the root object and on every
   visited TTNN module.

Hard-error enforcement: ``run_config.module_run`` asserts
``self._device is not None`` with a message that names ``set_device``, so a
forward called before ``set_device`` fails with a clear diagnostic.
"""

import functools
import os
import time
import warnings
from typing import Any, Optional

from torch import nn

from tt_symbiote.core.module import MeshShapeToDeviceArch, TTNNModule
from tt_symbiote.core.run_config import DispatchManager, DistributedConfig
from tt_symbiote.utils.graph_visualization import draw_model_graph

__all__ = ["DeviceInit", "set_device"]


class DeviceInit:
    DEVICE_TO_STATE_DICT: dict = {}

    @classmethod
    def init_state(cls, device) -> Optional[DistributedConfig]:
        """Initialize device state if not already initialized."""
        if device not in cls.DEVICE_TO_STATE_DICT:
            res = cls.init_state_impl(device)
            if res is not None:
                assert isinstance(res, DistributedConfig), f"Expected DistributedConfig, got {type(res)}"
            cls.DEVICE_TO_STATE_DICT[device] = res
        return cls.DEVICE_TO_STATE_DICT[device]

    @classmethod
    def init_state_impl(cls, device) -> DistributedConfig:
        """Implementation-specific device state initialization."""
        return DistributedConfig(device)


def _initialize_module_on_device(module: "TTNNModule", device, device_init=DeviceInit) -> None:
    """Bind a TTNN module to ``device`` and (for meshes) set its distributed config."""
    module.to_device(device)
    if device.get_num_devices() > 1:
        module.set_device_state(device_init.init_state(device))


def timed_call(original_call, module_name, module_class):
    """Wrap ``forward``/``call`` with the legacy timing instrumentation."""

    @functools.wraps(original_call)
    def new_call(*args, **kwargs):
        begin = time.time()
        DispatchManager.set_current_module_name(module_name)
        result = original_call(*args, **kwargs)
        DispatchManager.set_current_module_name(None)
        end = time.time()
        DispatchManager.record_timing("TorchModules", module_name, module_class, {}, end - begin)
        return result

    return new_call


def _active_device_arch() -> Any:
    """Resolve the active device architecture from ``MESH_DEVICE``.

    Returns the matching :class:`DeviceArch` enum value or ``None`` if the
    environment variable is unset or unrecognized. ``set_device`` uses this
    to decide whether each ``@run_on_devices``-stamped module is supported.
    """
    mesh = os.environ.get("MESH_DEVICE")
    if mesh is None:
        return None
    return MeshShapeToDeviceArch.get(mesh)


def _module_allowed_archs(module: TTNNModule):
    """Return ``__tt_allowed_archs__`` stamped on the module's ``forward``, or ``None``."""
    forward = getattr(type(module), "forward", None)
    if forward is None:
        return None
    return getattr(forward, "__tt_allowed_archs__", None)


def _is_arch_supported(module: TTNNModule) -> bool:
    """``True`` if the module has no arch restriction *or* the active arch is allowed."""
    allowed = _module_allowed_archs(module)
    if allowed is None:
        return True
    active = _active_device_arch()
    if active is None:
        # No MESH_DEVICE set; preserve current behavior and let the
        # call-time @run_on_devices check raise if it actually runs.
        return True
    return active in allowed


def _swap_module(parent: Any, key: Any, fallback: Any) -> None:
    """Replace the child at ``parent[key]`` (or attribute) with ``fallback``.

    Used when a ``@run_on_devices`` proactive check rejects a TTNN module
    on the current arch. The placement of ``key`` depends on the parent's
    container type:

    - ``nn.Module._modules`` slot (string name)
    - generic ``__dict__`` attribute (string name; via ``setattr``)
    - ``dict`` value (any hashable key)
    - ``list`` index (int)

    Tuples are handled by the caller (they require rebuilding the tuple).
    """
    if isinstance(parent, nn.Module) and isinstance(key, str) and key in parent._modules:
        parent._modules[key] = fallback
        return
    if isinstance(parent, dict):
        parent[key] = fallback
        return
    if isinstance(parent, list):
        parent[key] = fallback
        return
    if isinstance(key, str):
        try:
            setattr(parent, key, fallback)
            return
        except Exception:
            pass


def set_device(obj, device) -> None:
    """Bind every ``TTNNModule`` in ``obj`` to ``device``.

    This is **mandatory** before any model invocation. See the module
    docstring for the full contract.

    Strict two-argument signature. All runtime / diagnostic
    configuration is a model-construction decision and belongs on
    :meth:`tt_symbiote.AutoModel*.from_pretrained`. This function
    reads three configuration attributes that ``from_pretrained``
    attaches to the model (all default to the production-safe value
    so the function still works on hand-constructed ``TTNNModule``
    instances that bypass ``from_pretrained``):

      - ``obj._tt_register_forward_hook`` (default ``False``): if
        ``True``, every module's ``forward`` / ``call`` is wrapped
        with timing instrumentation.
      - ``obj._tt_dump_visualization`` (default ``False``): if
        ``True``, writes ``model_graph.png`` to the cwd at the end of
        binding.
      - ``obj._tt_kv_cache_kwargs`` (default ``{}``): forwarded
        verbatim to the recipe's ``make_kv_cache`` hook.

    The swapped-class registry consumed by
    :func:`tt_symbiote.utils.compatibility.report` is cleared at entry
    and repopulated at exit so it always mirrors the *current* model
    tree (running two demos in the same process never aliases). Runtime
    observation ledgers (success / fallback) are left intact — callers
    that want a clean slate call ``reset_runtime_observations()``
    explicitly.
    """
    # Read runtime config from the model (attached by from_pretrained).
    # All three flags default to the production-safe value when absent,
    # which is the case for hand-constructed TTNNModule instances in
    # tests/auto/test_set_device.py and the per-model test trees
    # (tests/models/* and tests/experimental/*).
    register_forward_hook = bool(getattr(obj, "_tt_register_forward_hook", False))
    dump_visualization = bool(getattr(obj, "_tt_dump_visualization", False))
    device_init = DeviceInit  # never overridden anywhere in-tree

    try:
        from tt_symbiote.utils.compatibility import reset_swapped_registry

        reset_swapped_registry()
    except Exception:
        pass

    initialized_modules: list = []  # collected for the weight-prep pass

    # Build module name mapping before recursion
    module_names = {}
    if isinstance(obj, nn.Module):
        module_names = {module: name for name, module in obj.named_modules()}

    def _bind(child: TTNNModule, parent: Any, key: Any) -> Optional[Any]:
        """Decide TTNN-vs-fallback, mutate parent on swap, return the now-in-place child."""
        if not _is_arch_supported(child):
            fallback = child._fallback_torch_layer
            if fallback is None:
                warnings.warn(
                    f"{child.module_name}: device arch unsupported and no "
                    f"_fallback_torch_layer available; leaving TTNN module in place.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Running {child.module_name} on CPU; " f"not supported on {_active_device_arch()}",
                    stacklevel=2,
                )
                _swap_module(parent, key, fallback)
                return fallback
        _initialize_module_on_device(child, device, device_init)
        child._tt_symbiote_device_set = True
        initialized_modules.append(child)
        return child

    def _set_device_recursive(current_obj, parent_is_ttnn: bool = False) -> None:
        if isinstance(current_obj, nn.Module):
            name = module_names.get(current_obj, "")

            if register_forward_hook:
                if hasattr(current_obj, "forward"):
                    if not hasattr(current_obj.forward, "_is_timed"):
                        current_obj.forward = timed_call(current_obj.forward, name, current_obj.__class__.__name__)
                        current_obj.forward._is_timed = True

            # _modules children
            for child_name, module in list(current_obj._modules.items()):
                if module is None:
                    continue
                if isinstance(module, TTNNModule):
                    bound = _bind(module, current_obj, child_name)
                    if bound is not module:
                        # swapped to fallback; recurse into the fallback to bind any
                        # nested TTNN modules it owns.
                        _set_device_recursive(bound, parent_is_ttnn=False)
                        continue
                _set_device_recursive(module, parent_is_ttnn=isinstance(current_obj, TTNNModule))

            # public attrs containing TTNN modules / dicts / lists / tuples
            for attr_name in dir(current_obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    value = getattr(current_obj, attr_name)
                except Exception:
                    continue
                if isinstance(value, TTNNModule):
                    bound = _bind(value, current_obj, attr_name)
                    if bound is not value:
                        _set_device_recursive(bound, parent_is_ttnn=False)
                        continue
                    _set_device_recursive(value, parent_is_ttnn=isinstance(current_obj, TTNNModule))
                elif isinstance(value, dict):
                    for k, v in list(value.items()):
                        if isinstance(v, TTNNModule):
                            bound = _bind(v, value, k)
                            if bound is not v:
                                _set_device_recursive(bound, parent_is_ttnn=False)
                                continue
                        _set_device_recursive(v, parent_is_ttnn=isinstance(current_obj, TTNNModule))
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, TTNNModule):
                            bound = _bind(v, value, i)
                            if bound is not v:
                                _set_device_recursive(bound, parent_is_ttnn=False)
                                continue
                        _set_device_recursive(v, parent_is_ttnn=isinstance(current_obj, TTNNModule))
                elif isinstance(value, tuple):
                    # Tuples are immutable; rebuild if any element was a TTNN module
                    # whose arch fails and is swapped.
                    new_value = list(value)
                    mutated = False
                    for i, v in enumerate(new_value):
                        if isinstance(v, TTNNModule):
                            bound = _bind(v, new_value, i)
                            if bound is not v:
                                mutated = True
                                _set_device_recursive(bound, parent_is_ttnn=False)
                                continue
                        _set_device_recursive(v, parent_is_ttnn=isinstance(current_obj, TTNNModule))
                    if mutated:
                        try:
                            setattr(current_obj, attr_name, tuple(new_value))
                        except Exception:
                            pass
        elif isinstance(current_obj, TTNNModule):
            if not getattr(current_obj, "_bypass_tensor_wrapping", False):
                current_obj._bypass_tensor_wrapping = parent_is_ttnn
            if register_forward_hook and hasattr(current_obj, "call"):
                if not hasattr(current_obj.call, "_is_timed"):
                    current_obj.call = timed_call(
                        current_obj.call, current_obj.module_name, current_obj.__class__.__name__
                    )
                    current_obj.call._is_timed = True
            for attr_name in dir(current_obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    value = getattr(current_obj, attr_name)
                except Exception:
                    continue
                if isinstance(value, (nn.Module, TTNNModule)):
                    if isinstance(value, TTNNModule):
                        bound = _bind(value, current_obj, attr_name)
                        if bound is not value:
                            _set_device_recursive(bound, parent_is_ttnn=False)
                            continue
                    _set_device_recursive(value, parent_is_ttnn=True)
                elif isinstance(value, dict):
                    for k, v in list(value.items()):
                        if isinstance(v, TTNNModule):
                            bound = _bind(v, value, k)
                            if bound is not v:
                                _set_device_recursive(bound, parent_is_ttnn=False)
                                continue
                        _set_device_recursive(v, parent_is_ttnn=True)
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, TTNNModule):
                            bound = _bind(v, value, i)
                            if bound is not v:
                                _set_device_recursive(bound, parent_is_ttnn=False)
                                continue
                        _set_device_recursive(v, parent_is_ttnn=True)
                elif isinstance(value, tuple):
                    new_value = list(value)
                    mutated = False
                    for i, v in enumerate(new_value):
                        if isinstance(v, TTNNModule):
                            bound = _bind(v, new_value, i)
                            if bound is not v:
                                mutated = True
                                _set_device_recursive(bound, parent_is_ttnn=False)
                                continue
                        _set_device_recursive(v, parent_is_ttnn=True)
                    if mutated:
                        try:
                            setattr(current_obj, attr_name, tuple(new_value))
                        except Exception:
                            pass

    # Root case: if the root is itself a TTNNModule, bind it directly.
    if isinstance(obj, TTNNModule):
        if not _is_arch_supported(obj):
            warnings.warn(
                f"Root {obj.module_name}: device arch unsupported. "
                f"Cannot swap root in place; call site should pass the fallback "
                f"layer instead.",
                stacklevel=2,
            )
        else:
            _initialize_module_on_device(obj, device, device_init)
            obj._tt_symbiote_device_set = True
            initialized_modules.append(obj)
    _set_device_recursive(obj)

    # Subsume the explicit preprocess_weights / move_weights_to_device loop
    # that callers would otherwise write by hand after every set_device call.
    for module in initialized_modules:
        try:
            module.preprocess_weights()
            module.move_weights_to_device()
        except Exception as e:
            warnings.warn(
                f"set_device: failed to (preprocess|move) weights for " f"{module.module_name}: {e!r}",
                stacklevel=2,
            )

    # If a recipe is registered for this model, give it a chance to
    # allocate model-specific state that requires a live device,
    # most notably the paged-attention KV cache. Mirrors the
    # ``tt_transformers`` "model owns its KV cache" pattern but delayed to
    # set_device time (since HF builds the model on CPU first). The cache
    # is attached as ``model._tt_kv_cache`` and the test/demo code passes
    # it back in as ``past_key_values=`` for ``model.generate``.
    #
    # The kv-cache shape is declared at ``from_pretrained`` time via
    # ``kv_cache_kwargs=`` and stashed on the model as
    # ``model._tt_kv_cache_kwargs``. ``set_device`` reads it here and
    # forwards it verbatim to the recipe. There is intentionally no
    # bind-site override: cache shape is a model-construction decision
    # (one of the things that makes the model what it is), so it pairs
    # with the construction call, not the binding call.
    try:
        from tt_symbiote.models.auto.auto_mappings import TT_MODEL_REGISTRY
    except Exception:
        TT_MODEL_REGISTRY = {}
    recipe = TT_MODEL_REGISTRY.get(type(obj).__name__)
    if recipe is not None and hasattr(recipe, "make_kv_cache"):
        stored_kv_kwargs = getattr(obj, "_tt_kv_cache_kwargs", None) or {}
        try:
            kv = recipe.make_kv_cache(obj, device, **stored_kv_kwargs)
            if kv is not None:
                obj._tt_kv_cache = kv
        except Exception as e:
            warnings.warn(
                f"set_device: make_kv_cache failed for {type(obj).__name__}: {e!r}",
                stacklevel=2,
            )

    # Root marker, read by code that wants to check "did the user call set_device?"
    try:
        setattr(obj, "_tt_symbiote_device_set", True)
    except Exception:
        pass

    # Single post-walk to populate the swapped-class registry that
    # ``compatibility.report`` reads. Every TTNNModule still present in the
    # tree after the bind/arch pass had ``_swap_module`` decline to replace it
    # — i.e. it is *actually* about to execute on device. We record the HF
    # source class (the type of ``_fallback_torch_layer``) so the registry
    # cross-references cleanly with the recipe's design-time lists. TTNN
    # modules without a fallback layer are skipped: their wrapper class name
    # is not meaningful for HF-side reporting.
    try:
        from tt_symbiote.utils.compatibility import record_swapped_class

        if isinstance(obj, nn.Module):
            iterator = obj.named_modules()
        elif isinstance(obj, TTNNModule):
            iterator = [(getattr(obj, "module_name", ""), obj)]
        else:
            iterator = []
        for module_name, module in iterator:
            if not isinstance(module, TTNNModule):
                continue
            fallback = getattr(module, "_fallback_torch_layer", None)
            if fallback is None:
                continue
            record_swapped_class(
                module_name or getattr(module, "module_name", type(module).__name__),
                type(fallback).__name__,
            )
    except Exception as e:
        warnings.warn(
            f"set_device: failed to populate swapped-class registry: {e!r}",
            stacklevel=2,
        )

    if dump_visualization:
        draw_model_graph(obj)
