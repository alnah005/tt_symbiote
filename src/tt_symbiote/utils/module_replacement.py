# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Module-replacement utilities for converting PyTorch modules to TTNN.

The canonical entry point is :func:`register_modules` (Phase 4 rename of
``register_module_replacement_dict``). The old name is retained as a thin
``DeprecationWarning``-emitting alias for one release per
``PROJECT_PROPOSAL.md`` §4.1.
"""

import warnings
from typing import Dict, Optional, Set, Union

from torch import nn

from tt_symbiote.core.module import TTNNModule


def initialize_module(
    old_module, old_class_to_new_class_dict, module_names, model_config, exclude_replacement: Optional[Set[str]] = None
) -> Optional[Union[TTNNModule, nn.Module]]:
    """Initialize a new TTNN module from a PyTorch module."""
    if old_module.__class__ in old_class_to_new_class_dict:
        if old_module in module_names and module_names[old_module] in exclude_replacement:
            return None
        new_module = old_class_to_new_class_dict[old_module.__class__].from_torch(old_module)
        if isinstance(new_module, TTNNModule):
            if old_module in module_names:
                new_module._unique_name = module_names[old_module]
                new_module.override_children_module_names()
            new_module.set_model_config(model_config)
        return new_module
    return None


def register_module_replacement_dict_with_module_names(
    model,
    old_class_to_new_class_dict,
    model_config,
    module_names,
    exclude_replacement: Optional[Set[str]] = None,
    result: Optional[Dict[str, TTNNModule]] = None,
):
    """Recursively replace PyTorch modules with TTNN equivalents."""
    from tt_symbiote.core.module import TTNNModule

    if exclude_replacement is None:
        exclude_replacement = set()
    if result is None:
        result = {}
    assert isinstance(exclude_replacement, set), "exclude_replacement must be a set"
    assert all(isinstance(k, str) for k in exclude_replacement), "All keys in exclude_replacement must be strings"
    if isinstance(model, nn.Module):
        for name, module in model._modules.items():
            if module is None:
                continue
            if module.__class__ in old_class_to_new_class_dict:
                new_module = initialize_module(
                    module, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                )
                if new_module is not None:
                    model._modules[name] = new_module
                    if isinstance(new_module, TTNNModule):
                        result[new_module.module_name] = new_module
            else:
                register_module_replacement_dict_with_module_names(
                    module, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                )
        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(model, attr_name)
            except Exception as e:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            value[k] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
            if isinstance(value, (list, tuple)):
                ls_value = list(value)
                for idx, v in enumerate(ls_value):
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            ls_value[idx] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
                setattr(model, attr_name, type(value)(ls_value))
    elif isinstance(model, TTNNModule):
        for attr_name in dir(model):
            if attr_name.startswith("_") or attr_name in ["torch_layer"]:
                continue
            try:
                value = getattr(model, attr_name)
            except Exception as e:
                continue
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            value[k] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
            if isinstance(value, (list, tuple)):
                ls_value = list(value)
                for idx, v in enumerate(ls_value):
                    if isinstance(v, nn.Module) and v.__class__ in old_class_to_new_class_dict:
                        new_module = initialize_module(
                            v, old_class_to_new_class_dict, module_names, model_config, exclude_replacement
                        )
                        if new_module is not None:
                            ls_value[idx] = new_module
                            if isinstance(new_module, TTNNModule):
                                result[new_module.module_name] = new_module
                    else:
                        register_module_replacement_dict_with_module_names(
                            v, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
                        )
                setattr(model, attr_name, type(value)(ls_value))


def register_modules(
    model,
    old_class_to_new_class_dict,
    model_config=None,
    exclude_replacement: Optional[Set[str]] = None,
) -> Dict[str, TTNNModule]:
    """Replace PyTorch sub-modules of ``model`` with their TTNN equivalents.

    ``old_class_to_new_class_dict`` maps PyTorch module *classes* to TTNN
    wrapper classes. For every PyTorch sub-module whose ``__class__`` matches
    a key, the wrapper's ``from_torch`` classmethod is invoked and the result
    is spliced into the model tree in place. Returns ``{module_name: TTNN
    instance}`` for the modules that were swapped on this call.
    """
    module_names = {module: name for name, module in model.named_modules()}
    result: Dict[str, TTNNModule] = {}
    register_module_replacement_dict_with_module_names(
        model, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
    )
    return result


def register_module_replacement_dict(
    model,
    old_class_to_new_class_dict,
    model_config=None,
    exclude_replacement: Optional[Set[str]] = None,
) -> Dict[str, TTNNModule]:
    """Deprecated alias for :func:`register_modules`.

    Will be removed in the next ``tt_symbiote`` release per
    ``PROJECT_PROPOSAL.md`` §4.1.
    """
    warnings.warn(
        "register_module_replacement_dict is deprecated and will be removed "
        "in the next tt_symbiote release; use register_modules instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return register_modules(
        model,
        old_class_to_new_class_dict,
        model_config=model_config,
        exclude_replacement=exclude_replacement,
    )
