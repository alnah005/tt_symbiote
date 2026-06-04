# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Base TTNNModule class for TTNN-accelerated neural network modules."""

import functools
import os
from enum import Enum
from functools import wraps
from typing import Dict, Optional, Tuple, Union

import torch

from tt_symbiote.core.run_config import (
    DistributedConfig,
    DistributedTensorConfig,
    get_tensor_run_implementation,
    trace_enabled,
)
from tt_symbiote.core.utils import tree_map

TENSOR_RUN_IMPLEMENTATION = get_tensor_run_implementation()


def set_distributed_tensor_config(distribute_tensor_config: DistributedTensorConfig):
    def _set_distributed_config(e):
        from tt_symbiote.core.tensor import TorchTTNNTensor

        res = e
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
            res.set_distributed_tensor_config(distribute_tensor_config)
        return res

    return _set_distributed_config


def set_module_name_recursively(module, prefix=""):
    """Override children's module names based on this module's name."""
    for name, child in module.__dict__.items():
        if isinstance(child, TTNNModule):
            child._unique_name = f"{prefix}.{name}"
            child.override_children_module_names()
        elif isinstance(child, torch.nn.Module):
            set_module_name_recursively(child, f"{prefix}.{name}")
        elif isinstance(child, dict):
            for k, v in child.items():
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{k}]"
                    v.override_children_module_names()
                elif isinstance(v, torch.nn.Module):
                    set_module_name_recursively(v, f"{prefix}.{name}[{k}]")
        elif isinstance(child, (list, tuple)):
            for i, v in enumerate(child):
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{i}]"
                    v.override_children_module_names()
                elif isinstance(v, torch.nn.Module):
                    set_module_name_recursively(v, f"{prefix}.{name}[{i}]")


class TTNNModule:
    """Base class for TTNN-accelerated modules with automatic fallback to PyTorch."""

    def __init__(self):
        self._device = None  # Device can be set later
        self._preprocessed_weight = False
        self._weights_on_device = False
        self._fallback_torch_layer = None
        self._unique_name = None
        self._device_state: Optional[DistributedConfig] = None
        self._model_config = {}
        self._bypass_tensor_wrapping = False

    def set_model_config(self, model_config):
        """Set model configuration dictionary."""
        self._model_config = model_config if model_config is not None else {}

    def call(self, *args, **kwds):
        return TENSOR_RUN_IMPLEMENTATION.module_run(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.call(*args, **kwds)

    def preprocess_weights(self):
        """Preprocess weights (called once before first use)."""
        from tt_symbiote.core.run_config import _TRACE_RUNNING

        if _TRACE_RUNNING:
            assert (
                self._preprocessed_weight
            ), f"Weights must be preprocessed for {self.module_name} before running traced execution."
            return
        if not self._preprocessed_weight:
            self._preprocessed_weight = True
        else:
            return
        self.preprocess_weights_impl()

    def move_weights_to_device(self):
        """Move preprocessed weights to device."""
        from tt_symbiote.core.run_config import _TRACE_RUNNING

        if _TRACE_RUNNING:
            assert (
                self._weights_on_device
            ), f"Weights must be on device for {self.module_name} before running traced execution."
            return
        assert (
            self._preprocessed_weight
        ), f"Weights must be preprocessed for {self.module_name} before moving to device."
        assert self.device is not None, f"Device must be set for {self.module_name} before moving weights to device."
        if not self._weights_on_device:
            self._weights_on_device = True
        else:
            return
        self.move_weights_to_device_impl()

    def deallocate_weights(self):
        """Deallocate weights from device."""
        self.deallocate_weights_impl()
        self._weights_on_device = False

    def preprocess_weights_impl(self):
        """Override to implement weight preprocessing."""

        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.preprocess_weights()
        return self

    def move_weights_to_device_impl(self):
        """Override to implement weight movement to device."""

        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.move_weights_to_device()
        return self

    def deallocate_weights_impl(self):
        """Override to implement weight deallocation."""
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.deallocate_weights()
        return self

    def to_device(self, device):
        """Set the device for this module."""
        self._device = device
        return self

    def set_device_state(self, device_state: DistributedConfig = None):
        """Set device-specific state for this module."""
        self._device_state = device_state
        if self._device_state is None:
            self._device_state = DistributedConfig(self.device)
        return self

    @property
    def model_config(self):
        """Get model configuration."""
        return self._model_config

    @property
    def device(self):
        """Get current device."""
        return self._device

    @property
    def device_state(self) -> Optional[DistributedConfig]:
        return self._device_state

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        """Create TTNN module from PyTorch module."""
        new_layer = cls(*args, **kwargs)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def train(self, mode: bool = True):
        """Set training mode (TTNN modules don't have training/eval modes)."""
        if self.torch_layer is not None:
            self.torch_layer.train(mode)
        return self

    def set_output_tensors_config(self, output_tensors):
        """Set output tensor configuration based on device state."""
        assert self.device_state is not None
        return self.set_output_tensors_config_impl(output_tensors)

    def set_output_tensors_config_impl(self, output_tensors):
        return tree_map(set_distributed_tensor_config(self.device_state.tensor_config), output_tensors)

    @property
    def module_name(self):
        """Get unique module name."""
        if self._unique_name is None:
            self._unique_name = f"{self.__class__.__name__}_{id(self)}"
        return self._unique_name

    def __repr__(self):
        # recursive representation
        # similar to pytorch nn.Module __repr__
        child_lines = []
        for key, value in self.__dict__.items():
            if isinstance(value, (torch.nn.Module, TTNNModule)) and key != "_fallback_torch_layer":
                mod_str = repr(value).replace("\n", "\n  ")
                child_lines.append(f"({key}): {mod_str}")
        main_str = f"{self.__class__.__name__}(module_name={self.module_name}"
        if child_lines:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str

    @property
    def torch_layer(self):
        """Get fallback PyTorch layer."""
        return self._fallback_torch_layer

    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def pre_trace_execute(self, func_args, func_kwargs):
        """Hook called before ttnn.execute_trace during trace replay.

        Override to copy trace-sensitive inputs to module-owned persistent
        buffers (avoiding TTNN trace allocator buffer aliasing).
        """

    def post_trace_execute(self, func_args, func_kwargs, result):
        """Hook called after ttnn.execute_trace during trace replay.

        Override to perform post-processing on trace outputs or update
        module state based on the replayed result.
        """

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        """Iterator over all modules in the network, yielding both the name of the module as well as the module itself."""
        if memo is None:
            memo = set()
        if remove_duplicate:
            if self in memo:
                return
            memo.add(self)
        yield prefix, self
        for name, child in self.__dict__.items():
            if isinstance(child, (torch.nn.Module, TTNNModule)):
                child_prefix = prefix + ("." if prefix else "") + name
                yield from child.named_modules(memo, child_prefix, remove_duplicate)

    def override_children_module_names(self):
        set_module_name_recursively(self, self.module_name)

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself."""
        for name, child in self.__dict__.items():
            if name in ["_fallback_torch_layer", "torch_layer"]:
                continue
            if isinstance(child, (torch.nn.Module, TTNNModule)):
                yield name, child
            elif isinstance(child, dict):
                for k, v in child.items():
                    if isinstance(v, (torch.nn.Module, TTNNModule)):
                        yield f"{name}[{k}]", v
            elif isinstance(child, (list, tuple)):
                for i, v in enumerate(child):
                    if isinstance(v, (torch.nn.Module, TTNNModule)):
                        yield f"{name}[{i}]", v


@trace_enabled
class TTNNLayerStack(TTNNModule):
    """A trace-enabled stack of TTNNModule layers.

    Captures the entire sequence of layer operations as a single trace.
    During replay, only hidden_states (positional arg) and ttnn.Tensor kwargs
    get ttnn.copy calls — O(1) instead of O(num_layers).

    Calls layer.forward() directly (not layer()) to bypass per-layer
    module_run and trace management.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)
        assert all(isinstance(layer, TTNNModule) for layer in self.layers), "All layers must be TTNNModule instances."

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, **kwargs)
        return hidden_states

    def to_device(self, device):
        super().to_device(device)
        for layer in self.layers:
            if isinstance(layer, TTNNModule):
                layer.to_device(device)
        return self

    def preprocess_weights_impl(self):
        for layer in self.layers:
            if isinstance(layer, TTNNModule):
                layer.preprocess_weights()

    def move_weights_to_device_impl(self):
        for layer in self.layers:
            if isinstance(layer, TTNNModule):
                layer.move_weights_to_device()

    def deallocate_weights_impl(self):
        for layer in self.layers:
            if isinstance(layer, TTNNModule):
                layer.deallocate_weights()


def deallocate_weights_after(func):
    """Decorator to deallocate weights after forward pass."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.deallocate_weights()
        return result

    return wrapper


class DeviceArch(Enum):
    """Supported device architectures."""

    N150 = "n150"
    N300 = "n300"
    T3K = "t3k_wh"
    TG = "gx_wh"
    P150 = "p150"
    P300 = "p300"
    P150x4 = "p150x4"
    P150x8 = "p150x8"
    BHGLX = "bhglx"


MeshShapeToDeviceArch = {
    "N150": DeviceArch.N150,
    "N300": DeviceArch.N300,
    "T3K": DeviceArch.T3K,
    "TG": DeviceArch.TG,
    "P150": DeviceArch.P150,
    "P300": DeviceArch.P300,
    "P150x4": DeviceArch.P150x4,
    "P150x8": DeviceArch.P150x8,
    "BHGLX": DeviceArch.BHGLX,
}

# Ring / mesh collectives used by column-sharded linears (reduce_scatter, all_gather, all_reduce).
# Keep in sync with keys in ``MeshShapeToDeviceArch`` so ``MESH_DEVICE`` always maps to an allowed arch.
SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS = (
    DeviceArch.N150,
    DeviceArch.N300,
    DeviceArch.T3K,
    DeviceArch.TG,
    DeviceArch.P150,
    DeviceArch.P300,
    DeviceArch.P150x4,
    DeviceArch.P150x8,
    DeviceArch.BHGLX,
)


def run_on_devices(
    *allowed_archs: DeviceArch,
    mesh_shape: Optional[Union[Tuple[int, int], Dict[DeviceArch, Tuple[int, int]]]] = None,
):
    """Decorator restricting a TTNNModule method to specific device architectures.

    Args:
        *allowed_archs: ``DeviceArch`` enum values that the module can run on.
        mesh_shape: Optional required mesh shape(s). A mesh shape is inherently
            *per-architecture* -- e.g. a data-parallel layout is ``(8, 1)`` on
            T3K, ``(2, 1)`` on N300, ``(1, 1)`` on N150 -- so when more than one
            arch is allowed you must say which shape goes with which arch.
            Accepted forms:

            * ``None`` (default): no mesh-shape constraint. Backward compatible
              with every existing ``@run_on_devices`` call site.
            * ``(rows, cols)`` tuple: sugar for the single-arch case. The shape
              is required on *every* allowed arch (only meaningful when one arch
              is allowed, or when the same shape genuinely fits all of them).
            * ``{DeviceArch: (rows, cols), ...}`` mapping: the required shape per
              arch. Keys must be a subset of ``allowed_archs``. Allowed archs not
              present in the mapping carry no shape constraint.

            The normalized ``{DeviceArch: (rows, cols)}`` dict is stamped on
            ``wrapper.__tt_mesh_shape__`` for :func:`tt_symbiote.set_device` to
            introspect. At call time only the *active* arch's required shape is
            enforced.

    Raises:
        ValueError: At decoration time, if a ``mesh_shape`` dict key is not in
            ``allowed_archs``.
        RuntimeError: At call time, if the active device's architecture is not
            in ``allowed_archs``, or if the active arch has a required mesh shape
            that the live mesh does not match.

    Example:
        @run_on_devices(DeviceArch.N300, DeviceArch.T3K)
        def forward(self, input_tensor):
            return ttnn.linear(input_tensor, self.tt_weight)

        # Single arch: tuple sugar. Data-parallel 8x1 on T3K.
        @run_on_devices(DeviceArch.T3K, mesh_shape=(8, 1))
        def forward(self, hidden_states, **kwargs):
            ...

        # Multiple archs: per-arch mapping (each arch its own DP shape).
        @run_on_devices(
            DeviceArch.N300, DeviceArch.T3K,
            mesh_shape={DeviceArch.N300: (2, 1), DeviceArch.T3K: (8, 1)},
        )
        def forward(self, hidden_states, **kwargs):
            ...
    """
    if not allowed_archs:
        raise ValueError("Must specify at least one allowed device architecture")

    allowed_set = frozenset(allowed_archs)

    # Normalize mesh_shape into a per-arch dict {DeviceArch: (rows, cols)}.
    if mesh_shape is None:
        required_mesh_shapes: Dict[DeviceArch, Tuple[int, int]] = {}
    elif isinstance(mesh_shape, dict):
        unknown = set(mesh_shape) - allowed_set
        if unknown:
            raise ValueError(
                f"run_on_devices: mesh_shape keys {sorted(a.name for a in unknown)} "
                f"are not in allowed archs {sorted(a.name for a in allowed_set)}"
            )
        required_mesh_shapes = {arch: tuple(int(d) for d in shp) for arch, shp in mesh_shape.items()}
    else:
        # Single (rows, cols) tuple -> required on every allowed arch.
        shp = tuple(int(d) for d in mesh_shape)
        required_mesh_shapes = {arch: shp for arch in allowed_set}

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "device") or self.device is None:
                raise RuntimeError(f"{self.__class__.__name__}: No device set. ")
            mesh_device = MeshShapeToDeviceArch.get(os.environ.get("MESH_DEVICE"))
            if mesh_device is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}: Unable to determine device architecture from MESH_DEVICE environment variable."
                )
            if mesh_device not in MeshShapeToDeviceArch.values():
                raise RuntimeError(
                    f"{self.__class__.__name__}: Unrecognized device architecture {mesh_device} for device {self.device}. Possible options: {list(MeshShapeToDeviceArch.values())}"
                )
            if mesh_device not in allowed_set:
                raise RuntimeError(
                    f"{self.__class__.__name__}: Device architecture {mesh_device} for device {self.device} not supported. "
                    f"Allowed architectures: {allowed_set}"
                )

            required = required_mesh_shapes.get(mesh_device)
            if required is not None:
                actual_shape = getattr(self.device, "shape", None)
                actual = tuple(int(d) for d in actual_shape) if actual_shape is not None else None
                if actual != required:
                    raise RuntimeError(
                        f"{self.__class__.__name__}: on {mesh_device.name} requires mesh shape "
                        f"{required} but the active device mesh is {actual}. "
                        f"Open the device as ttnn.MeshShape{required} for {mesh_device.name} "
                        f"(e.g. for dots.ocr data-parallel set DOTS_OCR_PARALLELISM=DP)."
                    )

            return func(self, *args, **kwargs)

        # Phase 4: stamp the allowed-arch set (and optional per-arch mesh shapes)
        # on the wrapped function so set_device can introspect them without
        # invoking the wrapper.
        wrapper.__tt_allowed_archs__ = allowed_set
        wrapper.__tt_mesh_shape__ = dict(required_mesh_shapes) if required_mesh_shapes else None
        return wrapper

    return decorator
