# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type

import torch
import ttnn

try:
    from tracy import signpost
except ImportError:
    # tracy is Tenstorrent's profiler. It ships only via a tt-metal build
    # and is not on PyPI, so users who pip-install tt_symbiote cannot
    # resolve it. signpost is only invoked when TT_SYMBIOTE_SIGNPOST_MODE
    # is set in the environment, so a no-op fallback is safe — the
    # default code path never calls it.
    def signpost(*args, **kwargs):  # noqa: D401 — fallback shim, matches tracy API shape
        return None


from tt_symbiote.core.ccl import TT_CCL
from tt_symbiote.core.utils import (
    TORCH_TO_TTNN,
    compare_fn_outputs,
    flat_map_bypass,
    torch_dtype_to_ttnn_dtype,
    tree_map,
    ttnn_dtype_to_torch_dtype,
)


def _source_class_name(module: Any) -> str:
    """Resolve the HF source class name for a TTNNModule for observability hooks.

    Reads ``_fallback_torch_layer`` first so the recorded name matches
    the recipe's ``tt_implemented`` / ``cpu_fallback`` / ``host_glue`` /
    ``out_of_scope`` lists (all of which use HF class names). Falls back
    to the TTNN wrapper's own class name when no fallback layer is
    attached.
    """
    fallback_layer = getattr(module, "_fallback_torch_layer", None)
    if fallback_layer is not None:
        return type(fallback_layer).__name__
    return type(module).__name__


def _record_runtime_fallback(module: Any) -> None:
    """Best-effort runtime hook for :mod:`tt_symbiote.utils.compatibility`.

    Called from every site that issues a "running torch fallback" warning.
    The import is lazy and exceptions are swallowed so observability never
    interferes with the actual fallback path: a broken ledger must not be
    able to bring down inference.
    """
    try:
        from tt_symbiote.utils.compatibility import record_runtime_fallback

        record_runtime_fallback(
            getattr(module, "module_name", type(module).__name__),
            _source_class_name(module),
        )
    except Exception:
        pass


def _record_runtime_success(module: Any) -> None:
    """Best-effort success hook mirroring :func:`_record_runtime_fallback`.

    Fires on the success path of every TTNN-attempting ``Run.module_run``
    so :func:`tt_symbiote.utils.compatibility.report` can distinguish
    "module is wrapped" from "module actually executed on device this
    run". The import is lazy and exceptions are swallowed for the same
    "observability must not break inference" reason as the fallback hook.
    """
    try:
        from tt_symbiote.utils.compatibility import record_runtime_success

        record_runtime_success(
            getattr(module, "module_name", type(module).__name__),
            _source_class_name(module),
        )
    except Exception:
        pass


@dataclass
class CCLManagerConfig:
    """Configuration for CCLManager."""

    mesh_device: Any
    num_links: Optional[int] = None
    topology: Optional[Any] = None

    def __post_init__(self):
        if self.num_links is None:
            self.num_links = 1
        if self.topology is None:
            self.topology = ttnn.Topology.Linear


@dataclass
class DistributedTensorConfig:
    """Configuration for distributed tensor operations."""

    mesh_mapper: Any
    mesh_composer: Any
    logical_shape_fn: Optional[Any] = None

    def get_logical_shape(self, sharded_shape):
        if self.logical_shape_fn is not None:
            return self.logical_shape_fn(sharded_shape)
        return sharded_shape


def logical_shape_for_batch_channel_sharding(mesh_shape):
    def _logical_shape(shape):
        shape = list(shape)
        logical_shape = [shape[0] * mesh_shape[0]] + shape[1:-1] + [shape[-1] * mesh_shape[1]]
        return tuple(logical_shape)

    return _logical_shape


@dataclass
class DistributedConfig:
    """Configuration for distributed operations."""

    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None

    def __post_init__(self):
        if self.tensor_config is None and self.mesh_device.get_num_devices() > 1:
            self.tensor_config = DistributedTensorConfig(
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.mesh_device.shape, (0, -1)),
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, self.mesh_device.shape, (0, -1)),
                logical_shape_fn=logical_shape_for_batch_channel_sharding(self.mesh_device.shape),
            )
        if self.ccl_manager is None and self.mesh_device.get_num_devices() > 1:
            self.ccl_manager = TT_CCL(self.mesh_device)

    def get_tensor_config_for_tensor(self, module_name, tensor):
        if tensor is not None:
            if (
                len(tensor.shape) < 2
                or tensor.shape[-1] % self.mesh_device.shape[-1] != 0
                or tensor.shape[0] % self.mesh_device.shape[0] != 0
            ):
                print(
                    f"Could not determine tensor config for {module_name} with shape {tensor.shape}. Assuming replication to all devices. Override set_output_tensors_config_impl in the module to set the correct config for this tensor."
                )
                return DistributedTensorConfig(
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    mesh_composer=ttnn.create_mesh_composer(
                        self.mesh_device, ttnn.MeshComposerConfig([0, len(tensor.shape)])
                    ),
                )
        return self.tensor_config


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    """Context manager to disable torch dispatch."""
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


def get_empty_torch_tensor_from_ttnn(tt_tensor, dtype=None) -> torch.Tensor:
    """Convert TTNN tensor shape/dtype to empty torch tensor on meta device."""
    ttnn_shape = [int(i) for i in tt_tensor.shape]
    ttnn_dtype = tt_tensor.dtype

    torch_dtype = ttnn_dtype_to_torch_dtype(ttnn_dtype) if dtype is None else dtype
    empty_torch_tensor = torch.empty(ttnn_shape, dtype=torch_dtype, device="meta")
    return empty_torch_tensor


def unwrap_to_torch(func):
    def _unwrap_to_torch(e):
        from tt_symbiote.core.tensor import TorchTTNNTensor

        res = e
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
            res = e.to_torch
        elif isinstance(e, TorchTTNNTensor):
            res = e.to_torch
        elif isinstance(e, torch.Tensor):
            res = e
        elif isinstance(e, ttnn.Tensor):
            res = TorchTTNNTensor(e).to_torch
        return res

    return _unwrap_to_torch


def copy_to_torch(func):
    def _unwrap_to_torch(e):
        from tt_symbiote.core.tensor import TorchTTNNTensor

        res = e
        if isinstance(e, TorchTTNNTensor):
            res = TorchTTNNTensor(e.to_torch.clone())
            res.ttnn_tensor = None
        elif isinstance(e, ttnn.Tensor):
            res = TorchTTNNTensor(e).to_torch
            res.ttnn_tensor = None
        elif isinstance(e, torch.Tensor):
            res = e.clone()
        return res

    return _unwrap_to_torch


def copy_to_ttnn(func):
    def _remove_ttnn_tensor(e):
        from tt_symbiote.core.tensor import TorchTTNNTensor

        res = e
        if isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None:
            res = TorchTTNNTensor(ttnn.from_torch(e.elem.clone()))
            res.ttnn_tensor = ttnn.to_layout(res.to_ttnn, e.ttnn_tensor.layout)
            # TODO: copy memory config without erroring out.
            if e.ttnn_tensor.is_allocated() and e.ttnn_tensor.device() is not None:
                res.ttnn_tensor = ttnn.to_device(res.to_ttnn, e.ttnn_tensor.device())
        return res

    return _remove_ttnn_tensor


def wrap_from_torch(e):
    from tt_symbiote.core.tensor import TorchTTNNTensor

    return TorchTTNNTensor(e) if isinstance(e, torch.Tensor) else e


class DispatchManager:
    timings: Dict[str, Any] = {}
    _modules_in_progress: List[str] = []
    current_module_name: Optional[str] = None
    ENABLED = True

    @staticmethod
    def set_current_module_name(module_name: Optional[str]) -> None:
        if module_name is None:
            assert DispatchManager._modules_in_progress, "No module name to pop"
            DispatchManager._modules_in_progress.pop()
            if DispatchManager._modules_in_progress:
                DispatchManager.current_module_name = DispatchManager._modules_in_progress[-1]
            else:
                DispatchManager.current_module_name = None
        else:
            DispatchManager._modules_in_progress.append(module_name)
            DispatchManager.current_module_name = module_name

    @staticmethod
    def DisableTiming():
        DispatchManager.ENABLED = False

    @staticmethod
    def record_timing(backend: str, module_name: str, func_name: str, attrs: dict, duration: float) -> None:
        if not DispatchManager.ENABLED:
            return
        if backend not in DispatchManager.timings:
            DispatchManager.timings[backend] = {}
        if "TimingEntries" not in DispatchManager.timings:
            DispatchManager.timings["TimingEntries"] = []
        DispatchManager.timings["TimingEntries"].append(
            {
                "attrs": attrs,
                "module_name": module_name,
                "func_name": func_name,
                "duration": duration,
                "backend": backend,
            }
        )

    @staticmethod
    def clear_timings():
        DispatchManager.timings = {}

    @staticmethod
    def get_timing_entries_stats():
        # convert DispatchManager.timings to a dataframe so users can turn into csv
        import pandas as pd

        df = pd.DataFrame(DispatchManager.timings.get("TimingEntries", []))
        return df

    @staticmethod
    def save_stats_to_file(file_name: str):
        assert isinstance(file_name, str), "file_name must be a string"
        assert file_name.endswith(".csv"), "file_name must end with .csv"
        df = DispatchManager.get_timing_entries_stats()
        if df.empty:
            print(f"[WARN] No timing entries recorded. Skipping save to {file_name}")
            return
        df.to_csv(file_name, index=True)
        pivot_table = df.pivot_table(
            index=["func_name", "module_name"], columns="backend", values="duration", aggfunc="sum", fill_value=0
        )
        # Add count of merged rows
        count_table = df.pivot_table(
            index=["func_name", "module_name"], columns="backend", values="duration", aggfunc="count", fill_value=0
        )
        # Add min, max, and average statistics
        min_table = df.pivot_table(
            index=["func_name", "module_name"], columns="backend", values="duration", aggfunc="min"
        )
        max_table = df.pivot_table(
            index=["func_name", "module_name"], columns="backend", values="duration", aggfunc="max", fill_value=0
        )
        columns = pivot_table.columns.tolist()
        pivot_table["Total_Duration"] = pivot_table[columns].sum(axis=1)
        # Add min, max, and average across all backends
        pivot_table["Min_Duration"] = min_table[columns].min(axis=1)
        pivot_table["Max_Duration"] = max_table[columns].max(axis=1)
        # Add total count of rows merged for each index
        pivot_table["Row_Count"] = count_table[columns].sum(axis=1).astype(int)

        # Display or save
        pivot_table.to_csv(file_name.replace(".csv", "_pivot.csv"))
        if "TorchModules" in pivot_table.columns:
            func_times = df.pivot_table(
                index=["func_name"], columns="backend", values="duration", aggfunc="sum", fill_value=0
            )
            module_times = func_times[func_times["TorchModules"] != 0]["TorchModules"].sort_values(ascending=False)
            print(
                "Top 30 Modules by total duration (s):\n\n\n",
                module_times.head(30),
            )
        if "Torch" in pivot_table.columns:
            func_times = df.pivot_table(
                index=["func_name"], columns="backend", values="duration", aggfunc="sum", fill_value=0
            )
            module_times = func_times[func_times["Torch"] != 0]["Torch"].sort_values(ascending=False)
            print(
                "Top 30 Torch Functions by total duration (s):\n\n\n",
                module_times.head(30),
            )
        print(f"Saved timing stats to {os.path.abspath(file_name)}")
        print(f"Saved pivot table to {os.path.abspath(file_name.replace('.csv', '_pivot.csv'))}")


def wrap_to_torch_ttnn_tensor(e):
    from tt_symbiote.core.tensor import TorchTTNNTensor

    result = TorchTTNNTensor(e) if isinstance(e, torch.Tensor) and not isinstance(e, TorchTTNNTensor) else e
    if not isinstance(e, TorchTTNNTensor) and isinstance(e, ttnn.Tensor):
        result = TorchTTNNTensor(e)
    return result


def to_ttnn_wrap(e):
    from tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(e, TorchTTNNTensor):
        e = e.to_ttnn
    return e


def to_ttnn_wrap_keep_torch(e):
    from tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(e, TorchTTNNTensor):
        e.to_ttnn
        return e
    return e


def set_device_wrap(device):
    from tt_symbiote.core.tensor import TorchTTNNTensor

    def _set_device_wrap(e):
        if isinstance(e, ttnn.Tensor) and device is not None and e.device() != device:
            e = ttnn.to_device(e, device)
        elif isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None and e.ttnn_tensor.device() != device:
            e.ttnn_tensor = ttnn.to_device(e.ttnn_tensor, device)
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
            assert e.ttnn_tensor.device() is not None
        return e

    return _set_device_wrap


def create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=False):
    from tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(torch_output, TorchTTNNTensor) and isinstance(ttnn_output, TorchTTNNTensor):
        assert len(torch_output.shape) == len(ttnn_output.shape) and all(
            [s1 == s2 for s1, s2 in zip(torch_output.shape, ttnn_output.shape)]
        ), "Mismatched output shapes between TTNN and Torch."
        assert torch_output.elem is not None, "torch_output.elem is None, cannot assign to ttnn_output."
        torch_output.ttnn_tensor = ttnn_output.to_ttnn
        if not assign_ttnn_to_torch:
            torch_output.elem = None
    elif isinstance(torch_output, (list, tuple)) and isinstance(ttnn_output, (list, tuple)):
        assert len(torch_output) == len(ttnn_output), "Mismatched output lengths between TTNN and Torch."
        for t_item, n_item in zip(torch_output, ttnn_output):
            if isinstance(t_item, TorchTTNNTensor) and isinstance(n_item, TorchTTNNTensor):
                assert len(t_item.shape) == len(n_item.shape) and all(
                    [s1 == s2 for s1, s2 in zip(t_item.shape, n_item.shape)]
                ), "Mismatched output shapes between TTNN and Torch."
                assert t_item.elem is not None, "t_item.elem is None, cannot assign to n_item."
                t_item.ttnn_tensor = n_item.to_ttnn
                if not assign_ttnn_to_torch:
                    t_item.elem = None
    else:
        print("Warning: Mismatched output types between TTNN and Torch in create_new_ttnn_tensors_using_torch_output.")
    return torch_output


def compose_transforms(*transforms):
    """Compose multiple transformation functions into a single pass."""

    def _composed(e):
        result = e
        for transform in transforms:
            result = transform(result)
        return result

    _composed.__name__ = "_".join([t.__name__ for t in transforms])
    return _composed


def fast_unwrap_to_device(device):
    """Lightweight transform: extract ttnn.Tensor and ensure on-device. No TorchTTNNTensor wrapping."""
    from tt_symbiote.core.tensor import TorchTTNNTensor

    def _transform(e):
        if isinstance(e, TorchTTNNTensor):
            t = e.ttnn_tensor if e.ttnn_tensor is not None else e.to_ttnn
            if device is not None and t.device() != device:
                t = ttnn.to_device(t, device)
            return t
        elif isinstance(e, ttnn.Tensor):
            if device is not None and e.device() != device:
                e = ttnn.to_device(e, device)
            return e
        return e

    return _transform


def post_process_ttnn_module_output(self, result):
    post_process_time_begin = time.time()
    result = tree_map(wrap_to_torch_ttnn_tensor, result)
    if self.device_state is not None:
        result = self.set_output_tensors_config(result)
    post_process_time_end = time.time()
    DispatchManager.record_timing(
        "TTNN",
        self.module_name,
        self.__class__.__name__ + "_post_process",
        {},
        post_process_time_end - post_process_time_begin,
    )
    return result


def get_default_distributed_tensor_config(mesh_device=None, torch_tensor=None, module_name=None):
    from tt_symbiote.utils.device_management import DeviceInit

    state = None
    if mesh_device is not None:
        assert (
            DeviceInit.DEVICE_TO_STATE_DICT.get(mesh_device) is not None
        ), f"Device {mesh_device} not found in DeviceInit.DEVICE_TO_STATE_DICT, cannot set distributed config for mesh device."
        state = DeviceInit.DEVICE_TO_STATE_DICT[mesh_device]
    elif DeviceInit.DEVICE_TO_STATE_DICT is not None and len(DeviceInit.DEVICE_TO_STATE_DICT) >= 1:
        state = next(iter(DeviceInit.DEVICE_TO_STATE_DICT.values()))
    if state is None:
        return None
    if torch_tensor is not None:
        return state.get_tensor_config_for_tensor(module_name, torch_tensor)
    return state.tensor_config


class NormalRun:
    verbose = False
    signpost_mode = None

    def __new__(cls, *args, **kwargs):
        raise TypeError("This class cannot be instantiated")

    @staticmethod
    def new_instance(cls, elem, *args, **kwargs):
        from tt_symbiote.core.tensor import TorchTTNNTensor

        delete_elem = False
        ttnn_tensor = None
        if isinstance(elem, ttnn.Tensor):
            ttnn_tensor = elem
            elem = get_empty_torch_tensor_from_ttnn(ttnn_tensor, dtype=kwargs.get("dtype"))
            delete_elem = True
        elif isinstance(elem, torch.Tensor) and not isinstance(elem, TorchTTNNTensor):
            if kwargs.get("dtype") is not None:
                elem = elem.to(dtype=kwargs.get("dtype"))
            if elem.device.type == "meta":
                print("Warning: wrapping meta tensor. This will fail if conversion to TTNN tensor is attempted.")
        output_shape = elem.size()
        strides = elem.stride()
        output_dtype = elem.dtype
        requires_grad = elem.requires_grad
        assert not isinstance(
            elem, TorchTTNNTensor
        ), "Wrapping a TorchTTNNTensor inside another TorchTTNNTensor. This is not allowed."
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            output_shape,
            strides=strides,
            storage_offset=0 if elem.device.type == "meta" else elem.storage_offset(),
            dtype=output_dtype,
            layout=elem.layout,
            device="cpu",
            requires_grad=requires_grad,
        )
        # ...the real tensor is held as an element on the tensor.
        r.ttnn_tensor = ttnn_tensor  # Initialize ttnn_tensor
        r.elem = elem if not delete_elem else None
        distributed_tensor_config = get_default_distributed_tensor_config(
            torch_tensor=elem, module_name=DispatchManager.current_module_name
        )
        r.set_distributed_tensor_config(distributed_tensor_config)
        assert isinstance(r.elem, torch.Tensor) or isinstance(
            ttnn_tensor, ttnn.Tensor
        ), f"elem must be a torch.Tensor (or None when ttnn.Tensor is defined), but got {type(r.elem)}"
        return r

    @staticmethod
    def repr(self):
        return (
            f"TTNNTensor({self.ttnn_tensor.__repr__()})"
            if self.ttnn_tensor is not None
            else f"TorchTensor({self.elem.__repr__()})"
        )

    @staticmethod
    def to_torch(self):
        """Convert to PyTorch tensor."""
        if self.elem is not None and self.elem.device.type != "meta" and self.ttnn_tensor is None:
            return self.elem

        def _to_torch(self):
            is_mesh_device = self.ttnn_distributed_tensor_config is not None
            if is_mesh_device:
                result = ttnn.to_torch(
                    self.ttnn_tensor, mesh_composer=self.ttnn_distributed_tensor_config.mesh_composer
                ).to(self.device, self.dtype)
            else:
                result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
            return result

        result = self.elem
        if self.ttnn_tensor is not None and self.elem is None:
            result = _to_torch(self)
        assert result is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if result.device.type == "meta" and self.ttnn_tensor is not None:
            result = _to_torch(self)
        self.elem = result if self.elem is None else self.elem
        return self.elem

    @staticmethod
    def to_ttnn(self):
        """Convert to TTNN tensor, creating if necessary."""
        if self.ttnn_tensor is not None:
            return self.ttnn_tensor
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if self.elem.device.type == "meta":
            raise RuntimeError(
                "Cannot convert META tensor to TTNN tensor. Please ensure the tensor is on a real device before conversion."
            )
        if self.elem.dtype not in TORCH_TO_TTNN:
            raise RuntimeError(f"Unsupported dtype {self.elem.dtype} for conversion to TTNN tensor.")
        self.ttnn_tensor = ttnn.from_torch(
            self.elem.cpu(),
            dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype),
            mesh_mapper=self.ttnn_distributed_tensor_config.mesh_mapper
            if self.ttnn_distributed_tensor_config
            else None,
            layout=ttnn.TILE_LAYOUT if self.dtype == torch.bool else None,
        )
        return self.ttnn_tensor

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert self.device is not None, (
            f"{self.module_name}: device is not set. "
            f"Call `tt_symbiote.set_device(model, device)` before invoking the model."
        )
        bypass = getattr(self, "_bypass_tensor_wrapping", False)
        if bypass:
            transform = fast_unwrap_to_device(self.device)
        else:
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
        _map = flat_map_bypass if bypass else tree_map
        func_args = _map(transform, args)
        # TODO: fix kwds not being passed correctly
        other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
        func_kwargs = _map(transform, other_kwargs)
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
        begin = time.time()
        self.preprocess_weights()
        end = time.time()
        DispatchManager.set_current_module_name(self.module_name)
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_preprocess_weights", {}, end - begin
        )
        begin = time.time()
        self.move_weights_to_device()
        end = time.time()
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_move_weights_to_device", {}, end - begin
        )
        if NormalRun.signpost_mode is not None:
            signpost(f"{self.module_name}", f"{self.__class__.__name__}")
        begin = time.time()
        try:
            if bypass:
                result = self.forward(*func_args, **func_kwargs)
            else:
                result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
            _record_runtime_success(self)
        except Exception as e:
            if self._fallback_torch_layer is None:
                raise
            warnings.warn(
                f"TTNN forward failed for {self.module_name}: {e!r}; running torch fallback",
                stacklevel=2,
            )
            _record_runtime_fallback(self)
            result = self._fallback_torch_layer(*args, **kwds)
        end = time.time()
        DispatchManager.record_timing("TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin)
        DispatchManager.set_current_module_name(None)
        return result


class LightweightRun(NormalRun):
    """Run mode that always executes via the torch fallback layer.

    Equivalent to the former ``LIGHTWEIGHT`` mode after Phase 3: no TTNN
    forward is attempted; the stored ``_fallback_torch_layer`` carries
    execution end-to-end.
    """

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} (torch fallback)")
        assert (
            self._fallback_torch_layer is not None
        ), f"_fallback_torch_layer must be set on {self.module_name} for LightweightRun."
        return self._fallback_torch_layer(*args, **kwds)


class NormalRunWithFallback(NormalRun):
    """``NormalRun`` with extra forgiveness: a missing device falls back to torch."""

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        if self.device is None:
            assert (
                self._fallback_torch_layer is not None
            ), f"_fallback_torch_layer must be set on {self.module_name} when device is unset."
            warnings.warn(
                f"{self.module_name}: device is not set; running torch fallback. "
                f"Call `tt_symbiote.set_device(model, device)` to enable TTNN execution.",
                stacklevel=2,
            )
            _record_runtime_fallback(self)
            return self._fallback_torch_layer(*args, **kwds)
        bypass = getattr(self, "_bypass_tensor_wrapping", False)
        if bypass:
            transform = fast_unwrap_to_device(self.device)
        else:
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
        func_args = tree_map(transform, args)
        func_kwargs = tree_map(transform, kwds)
        self.preprocess_weights()
        self.move_weights_to_device()
        try:
            if bypass:
                result = self.forward(*func_args, **func_kwargs)
            else:
                result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
            _record_runtime_success(self)
        except Exception as e:
            assert (
                self._fallback_torch_layer is not None
            ), f"_fallback_torch_layer must be set on {self.module_name} for forward fallback."
            warnings.warn(
                f"TTNN forward failed for {self.module_name}: {e!r}; running torch fallback",
                stacklevel=2,
            )
            _record_runtime_fallback(self)
            result = self._fallback_torch_layer(*args, **kwds)
        return result


class SELRun(NormalRun):
    """SEL (Selective comparison) run mode at module granularity.

    Runs the torch fallback and the TTNN forward back-to-back, compares
    outputs via ``compare_fn_outputs``, returns the TTNN result.
    """

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert (
            self._fallback_torch_layer is not None
        ), f"_fallback_torch_layer must be set on {self.module_name} for SELRun."
        copied_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        copied_kwargs = tree_map(copy_to_torch(self.__class__.__name__), kwds)
        torch_args = tree_map(wrap_to_torch_ttnn_tensor, copied_args)
        torch_kwargs = tree_map(wrap_to_torch_ttnn_tensor, copied_kwargs)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self._fallback_torch_layer(*torch_args, **torch_kwargs))
        result = torch_output
        if self.device is not None:
            transform = compose_transforms(to_ttnn_wrap, set_device_wrap(self.device))
            ttnn_args = tree_map(transform, torch_args)
            ttnn_kwargs = tree_map(transform, torch_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            try:
                ttnn_output = post_process_ttnn_module_output(self, self.forward(*ttnn_args, **ttnn_kwargs))
                compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
                result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output)
            except Exception as e:
                warnings.warn(
                    f"TTNN forward failed for {self.module_name}: {e!r}; SELRun returning torch result",
                    stacklevel=2,
                )
        return result


class DPLRun(NormalRun):
    """DPL (Debug Per Layer) run mode at module granularity.

    Runs both the torch fallback and TTNN forward, compares with PCC, and
    returns a torch-output tensor that carries the TTNN buffer (so error
    propagates through subsequent layers).
    """

    @staticmethod
    def module_run(self, *args, **kwds):
        assert (
            self._fallback_torch_layer is not None
        ), f"_fallback_torch_layer must be set on {self.module_name} for DPLRun."
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        copied_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        copied_kwargs = tree_map(copy_to_torch(self.__class__.__name__), kwds)
        torch_args = tree_map(wrap_to_torch_ttnn_tensor, copied_args)
        torch_kwargs = tree_map(wrap_to_torch_ttnn_tensor, copied_kwargs)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self._fallback_torch_layer(*torch_args, **torch_kwargs))
        result = torch_output
        if self.device is not None:
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
            ttnn_args = tree_map(transform, torch_args)
            ttnn_kwargs = tree_map(transform, torch_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            try:
                ttnn_output = post_process_ttnn_module_output(self, self.forward(*ttnn_args, **ttnn_kwargs))
                compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
                result = create_new_ttnn_tensors_using_torch_output(
                    torch_output, ttnn_output, assign_ttnn_to_torch=True
                )
            except Exception as e:
                warnings.warn(
                    f"TTNN forward failed for {self.module_name}: {e!r}; DPLRun returning torch result",
                    stacklevel=2,
                )
        return result


class DPLRunNoErrorProp(NormalRun):
    """DPL variant that does not propagate TTNN numerical drift to subsequent layers.

    Same comparison-at-module-level pattern as :class:`DPLRun`, but the
    TTNN inputs are freshly re-materialized from the torch copies, so the
    comparison is between independent runs rather than a chained one.
    """

    @staticmethod
    def module_run(self, *args, **kwds):
        assert (
            self._fallback_torch_layer is not None
        ), f"_fallback_torch_layer must be set on {self.module_name} for DPLRunNoErrorProp."
        copied_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        copied_kwargs = tree_map(copy_to_torch(self.__class__.__name__), kwds)
        torch_args = tree_map(wrap_to_torch_ttnn_tensor, copied_args)
        torch_kwargs = tree_map(wrap_to_torch_ttnn_tensor, copied_kwargs)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self._fallback_torch_layer(*torch_args, **torch_kwargs))
        result = torch_output
        if self.device is not None:
            independent_ttnn_args = tree_map(copy_to_ttnn(self.__class__.__name__), args)
            independent_ttnn_kwargs = tree_map(copy_to_ttnn(self.__class__.__name__), kwds)
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
            ttnn_args = tree_map(transform, independent_ttnn_args)
            ttnn_kwargs = tree_map(transform, independent_ttnn_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            try:
                ttnn_output = post_process_ttnn_module_output(self, self.forward(*ttnn_args, **ttnn_kwargs))
                compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
                result = create_new_ttnn_tensors_using_torch_output(
                    torch_output, ttnn_output, assign_ttnn_to_torch=True
                )
                print(
                    f"DPLNoErrorPropRun: Done Executing {self.__class__.__name__} from "
                    f"{self.module_name} on device {self.device}"
                )
            except Exception as e:
                warnings.warn(
                    f"TTNN forward failed for {self.module_name}: {e!r}; " f"DPLRunNoErrorProp returning torch result",
                    stacklevel=2,
                )
        return result


class CPU(NormalRun):
    """Run mode that pins execution to the torch fallback path (no TTNN)."""

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on CPU")
        assert (
            self._fallback_torch_layer is not None
        ), f"_fallback_torch_layer must be set on {self.module_name} for CPU run mode."
        func_args = tree_map(wrap_to_torch_ttnn_tensor, args)
        func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, kwds)
        return tree_map(wrap_to_torch_ttnn_tensor, self._fallback_torch_layer(*func_args, **func_kwargs))


# --- Trace Infrastructure ---


def _compute_tensor_signature(tensor) -> Tuple:
    """Compute hashable signature from tensor properties."""
    if isinstance(tensor, ttnn.Tensor):
        return (tuple(tensor.shape), tensor.dtype, tensor.layout)
    if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
        t = tensor.ttnn_tensor
        return (tuple(t.shape), t.dtype, t.layout)
    if isinstance(tensor, torch.Tensor):
        return (tuple(tensor.shape), tensor.dtype)
    return ()


def _compute_args_signature(args) -> Tuple:
    """Compute signature for all tensor args."""
    sigs = []
    for arg in args:
        sig = _compute_tensor_signature(arg)
        if sig:
            sigs.append(sig)
    return tuple(sigs)


@dataclass(slots=True)
class TraceEntry:
    """Single trace cache entry."""

    trace_id: int
    trace_inputs: List[Any]
    trace_kwargs: Dict[str, Any]  # Pre-allocated kwarg tensor buffers
    trace_output: Any
    device: Any


# Registry of trace-enabled classes
_TRACE_ENABLED_CLASSES: Set[Type] = set()
_TRACE_DISABLED_CLASSES: Set[Type] = set()
_TRACE_RUNNING = False


def trace_enabled(cls: Type) -> Type:
    """
    Decorator to mark a TTNNModule subclass as trace-enabled.

    The ``reset_trace_state`` contract is enforced by the class hierarchy, not here: a module
    cannot extend ``TTNNModule`` directly (the direct-subclass ban), so it is always a
    :class:`StatelessTTNNModule` (no-op reset, declared stateless) or a
    :class:`StatefulTTNNModule` (which requires its own ``reset_trace_state`` at class-creation
    time). A @trace_enabled module's ``forward`` is invoked TWICE during trace setup (warm-up
    then capture-record), so ``reset_trace_state`` is the hook a stateful module uses to
    (re)initialize internal state -- see ``TTNNModule.reset_trace_state``.

    Usage:
        @trace_enabled
        class MyModule(StatelessTTNNModule):   # or StatefulTTNNModule (+ reset_trace_state)
            ...
    """
    _TRACE_ENABLED_CLASSES.add(cls)
    return cls


def trace_disabled(cls: Type) -> Type:
    """
    Decorator to mark a TTNNModule subclass as trace-disabled, even if its parent class is trace-enabled.
    """
    _TRACE_DISABLED_CLASSES.add(cls)
    return cls


def is_trace_enabled(module) -> bool:
    """Check if module's class is trace-enabled."""
    return isinstance(module, tuple(_TRACE_ENABLED_CLASSES)) and not isinstance(module, tuple(_TRACE_DISABLED_CLASSES))


class TracedRun(LightweightRun):
    """
    Traced execution mode with automatic caching.
    Only traces modules decorated with @trace_enabled.

    Per-(module, cache_key) three-phase lifecycle:
      1. **Warm-up** (first encounter): Normal forward execution, no trace
         capture. Primes JIT, CCL, and device memory allocator.
      2. **Capture** (second encounter): ``_capture_trace`` records the op
         sequence into a DRAM buffer. The system is already in steady state.
      3. **Replay** (third encounter onward): ``execute_trace`` replays the
         clean trace with near-zero host dispatch overhead.

    ORDERING INVARIANT — replays always follow cold compiles. The warm-up of a
    never-seen ``cache_key`` is a *cold compile* (allocates device buffers). Because a
    cold compile that runs while captured traces exist can corrupt them, any such cold
    compile first discards every captured trace (``invalidate_captures_for_cold_compile``);
    the discarded keys re-capture on their next encounter, after this compile. The cache
    is class-level, so this holds across all ``@trace_enabled`` modules. Steady-state
    fixed-shape loops (e.g. a denoise/decode loop with one key) see no churn; sequential
    multi-key models simply re-capture once per new cold compile and then converge.
    """

    _device: Any = None
    _cq_id: int = 0
    _input_memory_config: Any = None
    _trace_cache: Dict[Tuple, TraceEntry] = {}
    _warmup_keys: Set[Tuple] = set()  # keys that have completed warm-up (run 1)
    _base_pre_trace_execute: Any = None
    _base_post_trace_execute: Any = None

    # --- Allocation-safety seam (reusable by any @trace_enabled model) ---
    # tests/CI set True so device_allocation_barrier RAISES on entry-with-captures
    # (a site entering it with live traces must use the uniform host-only fix instead).
    _strict_alloc_guard: bool = False
    # anti-thrash telemetry: incremented once per barrier release; asserted bounded,
    # reset in configure()/release_all(). MUST NOT scale with token count.
    _barrier_release_count: int = 0
    # Serializes release / capture / replay / barriered-alloc / per-token H2D copies.
    # Single decode-driver thread for dots.ocr serving (supports_async_decode=False forces
    # async_scheduling off; single driver Worker), so this RLock is uncontended -> effectively
    # a re-entrant no-op. It is wired regardless so enabling multi-threaded decode later is safe.
    _trace_lock = threading.RLock()

    @classmethod
    def configure(
        cls,
        device=None,
        cq_id: int = 0,
        input_memory_config=None,
    ) -> None:
        """Configure traced run mode."""
        from tt_symbiote.core.module import TTNNModule

        cls._device = device
        cls._cq_id = cq_id
        cls._input_memory_config = input_memory_config or ttnn.DRAM_MEMORY_CONFIG
        cls._trace_cache = {}
        cls._warmup_keys = set()
        cls._barrier_release_count = 0
        cls._base_pre_trace_execute = TTNNModule.pre_trace_execute
        cls._base_post_trace_execute = TTNNModule.post_trace_execute
        cls._install_alloc_guard()

    @classmethod
    def cache_size(cls) -> int:
        return len(cls._trace_cache)

    @classmethod
    def cached_keys(cls) -> List[Tuple]:
        return list(cls._trace_cache.keys())

    @classmethod
    def release_all(cls) -> None:
        """Release all cached traces AND clear the warm-up bookkeeping.

        ``_warmup_keys`` MUST be cleared alongside the cache: a cache_key is
        ``(module_name, arg_signature)`` and ``module_name`` is ``f"{cls}_{id(self)}"`` --
        Python reuses ``id()`` after GC, so a later module can collide with a stale warm-up
        key whose trace was already released. That collision sends it straight to the CAPTURE
        branch (warm-up skipped), capturing cold -> corrupt replay (observed across back-to-back
        traced tests: e2e PCC ~-0.06). Clearing both keeps the bookkeeping consistent."""
        for entry in cls._trace_cache.values():
            ttnn.release_trace(entry.device, entry.trace_id)
        cls._trace_cache.clear()
        cls._warmup_keys.clear()
        cls._barrier_release_count = 0

    @classmethod
    def release(cls, module_name: str) -> int:
        """Release all traces for a specific module. Returns count released."""
        to_remove = [k for k in cls._trace_cache if k[0] == module_name]
        for key in to_remove:
            entry = cls._trace_cache.pop(key)
            ttnn.release_trace(entry.device, entry.trace_id)
        return len(to_remove)

    @classmethod
    def has_active_captures(cls) -> bool:
        """True iff a captured trace currently exists. Allocating a device buffer while
        this is True risks corrupting those traces (tt-metal allocator.cpp:105). This is
        the framework's "is a trace active" proxy -- ttnn exposes no such query."""
        return bool(cls._trace_cache)

    @classmethod
    def before_device_allocation(cls, reason: str = "") -> int:
        """Release every captured trace IF any exist, so an imminent device-buffer
        allocation that is NOT a @trace_enabled warm-up cannot corrupt a live trace.

        Preserves ``_warmup_keys`` -> released keys RE-CAPTURE on next encounter (NOT
        re-warm: no redundant cold compile). Fast path (no traces live) is a single
        dict-emptiness check -> zero steady-state cost. Returns # released. Thread-safe
        via ``_trace_lock`` (uncontended/no-op when a single thread drives decode)."""
        with cls._trace_lock:
            if not cls._trace_cache:
                return 0
            n = len(cls._trace_cache)
            for entry in cls._trace_cache.values():
                ttnn.release_trace(entry.device, entry.trace_id)
            cls._trace_cache.clear()
            cls._barrier_release_count += 1
        from loguru import logger

        logger.info(
            f"before_device_allocation(reason={reason!r}): released {n} trace(s); "
            f"re-capture on next encounter. If this fires per-token it is a BUG."
        )
        return n

    @classmethod
    @contextlib.contextmanager
    def device_allocation_barrier(cls, reason: str = ""):
        """Make a PER-REQUEST device-buffer allocation safe w.r.t. captured traces.

        HARD RULE: NEVER use this in a per-token / per-decode-step path (catastrophic
        re-capture thrash; destroys the DP=8 perf path). Per-token sites MUST be made
        allocation-free instead (host-only ``from_torch`` + ``copy_host_to_device_tensor``).
        FALLBACK ONLY, for per-request setup that cannot be converted, plus cold-compile.

        strict mode (tests/CI, ``_strict_alloc_guard=True``): RAISE on entry-with-captures --
          the site must use the uniform fix instead.
        production: release captured traces (re-capture afterward; amortized per request)."""
        if cls._strict_alloc_guard and cls.has_active_captures():
            raise RuntimeError(
                f"device_allocation_barrier({reason!r}) entered with "
                f"{len(cls._trace_cache)} active captured trace(s). Use the uniform "
                f"host-only from_torch + copy_host_to_device_tensor fix instead."
            )
        cls.before_device_allocation(reason)
        yield

    @classmethod
    def _install_alloc_guard(cls) -> None:
        """Opt-in allocation tripwire (off by default). Diagnostic/regression sentinel only --
        NOT the completeness oracle (that is ``grep -c allocator.cpp:105 == 0``).

        Modes via ``TT_SYMBIOTE_TRACE_ALLOC_GUARD={off|warn|raise}``. Production default
        ``off`` -> not installed -> zero overhead. The guard fires only when traces are live
        AND not inside a trace warm-up/capture (``_TRACE_RUNNING`` is True during both -- those
        allocations are the legitimate trace buffers and are INTENTIONALLY exempt; do NOT remove
        that check). Idempotent: re-install replaces the prior wrappers' state."""
        mode = os.environ.get("TT_SYMBIOTE_TRACE_ALLOC_GUARD", "off").lower()
        if mode not in ("warn", "raise"):
            return
        if getattr(cls, "_alloc_guard_installed", False):
            return
        from loguru import logger

        # Broadened target set. Skip names absent at the pinned commit.
        target_names = [
            "from_torch", "to_device", "zeros", "allocate_tensor_on_device",
            "allocate_tensor", "to_layout", "reshape", "concat", "add",
        ]

        def _make_wrapper(name, orig):
            def wrapped(*args, **kwargs):
                if cls.has_active_captures() and not _TRACE_RUNNING:
                    stack = "".join(traceback.format_stack(limit=8))
                    msg = (f"TRACE-ALLOC-GUARD: ttnn.{name} allocating while "
                           f"{len(cls._trace_cache)} trace(s) live:\n{stack}")
                    if mode == "raise":
                        raise RuntimeError(msg)
                    logger.warning(msg)
                return orig(*args, **kwargs)
            return wrapped

        installed = []
        for name in target_names:
            orig = getattr(ttnn, name, None)
            if orig is None or not callable(orig):
                continue
            setattr(ttnn, name, _make_wrapper(name, orig))
            installed.append(name)
        cls._alloc_guard_installed = True
        logger.info(f"TRACE-ALLOC-GUARD installed (mode={mode}) on: {installed}")

    @classmethod
    def invalidate_captures_for_cold_compile(cls) -> int:
        """Release EVERY captured trace WITHOUT touching the warm-up bookkeeping.

        INVARIANT ENFORCED: a trace replay must always come AFTER every cold compile.
        A "cold compile" is a WARM-UP encounter (the first eager ``forward`` for a
        never-seen ``cache_key``); it allocates device buffers, and doing so while a
        captured trace exists can corrupt that trace (tt-metal warns: "Allocating
        device buffers is unsafe due to the existence of an active trace ... may be
        corrupted once a trace is executed").

        So when a NEW cold compile is about to run while captured traces exist, every
        captured trace is discarded here. Each discarded key is left in ``_warmup_keys``
        (its device-level program cache is still primed by its original warm-up), so its
        NEXT encounter re-enters the CAPTURE branch directly -- it RE-CAPTURES after this
        cold compile, with no redundant re-warm and no thrash. Net effect: every live
        captured trace was recorded after the most recent cold compile, so all replays
        follow all cold compiles. ``_trace_cache`` is class-level/global, so this holds
        ACROSS modules (one module's cold compile invalidates another's stale captures).

        Returns the number of traces released. Distinct from ``release_all`` (teardown:
        also clears ``_warmup_keys`` to dodge the cross-test ``id()``-reuse collision).

        Refactored to delegate to ``before_device_allocation`` so there
        is ONE invariant-enforcing release path; behavior is preserved (same release of all
        captured traces, ``_warmup_keys`` untouched, returns # released)."""
        return cls.before_device_allocation(reason="cold_compile")

    @staticmethod
    def _assert_no_stateful_descendants(module) -> None:
        """A STATELESS trace unit may NOT contain STATEFUL descendants.

        A ``@trace_enabled`` :class:`StatelessTTNNModule` declares its ``forward`` mutates no
        persistent state under the capture double-run. But the framework invokes
        ``reset_trace_state`` ONLY on the trace unit -- nested modules run with
        ``_TRACE_RUNNING`` set and never get reset. So a Stateful descendant's ``forward`` would
        mutate persistent state TWICE (warm-up + capture) with NO reset, baking a corrupt /
        double-applied mutation into the captured trace. Reject this at trace time.

        Resolution: make the trace unit a :class:`StatefulTTNNModule` whose ``reset_trace_state``
        also resets its stateful descendants, or keep the stateful descendant out of the traced
        subtree (trace it as its own unit / make it eager).
        """
        from tt_symbiote.core.module import StatefulTTNNModule, StatelessTTNNModule

        if not isinstance(module, StatelessTTNNModule):
            return
        offenders = [
            f"{name} [{type(child).__name__}]"
            for name, child in module.named_modules()
            if child is not module and isinstance(child, StatefulTTNNModule)
        ]
        if offenders:
            raise TypeError(
                f"Stateless trace unit {type(module).__name__} ({module.module_name}) is being "
                f"traced but has Stateful descendant(s) {offenders}. A StatelessTTNNModule trace "
                f"unit declares no persistent state, yet the framework resets only the trace unit "
                f"-- a Stateful descendant's forward would mutate state twice (warm-up + capture) "
                f"unreset, corrupting the trace. Make the trace unit a StatefulTTNNModule (whose "
                f"reset_trace_state also resets these descendants), or keep the stateful descendant "
                f"out of the traced subtree."
            )

    @staticmethod
    def _reset_trace_state_tree(module) -> None:
        """Reset the trace unit's own state, then every Stateful descendant's own state.

        The framework resets only the TRACE UNIT, but a Stateful trace unit may legitimately
        contain Stateful descendants (the soundness check permits that; only STATELESS trace units
        with Stateful descendants are rejected). Each module's ``reset_trace_state`` handles ONLY
        its own state; this single top-down walk guarantees every stateful module in the captured
        subtree is reset before each of the two trace-setup forwards, with no double-reset
        (``named_modules`` yields each module once). This is what makes the stateless-with-stateful-
        descendant ban meaningful: once the unit is Stateful, its descendants actually get reset.
        """
        from tt_symbiote.core.module import StatefulTTNNModule

        module.reset_trace_state()
        for _name, child in module.named_modules():
            if child is not module and isinstance(child, StatefulTTNNModule):
                child.reset_trace_state()

    @staticmethod
    def _make_cache_key(module_name: str, args) -> Tuple:
        """Create cache key from module name and input signatures."""
        return (module_name, _compute_args_signature(args))

    @staticmethod
    def _copy_inputs_to_trace_buffer(new_args, trace_inputs) -> None:
        """Copy new inputs to trace input buffers."""
        trace_idx = 0
        for arg in new_args:
            if trace_idx >= len(trace_inputs):
                break
            trace_input = trace_inputs[trace_idx]
            if trace_input is None:
                trace_idx += 1
                continue

            if isinstance(arg, ttnn.Tensor):
                if arg is not trace_input:
                    ttnn.copy(arg, trace_input)
                trace_idx += 1
            elif hasattr(arg, "ttnn_tensor") and arg.ttnn_tensor is not None:
                if arg.ttnn_tensor is not trace_input:
                    ttnn.copy(arg.ttnn_tensor, trace_input)
                trace_idx += 1

    @staticmethod
    def _copy_one_to_trace_buffer(new_val, trace_buf) -> None:
        """Copy a single value into its pre-allocated trace buffer."""
        if isinstance(new_val, ttnn.Tensor):
            if new_val is not trace_buf:
                ttnn.copy(new_val, trace_buf)
        elif hasattr(new_val, "ttnn_tensor") and new_val.ttnn_tensor is not None:
            if new_val.ttnn_tensor is not trace_buf:
                ttnn.copy(new_val.ttnn_tensor, trace_buf)

    @staticmethod
    def _copy_kwargs_to_trace_buffer(new_kwargs, trace_kwargs) -> None:
        """Copy new kwargs to trace kwarg buffers.

        Handles both scalar tensors and list/tuple of tensors (e.g.
        position_embeddings = [cos, sin]).
        """
        for key, trace_buf in trace_kwargs.items():
            if trace_buf is None:
                continue
            new_val = new_kwargs.get(key)
            if new_val is None:
                continue
            if isinstance(trace_buf, (list, tuple)):
                # List/tuple of trace buffers — copy element-wise
                for tb, nv in zip(trace_buf, new_val):
                    if tb is not None:
                        TracedRun._copy_one_to_trace_buffer(nv, tb)
            else:
                TracedRun._copy_one_to_trace_buffer(new_val, trace_buf)

    @staticmethod
    def _capture_trace(module, func_args, func_kwargs, cache_key) -> TraceEntry:
        """Capture trace for module."""
        from loguru import logger

        device = module.device
        cq_id = TracedRun._cq_id
        mem_config = TracedRun._input_memory_config or ttnn.DRAM_MEMORY_CONFIG

        logger.debug(f"Capturing trace for {module.module_name}")

        # Allocate persistent input buffers
        trace_inputs = []
        trace_func_args = []

        for arg_idx, arg in enumerate(func_args):
            if isinstance(arg, ttnn.Tensor):
                host_tensor = arg.cpu() if arg.storage_type() != ttnn.StorageType.HOST else arg
                trace_input = ttnn.to_device(host_tensor, device, memory_config=mem_config)
                trace_inputs.append(trace_input)
                trace_func_args.append(trace_input)
            elif hasattr(arg, "ttnn_tensor") and arg.ttnn_tensor is not None:
                t = arg.ttnn_tensor
                host_tensor = t.cpu() if t.storage_type() != ttnn.StorageType.HOST else t
                trace_input = ttnn.to_device(host_tensor, device, memory_config=mem_config)
                trace_inputs.append(trace_input)
                # Clone the wrapper and set trace input
                from tt_symbiote.core.tensor import TorchTTNNTensor

                new_arg = TorchTTNNTensor(trace_input)
                trace_func_args.append(new_arg)
            else:
                trace_inputs.append(None)
                trace_func_args.append(arg)

        # Pre-allocate persistent keyword argument buffers
        trace_func_kwargs = {}
        trace_kwargs_map = {}  # key -> trace buffer (or list of trace buffers)

        def _alloc_kwarg_tensor(t):
            """Pre-allocate a single device buffer for a kwarg tensor."""
            host = t.cpu() if t.storage_type() != ttnn.StorageType.HOST else t
            return ttnn.to_device(host, device, memory_config=mem_config)

        for key, val in func_kwargs.items():
            if isinstance(val, ttnn.Tensor):
                trace_kwarg = _alloc_kwarg_tensor(val)
                trace_kwargs_map[key] = trace_kwarg
                trace_func_kwargs[key] = trace_kwarg
            elif hasattr(val, "ttnn_tensor") and val.ttnn_tensor is not None:
                trace_kwarg = _alloc_kwarg_tensor(val.ttnn_tensor)
                trace_kwargs_map[key] = trace_kwarg
                from tt_symbiote.core.tensor import TorchTTNNTensor

                trace_func_kwargs[key] = TorchTTNNTensor(trace_kwarg)
            elif isinstance(val, (list, tuple)):
                # Handle list/tuple of tensors (e.g. position_embeddings = [cos, sin])
                bufs = []
                func_vals = []
                has_tensors = False
                for elem in val:
                    if isinstance(elem, ttnn.Tensor):
                        tb = _alloc_kwarg_tensor(elem)
                        bufs.append(tb)
                        func_vals.append(tb)
                        has_tensors = True
                    elif hasattr(elem, "ttnn_tensor") and elem.ttnn_tensor is not None:
                        tb = _alloc_kwarg_tensor(elem.ttnn_tensor)
                        bufs.append(tb)
                        from tt_symbiote.core.tensor import TorchTTNNTensor

                        func_vals.append(TorchTTNNTensor(tb))
                        has_tensors = True
                    else:
                        bufs.append(None)
                        func_vals.append(elem)
                if has_tensors:
                    trace_kwargs_map[key] = bufs
                    trace_func_kwargs[key] = type(val)(func_vals)
                else:
                    trace_func_kwargs[key] = val
            else:
                trace_func_kwargs[key] = val

        # Capture — the output from THIS forward is the trace_output whose
        # device buffer will be rewritten by every subsequent execute_trace.
        # The separate RUN-1 warm-up encounter (tracked in ``_warmup_keys``)
        # already primed caches/ops, so we capture directly with no extra
        # in-capture warm-up forward (an extra forward double-mutates stateful
        # graphs and frees buffers the captured trace references).
        with TracedRun._trace_lock:
            trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
            trace_output = module.forward(*trace_func_args, **trace_func_kwargs)
            ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)
            ttnn.synchronize_device(device)

        entry = TraceEntry(
            trace_id=trace_id,
            trace_inputs=trace_inputs,
            trace_kwargs=trace_kwargs_map,
            trace_output=trace_output,
            device=device,
        )
        TracedRun._trace_cache[cache_key] = entry
        logger.debug(f"Trace cached id={trace_id}, total={TracedRun.cache_size()}")
        return entry

    @staticmethod
    def _replay(self, func_args, func_kwargs, entry, *, capture_encounter):
        """Refresh trace input/kwarg buffers, run the recorded trace, return its output.

        Used by BOTH the steady-state replay branch AND the capture encounter's first
        replay, so "consumed == replay" holds universally. No standalone execute_trace
        lives in the capture branch anymore.

        ``blocking=True`` on the capture encounter (matches the removed hack's deliberate
        choice): ``_capture_trace`` already synchronized after ``end_trace_capture``, but
        this is the FIRST execution of the freshly-recorded trace and its output is
        consumed immediately by the caller (denoise: ttnn.multiply/add; dots_ocr test:
        ttnn.to_torch). A blocking execute guarantees ``trace_output`` is fully written
        before the consumer reads it, removing any CQ-ordering assumption. Steady-state
        replays keep ``blocking=False`` (unchanged perf path).
        """
        pre_trace_begin = time.time()
        TracedRun._copy_inputs_to_trace_buffer(func_args, entry.trace_inputs)
        TracedRun._copy_kwargs_to_trace_buffer(func_kwargs, entry.trace_kwargs)
        pre_trace_end = time.time()
        DispatchManager.record_timing(
            "TTNN",
            self.module_name,
            self.__class__.__name__ + "_pre_trace_copy",
            {},
            pre_trace_end - pre_trace_begin,
        )
        pre_trace_begin = time.time()
        if type(self).pre_trace_execute is not TracedRun._base_pre_trace_execute:
            self.pre_trace_execute(func_args, func_kwargs)
        pre_trace_end = time.time()
        DispatchManager.record_timing(
            "TTNN",
            self.module_name,
            self.__class__.__name__ + "_pre_trace_execute",
            {},
            pre_trace_end - pre_trace_begin,
        )
        with TracedRun._trace_lock:
            ttnn.execute_trace(entry.device, entry.trace_id, cq_id=TracedRun._cq_id, blocking=capture_encounter)
        result = entry.trace_output
        post_trace_begin = time.time()
        if type(self).post_trace_execute is not TracedRun._base_post_trace_execute:
            self.post_trace_execute(func_args, func_kwargs, result)
        post_trace_end = time.time()
        DispatchManager.record_timing(
            "TTNN",
            self.module_name,
            self.__class__.__name__ + "_post_trace_execute",
            {},
            post_trace_end - post_trace_begin,
        )
        return result

    @staticmethod
    def module_run(self, *args, **kwds):
        assert self.device is not None, (
            f"{self.module_name}: device is not set. "
            f"Call `tt_symbiote.set_device(model, device)` before invoking the model."
        )
        # Transform inputs
        bypass = getattr(self, "_bypass_tensor_wrapping", False)
        if bypass:
            transform = fast_unwrap_to_device(self.device)
        else:
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
        _map = flat_map_bypass if bypass else tree_map
        func_args = _map(transform, args)
        other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
        func_kwargs = _map(transform, other_kwargs)
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})

        begin = time.time()
        self.preprocess_weights()
        end = time.time()
        DispatchManager.set_current_module_name(self.module_name)
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_preprocess_weights", {}, end - begin
        )
        begin = time.time()
        self.move_weights_to_device()
        end = time.time()
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_move_weights_to_device", {}, end - begin
        )
        if NormalRun.signpost_mode is not None:
            signpost(f"{self.module_name}", f"{self.__class__.__name__}")

        begin = time.time()
        # Check if this module is trace-enabled
        global _TRACE_RUNNING
        if not is_trace_enabled(self) or _TRACE_RUNNING:
            if _TRACE_RUNNING:
                print(
                    f"{self.__class__.__name__}: {self.module_name} on device {self.device} [Not Trace-Enabled, Already Running Trace Elsewhere, Running Normally]"
                )
            else:
                print(
                    f"{self.__class__.__name__}: {self.module_name} on device {self.device} [Not Trace-Enabled, Running Normally]"
                )
            # Fall back to normal execution
            result = self.forward(*func_args, **func_kwargs)
            end = time.time()
            DispatchManager.record_timing(
                "TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin
            )
            DispatchManager.set_current_module_name(None)
            if bypass:
                return result
            return post_process_ttnn_module_output(self, result)

        # Traced execution path — per-(module, cache_key) lifecycle:
        #   1st encounter: warm-up   -> forward runs eagerly (FORWARD INVOCATION 1 of 2)
        #   2nd encounter: capture   -> forward runs again, RECORDED (FORWARD INVOCATION 2 of 2),
        #                               then the recorded trace is executed once (consumed == replay)
        #   3rd+ encounter: replay   -> execute_trace only (forward NOT re-invoked)
        # NOTE TO MODULE OWNERS: a @trace_enabled module's forward() is invoked TWICE before any
        # pure replay (warm-up then capture-record), and the capture forward's side effects are
        # what the recorded trace replays thereafter. If your forward mutates INTERNAL/PERSISTENT
        # state (KV fill_cache, lazy buffer alloc, counters), override TTNNModule.reset_trace_state()
        # to (re)initialize that state to a clean baseline + PRE-ALLOCATE persistent buffers, so
        # both setup forwards are safe and nothing is lazily allocated DURING capture. The hook is
        # invoked below before each of the two setup forwards; see reset_trace_state.__doc__.
        # The CONSUMED result is ALWAYS a replay -- the capture encounter falls through to the
        # SAME _replay helper as every later encounter, so the un-executed (uninitialized)
        # entry.trace_output is NEVER returned. No standalone execute_trace lives in the
        # capture branch; tracing flows solely via @trace_enabled + this lifecycle.
        cache_key = TracedRun._make_cache_key(self.module_name, func_args)

        if cache_key not in TracedRun._trace_cache and cache_key not in TracedRun._warmup_keys:
            # === RUN 1: WARM-UP (COLD COMPILE; normal forward, no trace) — FORWARD INVOCATION 1 of 2 ===
            # SOUNDNESS: a STATELESS trace unit may not hide STATEFUL descendants -- their
            # forward would mutate state twice (warm-up + capture) unreset. Fail fast, before any
            # release/forward, the first time this module is traced.
            TracedRun._assert_no_stateful_descendants(self)
            # INVARIANT: replays must always follow cold compiles. This cold compile will
            # allocate device buffers; if any traces are already captured, that allocation
            # can corrupt them. Discard every captured trace now so they re-capture AFTER
            # this compile (warm-up bookkeeping is kept -> they re-capture, not re-warm).
            if TracedRun._trace_cache:
                released = TracedRun.invalidate_captures_for_cold_compile()
                print(
                    f"{self.__class__.__name__}: {self.module_name} on device {self.device} "
                    f"[Cold compile for new key -> released {released} captured trace(s); "
                    f"they re-capture after this compile]"
                )
            TracedRun._warmup_keys.add(cache_key)
            _TRACE_RUNNING = True
            print(
                f"{self.__class__.__name__}: {self.module_name} on device {self.device} "
                f"[Warm-up — forward run 1/2 (forward runs AGAIN at capture)]"
            )
            TracedRun._reset_trace_state_tree(self)  # baseline the trace unit + stateful descendants
            result = self.forward(*func_args, **func_kwargs)
            _TRACE_RUNNING = False
            # fall through to shared tail

        elif cache_key not in TracedRun._trace_cache:
            # === RUN 2: CAPTURE (record only; entry inserted into _trace_cache) — FORWARD INVOCATION 2 of 2 ===
            _TRACE_RUNNING = True
            print(
                f"{self.__class__.__name__}: {self.module_name} on device {self.device} "
                f"[Capturing Trace — forward run 2/2; recorded ops replay hereafter]"
            )
            TracedRun._reset_trace_state_tree(self)  # baseline the trace unit + stateful descendants
            begin2 = time.time()
            entry = TracedRun._capture_trace(self, func_args, func_kwargs, cache_key)
            end2 = time.time()
            DispatchManager.record_timing(
                "TTNN", self.module_name, self.__class__.__name__ + "_capture_trace", {}, end2 - begin2
            )
            _TRACE_RUNNING = False
            # The recorded trace_output is UNINITIALIZED. Do NOT consume it. Produce the
            # consumed result via the SAME replay path used by every later encounter, so
            # "consumed == replay" holds universally and the un-executed trace_output is
            # never returned. blocking=True here (this is the trace's first execution and
            # its output is consumed immediately by the caller).
            result = TracedRun._replay(self, func_args, func_kwargs, entry, capture_encounter=True)

        else:
            # === RUN 3+: REPLAY ===
            entry = TracedRun._trace_cache[cache_key]
            print(f"{self.__class__.__name__}: {self.module_name} on device {self.device} [TRACED]")
            result = TracedRun._replay(self, func_args, func_kwargs, entry, capture_encounter=False)

        end = time.time()
        DispatchManager.record_timing("TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin)
        DispatchManager.set_current_module_name(None)
        if bypass:
            return result
        return post_process_ttnn_module_output(self, result)


def disable_trace(fn):
    def new_fn(*args, **kwargs):
        global _TRACE_RUNNING
        was_tracing = _TRACE_RUNNING
        _TRACE_RUNNING = True
        try:
            return fn(*args, **kwargs)
        finally:
            _TRACE_RUNNING = was_tracing

    return new_fn


# Add at module level
_RUN_MODE_REGISTRY = {
    "LIGHTWEIGHT": LightweightRun,
    "NORMAL": NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL": SELRun,
    "DPL": DPLRun,
    "DPL_NO_ERROR_PROP": DPLRunNoErrorProp,
    "CPU": CPU,
    "TRACED": TracedRun,
}

_current_run_mode = None  # Default


def set_run_mode(mode: str) -> None:
    """Set the global run mode. Must be called before any tensor operations."""
    global _current_run_mode
    assert (
        _current_run_mode is None or _current_run_mode == mode
    ), "Run mode has already been set and cannot be changed."
    if mode not in _RUN_MODE_REGISTRY:
        raise ValueError(f"Invalid run mode '{mode}'. Valid modes: {list(_RUN_MODE_REGISTRY.keys())}")
    _current_run_mode = mode


def add_run_mode(mode: str, implementation: Any) -> None:
    """Add a new run mode to the registry."""
    global _RUN_MODE_REGISTRY
    if mode in _RUN_MODE_REGISTRY:
        raise ValueError(f"Run mode '{mode}' already exists.")
    _RUN_MODE_REGISTRY[mode] = implementation


def get_tensor_run_implementation():
    # Environment variable takes precedence for backward compatibility
    global _current_run_mode
    global _RUN_MODE_REGISTRY
    env_mode = os.environ.get("TT_SYMBIOTE_RUN_MODE", _current_run_mode)
    signpost_mode = os.environ.get("TT_SYMBIOTE_SIGNPOST_MODE", None)
    if env_mode is None and _current_run_mode is None:
        _current_run_mode = "NORMAL"
    if env_mode != _current_run_mode and _current_run_mode is not None and env_mode is not None:
        print(
            f"Warning: Run mode from environment variable '{env_mode}' overrides the previously set run mode '{_current_run_mode}'."
        )

    if env_mode is None:
        result = _RUN_MODE_REGISTRY[_current_run_mode]
    else:
        if env_mode not in _RUN_MODE_REGISTRY:
            raise ValueError(
                f"Invalid run mode '{env_mode}' from environment variable. Valid modes: {list(_RUN_MODE_REGISTRY.keys())}"
            )
        result = _RUN_MODE_REGISTRY[env_mode]
    result.signpost_mode = signpost_mode
    return result
