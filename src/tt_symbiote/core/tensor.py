# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TorchTTNNTensor: A torch.Tensor subclass that wraps a TTNN tensor.

After Phase 3, the per-op ``__torch_dispatch__`` routing layer was removed
(``docs/internal/PROJECT_PROPOSAL.md`` §6). This class is now a thin container that holds
both a torch view (``elem``) and an optional ``ttnn_tensor`` backing store.
Torch ops fall back to plain torch execution on ``elem`` (no TTNN routing).
Module forwards that need TTNN must call ``ttnn.*`` directly on the
``ttnn_tensor`` attribute.
"""

import warnings
from typing import Optional

import torch

from tt_symbiote.core.run_config import DistributedTensorConfig, get_tensor_run_implementation

TENSOR_RUN_IMPLEMENTATION = get_tensor_run_implementation()

_WARNED_OPS: set = set()


class TorchTTNNTensor(torch.Tensor):
    """torch.Tensor subclass that can carry a TTNN tensor alongside the torch view.

    Per-op TTNN routing is no longer performed here; see module-level
    try/except fallback in :func:`tt_symbiote.core.run_config.NormalRun.module_run`.
    """

    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        return TENSOR_RUN_IMPLEMENTATION.new_instance(cls, elem, *args, **kwargs)

    def __repr__(self):
        return TENSOR_RUN_IMPLEMENTATION.repr(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Always fall back to torch on the unwrapped ``elem`` views.

        Phase 3 removed the dispatcher subsystem. Any torch op on a
        ``TorchTTNNTensor`` now runs through plain torch on the cached
        ``elem`` tensors. The first time a given op name reaches this path
        we emit a one-shot warning so model authors can see that they are
        leaving the TTNN execution path.
        """
        if kwargs is None:
            kwargs = {}

        op_name = func.name() if hasattr(func, "name") else str(func)
        if op_name not in _WARNED_OPS:
            _WARNED_OPS.add(op_name)
            warnings.warn(
                f"TorchTTNNTensor.__torch_dispatch__: {op_name} routed to torch; "
                f"TTNN op-level dispatch was removed in Phase 3. Rewrite the calling "
                f"forward() to call ttnn.* directly if TTNN execution is required.",
                stacklevel=2,
            )

        def _unwrap(e):
            if isinstance(e, TorchTTNNTensor):
                if e.elem is not None:
                    return e.elem
                if e.ttnn_tensor is not None:
                    return TENSOR_RUN_IMPLEMENTATION.to_torch(e)
            return e

        torch_args = tuple(_unwrap(a) for a in args)
        torch_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return func(*torch_args, **torch_kwargs)

    @property
    def shape(self):
        if self.ttnn_distributed_tensor_config is not None and self.ttnn_tensor is not None:
            return self.ttnn_distributed_tensor_config.get_logical_shape(self.ttnn_tensor.shape)
        return self.elem.shape if self.elem is not None else tuple(int(i) for i in self.ttnn_tensor.shape)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __add__(self, other):
        return torch.add(self, other)

    def __radd__(self, other):
        return torch.add(other, self)

    def __abs__(self):
        return torch.abs(self)

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def bool(self):
        if self.ttnn_tensor is not None:
            return TorchTTNNTensor(self.ttnn_tensor, dtype=torch.bool)
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        return TorchTTNNTensor(self.elem.bool())

    @property
    def to_ttnn(self):
        return TENSOR_RUN_IMPLEMENTATION.to_ttnn(self)

    @property
    def to_torch(self):
        return TENSOR_RUN_IMPLEMENTATION.to_torch(self)

    def tolist(self):
        return self.to_torch.tolist()

    def numpy(self):
        return self.to_torch.numpy()

    def clone(self, **kwargs):
        return TorchTTNNTensor(
            self.ttnn_tensor.clone() if self.ttnn_tensor is not None else self.elem.clone(**kwargs), dtype=self.dtype
        )

    def set_distributed_tensor_config(self, distributed_tensor_config: DistributedTensorConfig):
        self._distributed_tensor_config = distributed_tensor_config

    @property
    def ttnn_distributed_tensor_config(self) -> Optional[DistributedTensorConfig]:
        return self.__dict__.get("_distributed_tensor_config", None)
