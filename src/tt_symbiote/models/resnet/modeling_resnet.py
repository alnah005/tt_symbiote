# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of HuggingFace's ``ResNetForImageClassification`` family.

Covers all five canonical Microsoft variants
(``microsoft/resnet-{18,34,50,101,152}``) through a single recipe. The
``(depths, layer_type)`` triplet on :class:`ResNetConfig` is the only
inter-variant axis: depths drive the stage counts in the unchanged HF
:class:`ResNetEncoder`, and ``layer_type`` selects between
:class:`TTNNResNetBasicLayer` (2-conv, used by resnet-18/34) and
:class:`TTNNResNetBottleNeckLayer` (3-conv, used by resnet-50/101/152).
A per-variant TTNN tuning table (slice strategy, ``l1_small_size``)
lives in :mod:`.configuration_resnet`.

Phase 5 / Option 1 contract recap: each TTNN wrapper class owns the
conversion of its own subtree inside ``from_torch``. The recipe
describes only the flat top-level mapping
(:meth:`ResNetRecipe.build_module_dict`) — the walker is invoked once
and each ``from_torch`` rebuilds its block's children in NHWC layout.

NHWC convention
---------------

HF gives us pixel tensors in NCHW. TTNN convs are NHWC. The permute
happens at the network boundary inside :class:`TTNNResNetEmbeddings`
(stem) and is undone right before the final ``nn.AdaptiveAvgPool2d`` /
``nn.Linear`` classifier head (which still run in HF's NCHW host
fallback). Every TTNN wrapper between those two points expects and
returns NHWC.
"""

from __future__ import annotations

import torch
import ttnn
from torch import nn
from transformers.models.resnet.modeling_resnet import (
    ResNetBasicLayer,
    ResNetBottleNeckLayer,
    ResNetConvLayer,
    ResNetEmbeddings,
    ResNetShortCut,
)

from tt_symbiote.core.module import TTNNModule, SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS, run_on_devices
from tt_symbiote.core.run_config import trace_enabled
from tt_symbiote.models.auto.auto_mappings import register_recipe
from tt_symbiote.models.resnet.configuration_resnet import lookup_ttnn_tuning
from tt_symbiote.modules.ttnn_activation import TTNNReLU
from tt_symbiote.modules.ttnn_conv import TTNNConv2dBNActivationNHWC, TTNNConv2dBNNHWC, TTNNMaxPool2dNHWC
from tt_symbiote.modules.ttnn_linear import TTNNLinear  # noqa: F401 — re-exported via recipe
from tt_symbiote.modules.ttnn_tensor import TTNNPermute

__all__ = [
    "ResNetRecipe",
    "TTNNResNetAdaptiveAvgPool2dNHWC",
    "TTNNResNetBasicLayer",
    "TTNNResNetBottleNeckLayer",
    "TTNNResNetConvLayer",
    "TTNNResNetEmbeddings",
    "TTNNResNetShortCut",
]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
#
# HF's :class:`ResNetConvLayer` is the canonical
# Conv2d + BatchNorm2d + activation triple. We delegate to the existing
# :class:`TTNNConv2dBNActivationNHWC` integration, which folds BN into
# the conv at weight-preprocess time and runs ReLU on-device. The wrapper
# below exists only to translate HF's nested attribute shape
# (``layer.convolution``, ``layer.normalization``, ``layer.activation``)
# into the constructor arguments that the existing TTNN class expects.


class TTNNResNetConvLayer(TTNNModule):
    """Adapter: HF ``ResNetConvLayer`` -> on-device fused conv+BN+activation.

    HF's ``ResNetConvLayer.activation`` is a function from ``ACT2FN``
    (e.g. ``torch.nn.functional.relu``) rather than an ``nn.Module``.
    The existing :class:`TTNNConv2dBNActivationNHWC` only supports
    :class:`torch.nn.ReLU` (the only activation any Microsoft ResNet
    variant uses), so we wrap with a freshly constructed ``nn.ReLU()``
    when the source activation is ``relu``. Any other activation falls
    back to fused conv+BN with a separate TTNN activation downstream — or
    raises if no path is available.
    """

    @classmethod
    def from_torch(cls, layer: ResNetConvLayer) -> "TTNNResNetConvLayer":
        new = cls()
        new._fallback_torch_layer = layer

        is_relu = isinstance(layer.activation, nn.ReLU) or layer.activation is nn.functional.relu
        is_identity = isinstance(layer.activation, nn.Identity)
        # NOTE on naming: set_device's recursive walker skips any TTNNModule
        # attribute whose name starts with "_" (see device_management.py
        # line 279). That filter is by design — underscore-prefixed
        # attributes are treated as private state, not part of the
        # tree. Our children must therefore be public-attribute named
        # (no leading underscore) so set_device descends into them and
        # binds their device. The same convention applies to every child
        # ``TTNNModule`` reference below in this file.
        if is_relu:
            new.inner = TTNNConv2dBNActivationNHWC.from_torch(layer.convolution, layer.normalization, nn.ReLU())
        elif is_identity:
            new.inner = TTNNConv2dBNNHWC.from_torch(layer.convolution, layer.normalization)
        else:
            raise NotImplementedError(
                f"TTNNResNetConvLayer: unsupported activation {type(layer.activation).__name__}; "
                f"only ReLU and Identity are wired up today."
            )
        return new

    def preprocess_weights_impl(self):
        self.inner.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.inner.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, hidden_state: ttnn.Tensor, reshape_output: bool = True) -> ttnn.Tensor:
        return self.inner(hidden_state, reshape_output=reshape_output)


class TTNNResNetShortCut(TTNNModule):
    """Adapter: HF ``ResNetShortCut`` -> on-device fused conv+BN (no activation).

    HF's shortcut is 1x1 conv + BN, used to match channel/stride between
    a residual branch and its skip. We map it onto the existing
    :class:`TTNNConv2dBNNHWC` so weight prep folds BN at load time.
    """

    @classmethod
    def from_torch(cls, layer: ResNetShortCut) -> "TTNNResNetShortCut":
        new = cls()
        new._fallback_torch_layer = layer
        new.inner = TTNNConv2dBNNHWC.from_torch(layer.convolution, layer.normalization)
        return new

    def preprocess_weights_impl(self):
        self.inner.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.inner.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, hidden_state: ttnn.Tensor, reshape_output: bool = True) -> ttnn.Tensor:
        return self.inner(hidden_state, reshape_output=reshape_output)


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------
#
# We keep the basic-layer and bottleneck-layer wrappers in this file
# (not in ``integrations/``) for the same reason
# :class:`TTNNBailingMoEDecoderLayer` lives next to its model: the
# *shape* of the block (which sub-layers, in what order, where the
# residual add is) is model-specific even when the constituent ops
# (conv+BN+ReLU) are generic. Block boundaries are also where on-device
# residual adds happen — the same Phase 5 optimization that avoided
# host round-trips in the LLM decoder layer.


@trace_enabled
class TTNNResNetBasicLayer(TTNNModule):
    """HF ``ResNetBasicLayer`` -> two 3x3 convs with on-device residual.

    Used by resnet-18 and resnet-34. HF's shape:

        layer.layer      -> nn.Sequential(ResNetConvLayer, ResNetConvLayer)
        layer.shortcut   -> nn.Identity or ResNetShortCut
        layer.activation -> functional relu

    Both inner conv+BN+relu pairs run on device; the final activation
    after the residual add also runs on device via :class:`TTNNReLU`.
    """

    @classmethod
    def from_torch(cls, layer: ResNetBasicLayer) -> "TTNNResNetBasicLayer":
        new = cls()
        new._fallback_torch_layer = layer
        # Child TTNN modules MUST be public-attribute named (no leading
        # underscore) so set_device's recursive walker descends into
        # them; see comment in TTNNResNetConvLayer.from_torch above.
        new.conv1 = TTNNResNetConvLayer.from_torch(layer.layer[0])
        new.conv2 = TTNNResNetConvLayer.from_torch(layer.layer[1])
        new._has_shortcut = isinstance(layer.shortcut, ResNetShortCut)
        if new._has_shortcut:
            new.shortcut = TTNNResNetShortCut.from_torch(layer.shortcut)
        new.final_relu = TTNNReLU()
        return new

    def preprocess_weights_impl(self):
        self.conv1.preprocess_weights()
        self.conv2.preprocess_weights()
        if self._has_shortcut:
            self.shortcut.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.conv1.move_weights_to_device()
        self.conv2.move_weights_to_device()
        if self._has_shortcut:
            self.shortcut.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, hidden_state: ttnn.Tensor) -> ttnn.Tensor:
        residual = hidden_state if not self._has_shortcut else self.shortcut(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = ttnn.add(hidden_state, residual)
        hidden_state = self.final_relu(hidden_state)
        return hidden_state


@trace_enabled
class TTNNResNetBottleNeckLayer(TTNNModule):
    """HF ``ResNetBottleNeckLayer`` -> three convs with on-device residual.

    Used by resnet-50/101/152. HF's shape (note the nested
    ``nn.Sequential`` — different from torchvision's flat
    ``conv1/bn1/conv2/bn2/conv3/bn3`` that the legacy
    :class:`tt_symbiote.modules.ttnn_conv.TTNNBottleneck` was
    written against):

        layer.layer      -> nn.Sequential(
                                ResNetConvLayer(1x1, in -> reduced),
                                ResNetConvLayer(3x3, reduced -> reduced),
                                ResNetConvLayer(1x1, reduced -> out, activation=None),
                            )
        layer.shortcut   -> nn.Identity or ResNetShortCut
        layer.activation -> functional relu

    Stride placement depends on ``config.downsample_in_bottleneck``;
    we don't care — each inner ``ResNetConvLayer`` was constructed
    with the right stride upstream and we just walk in order.
    """

    @classmethod
    def from_torch(cls, layer: ResNetBottleNeckLayer) -> "TTNNResNetBottleNeckLayer":
        new = cls()
        new._fallback_torch_layer = layer
        # Child TTNN modules MUST be public-attribute named (no leading
        # underscore) so set_device's recursive walker descends into
        # them; see comment in TTNNResNetConvLayer.from_torch above.
        new.conv1 = TTNNResNetConvLayer.from_torch(layer.layer[0])
        new.conv2 = TTNNResNetConvLayer.from_torch(layer.layer[1])
        new.conv3 = TTNNResNetConvLayer.from_torch(layer.layer[2])
        new._has_shortcut = isinstance(layer.shortcut, ResNetShortCut)
        if new._has_shortcut:
            new.shortcut = TTNNResNetShortCut.from_torch(layer.shortcut)
        new.final_relu = TTNNReLU()
        return new

    def preprocess_weights_impl(self):
        self.conv1.preprocess_weights()
        self.conv2.preprocess_weights()
        self.conv3.preprocess_weights()
        if self._has_shortcut:
            self.shortcut.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.conv1.move_weights_to_device()
        self.conv2.move_weights_to_device()
        self.conv3.move_weights_to_device()
        if self._has_shortcut:
            self.shortcut.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, hidden_state: ttnn.Tensor) -> ttnn.Tensor:
        residual = hidden_state if not self._has_shortcut else self.shortcut(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)
        hidden_state = ttnn.add(hidden_state, residual)
        hidden_state = self.final_relu(hidden_state)
        return hidden_state


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------
#
# HF's ``ResNetEmbeddings`` is the stem: one 7x7 stride-2 conv+BN+relu
# (``embedder``) followed by a 3x3 stride-2 maxpool (``pooler``). The
# stem is also where we do the NCHW->NHWC permute so the rest of the
# network can stay NHWC.


class TTNNResNetEmbeddings(TTNNModule):
    """HF ``ResNetEmbeddings`` -> NCHW->NHWC permute, then conv+BN+relu, then maxpool."""

    @classmethod
    def from_torch(cls, layer: ResNetEmbeddings) -> "TTNNResNetEmbeddings":
        new = cls()
        new._fallback_torch_layer = layer
        # Child TTNN modules MUST be public-attribute named (no leading
        # underscore) so set_device's recursive walker descends into
        # them; see comment in TTNNResNetConvLayer.from_torch above.
        new.permute = TTNNPermute()
        new.embedder = TTNNResNetConvLayer.from_torch(layer.embedder)
        new.pooler = TTNNMaxPool2dNHWC.from_torch(layer.pooler)
        new._num_channels = layer.num_channels
        return new

    def preprocess_weights_impl(self):
        self.embedder.preprocess_weights()
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.embedder.move_weights_to_device()
        super().move_weights_to_device_impl()

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        # HF gives NCHW; we want NHWC for every conv from here on.
        if pixel_values.shape[1] == self._num_channels:
            pixel_values = self.permute(pixel_values, perm=[0, 2, 3, 1])
        hidden_state = self.embedder(pixel_values)
        hidden_state = self.pooler(hidden_state)
        return hidden_state


# ---------------------------------------------------------------------------
# Head: NHWC adaptive avg pool
# ---------------------------------------------------------------------------
#
# HF's ``ResNetModel.pooler`` is :class:`torch.nn.AdaptiveAvgPool2d` with
# ``output_size=(1, 1)`` — i.e. a global mean over the spatial axes. It
# expects NCHW input. After our NHWC encoder, the last hidden state is
# ``(B, H, W, C)``, so we cannot just pass it through HF's pooler. The
# wrapper below performs the spatial mean on the NHWC axes [1, 2] and
# then permutes the result back to NCHW ``(B, C, 1, 1)`` so HF's
# downstream classifier head (``nn.Flatten() -> nn.Linear``) sees the
# expected layout.


class TTNNResNetAdaptiveAvgPool2dNHWC(TTNNModule):
    """NHWC-aware replacement for HF ``ResNetModel.pooler``.

    Implements ``AdaptiveAvgPool2d(output_size=(1, 1))`` for an NHWC
    tensor by reducing over axes ``[1, 2]`` (height, width) with
    ``keepdim=True``, then permuting NHWC -> NCHW. The resulting shape
    ``(B, C, 1, 1)`` matches what HF's ``classifier = nn.Sequential(
    nn.Flatten(), nn.Linear)`` head expects, so no further changes to
    HF code are required.

    Only ``output_size == (1, 1)`` is supported (or its scalar form
    ``1``). Other adaptive sizes — extremely rare in image-classification
    backbones — should raise rather than silently produce wrong output.
    """

    @classmethod
    def from_torch(cls, pool: nn.AdaptiveAvgPool2d) -> "TTNNResNetAdaptiveAvgPool2dNHWC":
        output_size = pool.output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if tuple(output_size) != (1, 1):
            raise NotImplementedError(
                f"TTNNResNetAdaptiveAvgPool2dNHWC only supports output_size=(1, 1); " f"got {output_size}."
            )
        new = cls()
        new._fallback_torch_layer = pool
        new.permute = TTNNPermute()
        return new

    @run_on_devices(*SHARDED_COLLECTIVE_LINEAR_DEVICE_ARCHS)
    def forward(self, hidden_state: ttnn.Tensor) -> ttnn.Tensor:
        # Reduce H, W (NHWC axes 1, 2) -> (B, 1, 1, C).
        pooled = ttnn.mean(hidden_state, dim=[1, 2], keepdim=True)
        # Match HF NCHW (B, C, 1, 1) so the downstream Flatten+Linear
        # head sees the expected layout.
        pooled = self.permute(pooled, perm=[0, 3, 1, 2])
        return pooled


# ---------------------------------------------------------------------------
# Design-time coverage manifests
# ---------------------------------------------------------------------------
#
# Backfilled post-Phase 8 so ``tt_symbiote.compatibility.report`` produces
# a populated design-time view of the ResNet port. The convention mirrors
# the Gemma-4 / Qwen3-VL recipes: list HF class names from the upstream
# ``transformers/models/resnet/modeling_resnet.py`` file so the runtime
# hook in :mod:`tt_symbiote.core.run_config` can cross-reference observed
# instances against the recipe's declared intent.
#
# ResNet (every Microsoft variant) is a *full* TTNN port: every
# compute-bearing HF class is wrapped, and the only host modules left
# are pure container / orchestration code with no FLOPs of their own.
# That makes the ``cpu_fallback`` list intentionally empty — a clean
# run leaves ``compatibility.report(model)["regressions"]`` empty.
#
# Membership rules:
#   * ``tt_implemented``: HF class swapped to a TTNN wrapper by
#     ``build_module_dict`` below.
#   * ``cpu_fallback``: exercised on the demo path but kept on PyTorch.
#     Empty for ResNet — see above.
#   * ``host_glue``: intentionally host-only by policy (pure container
#     modules and the abstract HF base class — no FLOPs to accelerate).
#   * ``out_of_scope``: present in the upstream model file but never
#     instantiated by the verified ``run_resnet*.py`` scripts (alternate
#     ``ResNetBackbone`` head and HF output dataclasses).


_TT_IMPLEMENTED: list[str] = [
    "ResNetConvLayer",  # -> TTNNResNetConvLayer (fused Conv2d+BN+ReLU NHWC)
    "ResNetShortCut",  # -> TTNNResNetShortCut (1x1 Conv2d+BN NHWC)
    "ResNetBasicLayer",  # -> TTNNResNetBasicLayer (2-conv block, resnet-18/34)
    "ResNetBottleNeckLayer",  # -> TTNNResNetBottleNeckLayer (3-conv block, resnet-50/101/152)
    "ResNetEmbeddings",  # -> TTNNResNetEmbeddings (stem: NCHW->NHWC permute + conv + maxpool)
]


# ResNet is a full TTNN port — no class is intentionally left on CPU.
# nn.AdaptiveAvgPool2d (HF's pooler) and nn.Linear (HF's classifier head)
# are also swapped via ``build_module_dict`` below; they are torch
# primitives rather than HF-specific classes and so don't appear in this
# list, matching the Gemma-4 / Qwen3-VL convention.
_CPU_FALLBACK: list[str] = []


_HOST_GLUE: list[str] = [
    # Abstract base class providing the HF PreTrainedModel mixins.
    "ResNetPreTrainedModel",
    # Pure container modules — both just iterate their children. No
    # compute beyond a Python ``for`` loop.
    "ResNetStage",
    "ResNetEncoder",
    # Backbone wrapper: runs the (swapped) stem, then the (host_glue)
    # encoder, then the (swapped) NHWC adaptive avg pool. Contributes
    # only an output-dataclass packaging step.
    "ResNetModel",
    # Top-level classification head: delegates to ``ResNetModel``, then
    # the (swapped) ``nn.Flatten() + nn.Linear`` classifier, plus an
    # optional cross-entropy loss when ``labels`` is provided.
    "ResNetForImageClassification",
]


_OUT_OF_SCOPE: list[str] = [
    # ----- Output dataclasses (not torch modules) -----
    "BaseModelOutputWithNoAttention",
    "BaseModelOutputWithPoolingAndNoAttention",
    "ImageClassifierOutputWithNoAttention",
    # ----- Alternate top-level head (feature-extraction / detection
    # ----- frontend). Not exercised by the verified
    # ----- ``examples/e2e/resnet/run_resnet*.py`` scripts. ResNetBackbone
    # ----- reuses the same swapped children, so wiring it up later is
    # ----- additive — no recipe changes required.
    "ResNetBackbone",
]


# ---------------------------------------------------------------------------
# Recipe (Phase 5 / Option 1)
# ---------------------------------------------------------------------------
#
# Single-dict module-replacement recipe for ``ResNetForImageClassification``.
# Per the Option 1 contract:
#
#   * ``build_module_dict`` returns one flat ``{torch_class: ttnn_class}``
#     dict that the auto factory hands to ``register_modules`` in a
#     single pass.
#   * Each TTNN wrapper class is responsible for converting its own
#     subtree in ``from_torch``. The recipe describes the seven top-level
#     class swaps — the encoder, the stages, and the model wrapper stay
#     as HF code (they are pure containers and reuse the swapped
#     children without modification).
#   * ``post_register`` patches ``model.device`` to ``cpu`` (so HF's
#     fallback path for ``next(self.parameters())`` works after weight
#     replacement, matching the Phase 5 :class:`BailingMoEV2Recipe`
#     pattern) and stashes the resolved per-variant TTNN tuning on
#     ``model._tt_runtime_config`` for downstream wrappers to consult.
#   * ``make_kv_cache`` is N/A for vision — the no-op installed by
#     ``@register_recipe`` is the right answer.


@register_recipe(hf_class_name="ResNetForImageClassification")
class ResNetRecipe:
    """TTNN recipe for HuggingFace ResNet image-classification models."""

    tt_implemented: list[str] = _TT_IMPLEMENTED
    cpu_fallback: list[str] = _CPU_FALLBACK
    host_glue: list[str] = _HOST_GLUE
    out_of_scope: list[str] = _OUT_OF_SCOPE

    def build_module_dict(self, model):
        return {
            ResNetConvLayer: TTNNResNetConvLayer,
            ResNetShortCut: TTNNResNetShortCut,
            ResNetBasicLayer: TTNNResNetBasicLayer,
            ResNetBottleNeckLayer: TTNNResNetBottleNeckLayer,
            ResNetEmbeddings: TTNNResNetEmbeddings,
            nn.AdaptiveAvgPool2d: TTNNResNetAdaptiveAvgPool2dNHWC,
            nn.Linear: TTNNLinear,
        }

    def post_register(self, model):
        type(model).device = property(lambda self: torch.device("cpu"))
        model._tt_runtime_config = lookup_ttnn_tuning(model)
