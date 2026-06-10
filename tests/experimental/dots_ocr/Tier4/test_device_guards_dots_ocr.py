# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier 4 device-guard tests for dots.ocr (software-only; no hardware needed).

Every TTNN ``forward`` must carry an ``@run_on_devices`` guard whitelisting the
target arch (T3K). ``set_device`` reads ``forward.__tt_allowed_archs__`` and
swaps in the torch fallback on a non-allowed arch, so these guards are what gate
the pure-TTNN path onto T3K (8, 1)."""

from tt_symbiote.core.module import DeviceArch
from tt_symbiote.models.dots_ocr import (
    TTNNDotsOCRMLP,
    TTNNDotsOCRVisionTower,
    TTNNEmbedding,
)
from tt_symbiote.models.dots_ocr.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLocalShardRMSNorm,
)


def _allowed_archs(cls):
    fn = cls.forward
    archs = getattr(fn, "__tt_allowed_archs__", None)
    assert archs is not None, f"{cls.__name__}.forward is missing the @run_on_devices guard"
    return set(archs)


def test_decoder_layer_guard_includes_t3k():
    archs = _allowed_archs(TTNNDotsOCRDecoderLayer)
    assert DeviceArch.T3K in archs, f"{archs} should whitelist T3K"


def test_decoder_layer_mesh_shape_t3k_is_8x1():
    mesh = getattr(TTNNDotsOCRDecoderLayer.forward, "__tt_mesh_shape__", None)
    assert mesh is not None, "decoder layer forward should declare a mesh shape"
    # per-arch mapping {DeviceArch: (rows, cols)}
    assert isinstance(mesh, dict) and mesh.get(DeviceArch.T3K) == (8, 1), f"T3K mesh should be (8,1), got {mesh}"


def test_core_module_forwards_have_t3k_guard():
    for cls in (TTNNDotsOCRMLP, TTNNDotsOCRLocalShardRMSNorm, TTNNEmbedding, TTNNDotsOCRVisionTower):
        archs = _allowed_archs(cls)
        assert DeviceArch.T3K in archs, f"{cls.__name__}.forward guard {archs} should include T3K"
