# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Paged KV-cache helper for dots.ocr.

``_create_paged_kv_cache`` and ``PipelineConfig`` are defined in the faithful
full port ``pipeline.py`` (mirroring ``models/dots_ocr.py`` in tt-metal). This
module re-exports them so the single-layer decoder tests can build a paged KV
cache without depending on the import path of the full pipeline. The pipeline
remains the single source of truth -- this is a thin compatibility shim.

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"
"""

from tt_symbiote.models.dots_ocr.pipeline import PipelineConfig, _create_paged_kv_cache

TT_METAL_COMMIT = "c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195"

__all__ = ["_create_paged_kv_cache", "PipelineConfig"]
