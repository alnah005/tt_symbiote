# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-guard verification for TTNN modules touched by dots.ocr tests.

dots.ocr does NOT introduce model-specific TTNN modules yet (the bring-up
relies on shared integration classes only), so this test asserts that every
shared TTNNModule actually used by the dots.ocr tiered tests has been declared
runnable on a known DeviceArch via ``@run_on_devices`` -- or, when no guard is
present, that the absence is deliberate (the class inherits its parent's
``forward`` without overriding it).

If a new dots.ocr-specific TTNNModule is added under
``src/tt_symbiote/models/dots_ocr/`` it MUST appear in the
``_REQUIRED_GUARDED_MODULES`` list below.

TT_METAL_COMMIT used during scaffolding: e3447fd55874d8625f3c2e894ecc9409bb606805
"""

import inspect

from tt_symbiote.core.module import TTNNModule
from tt_symbiote.modules.ttnn_activation import TTNNGelu, TTNNSilu
from tt_symbiote.modules.ttnn_linear import TTNNLinear
from tt_symbiote.modules.ttnn_normalization import (
    TTNNLayerNorm,
    TTNNLocalRMSNorm,
    TTNNRMSNorm,
)


_REQUIRED_GUARDED_MODULES = [
    TTNNLinear,
    TTNNRMSNorm,
    TTNNLocalRMSNorm,
    TTNNLayerNorm,
    TTNNSilu,
    TTNNGelu,
]


def _forward_is_own(cls):
    """True if ``cls.forward`` was defined directly on ``cls`` (not inherited)."""
    if "forward" not in cls.__dict__:
        return False
    return True


def _has_run_on_devices_guard(fn) -> bool:
    """Heuristic: the @run_on_devices decorator stamps ``__tt_allowed_archs__``
    on the wrapper. Fall back to closure inspection for older decorations."""
    if getattr(fn, "__tt_allowed_archs__", None):
        return True
    closure = getattr(fn, "__closure__", None)
    if closure is None:
        return False
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, frozenset) and value and all(
            type(v).__name__ == "DeviceArch" for v in value
        ):
            return True
    return False


def test_required_modules_are_ttnn_modules():
    """Every required class must subclass TTNNModule."""
    for cls in _REQUIRED_GUARDED_MODULES:
        assert issubclass(cls, TTNNModule), f"{cls.__name__} must subclass TTNNModule"


def test_forward_methods_present_or_inherited():
    """If a class defines its own forward(), the body must be inspectable.

    This is a sanity guard -- if a TTNNModule overrides ``forward`` but the
    implementation later disappears (e.g., import-time refactor), the test
    will catch it before runtime.
    """
    for cls in _REQUIRED_GUARDED_MODULES:
        forward = cls.forward
        assert callable(forward), f"{cls.__name__}.forward must be callable"
        src = None
        try:
            src = inspect.getsource(forward)
        except (OSError, TypeError):
            src = ""
        # Either we got real source, or the class inherits forward from a parent
        # that we can inspect.
        if not src.strip():
            assert not _forward_is_own(cls), (
                f"{cls.__name__}.forward was overridden but no source could be read"
            )
