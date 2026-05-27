# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test-only fixtures for ``tests/auto/``.

These tests do not need real hardware, but the ``tt_symbiote`` package
unconditionally imports ``ttnn`` and ``tracy``. We install lightweight
stand-ins for both at collection time so the tests run on machines that
do not have either C extension built.
"""

import sys
import types


class _Anything:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Anything()

    def __getattr__(self, _):
        return _Anything()

    def __getitem__(self, _):
        return _Anything()

    def __setitem__(self, *a, **k):
        pass

    def __iter__(self):
        return iter([])

    def __mul__(self, _):
        return _Anything()

    def __rmul__(self, _):
        return _Anything()

    def __or__(self, _):
        return _Anything()

    def __ror__(self, _):
        return _Anything()

    def __bool__(self):
        return False


class _StubMod(types.ModuleType):
    def __getattr__(self, name):
        return _Anything()


for name in (
    "ttnn",
    "ttnn.model_preprocessing",
    "ttnn.distributed",
    "tracy",
    "tracy.signpost",
):
    if name not in sys.modules:
        mod = _StubMod(name)
        mod.__file__ = f"<stub:{name}>"
        sys.modules[name] = mod
sys.modules["tracy"].signpost = _Anything()
