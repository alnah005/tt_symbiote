# Local-only conftest used for the Phase 2 smoke check. Stubs out the C
# extensions that are not pip-installable on this host (ttnn, tracy). NOT
# committed to the repo; used only when running:
#
#   python -m pytest tests/ --collect-only \
#       -p scripts._smoke_conftest --ignore=tests/models/whisper
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
