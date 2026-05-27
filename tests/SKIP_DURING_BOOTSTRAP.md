# Phase 2 smoke check — what was actually verified

The Phase 1+2 plan requested a `pip install -e . && pytest tests/ --collect-only`
smoke check at the end of Phase 2. The achievable smoke check on the migration
host differed from the original wording; this file records the actual outcome
so subsequent phases know what is and isn't proven.

## Achievable on the migration host

- **AST parse:** 100% (45 source files, 47 test files, 92 total) parse with the
  CPython 3.10 AST module. No syntax errors.
- **Editable install:** `pip install -e . --no-build-isolation --no-deps`
  succeeds and registers `tt_symbiote` as an editable package.
- **Stubbed import smoke:** 31 of 33 non-`__init__` modules under
  `src/tt_symbiote/` import cleanly when `ttnn` and `tracy` are stubbed via
  `scripts/_smoke_conftest.py`.
- **Top-level `import tt_symbiote`** succeeds.

## Known gaps (not blocking Phase 2)

### Two dispatcher modules that don't import

```
tt_symbiote/core/dispatchers/dispatcher_config.py:
    NameError: name 'default_dispatcher' is not defined
tt_symbiote/core/dispatchers/tensor_operations_dispatcher.py:
    NameError: name 'handle_view' is not defined
```

Both are caused by the codemod intentionally dropping
`from models.experimental.tt_symbiote.core.dispatchers.*` imports (per the plan,
§2.3 rule 1). The entire `core/dispatcher.py`, `core/torch_dispatcher.py`, and
`core/dispatchers/` subtree is **slated for deletion in Phase 3** — these
NameErrors disappear when the dispatcher subsystem is removed.

### pytest collect-only requires a fuller runtime

`pytest tests/ --collect-only` cannot be made fully clean on the migration host
because the test suite needs:

- a real (built-from-tt-metal) `ttnn` import
- `tracy` (currently a tt-metal-built C extension; not on PyPI)
- optional model-specific deps such as `torchvision` (for `test_modeling_resnet`),
  `decord`/`av` (for video tests), etc.

`transformers==5.9.0` is pinned in `pyproject.toml` and installs cleanly from
PyPI — it is no longer in this list. The remaining gaps (`ttnn`, `tracy`) are
documented as **open question OQ-1** in
[`PROJECT_PROPOSAL.md`](../PROJECT_PROPOSAL.md) §11. They are environment
concerns; nothing about the test files is intrinsically broken by the migration.

### Tests with surviving cross-repo imports (will need manual help during port)

| Test path | Surviving cross-repo imports | Resolve in phase |
|---|---|---|
| `tests/models/whisper/test_modeling_whisper.py` | `models.demos.utils.common_demo_utils.get_mesh_mappers`; `models.demos.whisper.tt.ttnn_optimized_functional_whisper.{create_custom_mesh_preprocessor, encoder_layer}` | Whisper port (Phase 7, P3 tier) |

When running the smoke check, ignore this file:

```bash
python -m pytest tests/ --collect-only -q \
    -p scripts._smoke_conftest \
    --ignore=tests/models/whisper
```

## Re-running the achievable smoke check

From the repo root:

```bash
pip install -e . --no-build-isolation
pip install pytest

# AST + stubbed-import sweep
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import _smoke_conftest  # installs ttnn/tracy stubs
from pathlib import Path
import importlib
errs = []
for p in sorted(Path('src/tt_symbiote').rglob('*.py')):
    if p.name == '__init__.py': continue
    mod = 'tt_symbiote.' + str(p.relative_to('src/tt_symbiote')).replace('/','.').removesuffix('.py')
    try: importlib.import_module(mod)
    except Exception as e: errs.append((mod, type(e).__name__, str(e)[:120]))
print(f'{len(errs)} import errors (expected: 2 dispatcher NameErrors)')
for m, t, e in errs: print(f'  [{t}] {m}: {e}')
"
```

Expected output: `2 import errors (expected: 2 dispatcher NameErrors)`.
