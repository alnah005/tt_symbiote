# PyPI release checklist (v0.1.0)

Companion to [`release_process.md`](release_process.md). That doc is the
canonical recipe for **every** release; this checklist is a tactical,
one-time worksheet for the very first PyPI publication on the
branch-driven model, tracking which items already shipped in commits
and which still require human action.

Tick items with `- [x]`. The format is:

```text
- [x] Done — what landed it, when.
- [ ] Pending — what's still needed and what unblocks it.
```

## Stage A: in-repo infrastructure

All in-repo deliverables required to build, smoke-test, and publish a
release.

- [x] **`pyproject.toml` PEP 621 metadata** — name, description, license,
  authors, URLs, dependencies, literal `version = "0.1.0"`.
- [x] **`pyproject.toml` Trove classifiers** — `Development Status :: 4 -
  Beta`, Python 3.10/3.11/3.12, Apache-2.0, Linux, AI/ML topic.
- [x] **`pyproject.toml` `[ttnn]` optional extra** — pins `ttnn==0.68.0`
  to match `scripts/ttnn-pin.txt`.
- [x] **`pyproject.toml` `[dev]` extra includes `build`/`twine`** — so
  `make install` provisions the local pre-publish gate.
- [x] **`pyproject.toml` excludes `*.bak` from packaged data** — keeps
  migration-era backups out of the published wheel.
- [x] **`pyproject.toml` excludes `_experimental` package** —
  unregistered modeling code stays out of the wheel; see
  `src/tt_symbiote/_experimental/__init__.py`.
- [x] **`MANIFEST.in` prunes internal docs / experimental trees** —
  `docs/internal/`, `tests/experimental/`,
  `src/tt_symbiote/_experimental/`, `.cursor/`, `CLAUDE.md`, plus a
  global exclude for `.bak` / `.orig` / `.rej` / `__pycache__` / `.pyc`
  so the sdist is as clean as the wheel.
- [x] **Literal `version = "0.1.0"` (no `setuptools-scm`)** — tt_symbiote
  is branch-driven; the release branch is the release marker. See
  `docs/release_process.md` §Versioning model.
- [x] **`src/tt_symbiote/__init__.py` exposes `__version__`** — via
  `importlib.metadata.version("tt_symbiote")`, so installed users can
  introspect the version.
- [x] **`src/tt_symbiote/utils/graph_visualization.py` lazy-imports
  matplotlib** — otherwise `pip install tt_symbiote` would fail at
  `import tt_symbiote` time because matplotlib is not in dependencies.
- [x] **`src/tt_symbiote/core/run_config.py` guards `from tracy import
  signpost`** — tracy is not on PyPI; the no-op fallback shim unblocks
  the happy-path pip install.
- [x] **`README.md` Installation section** — `pip install
  "tt_symbiote[ttnn]"` as the recommended path, loud sfpi caveat,
  bootstrap script as the contributor path.
- [x] **`docs/install_prerequisites.md`** — what sfpi is, how to install
  + verify it, the `(ttnn, sfpi)` compatibility table, troubleshooting.
- [x] **`docs/release_process.md`** — branch-driven recipe, one-time
  Trusted Publisher setup, rollback policy, API-token fallback.
- [x] **`docs/supported_models.md` links to install_prerequisites** —
  one-line callout above the model tables.
- [x] **`Makefile` cleanup** — `test` target points at `tests/auto`
  (not the nonexistent `tests/capabilities`); new `install-ttnn` and
  `dist` targets.
- [x] **`.github/workflows/release.yml`** — `workflow_dispatch` with a
  `target` input (testpypi | pypi); jobs: build → smoke → publish-...,
  all via OIDC Trusted Publishers. **No tag trigger.**
- [x] **`LICENSE` present** — Apache-2.0.
- [x] **`scripts/bootstrap_venv.sh` + `scripts/ttnn-pin.txt` consistent
  with `[ttnn]` extra** — both pin `ttnn==0.68.0` ↔ `sfpi 7.35.3`.

## Stage B: local pre-publish gate

Run these commands on the maintainer's machine before bumping the
version.

- [x] **`python -m build` succeeds** — produces `tt_symbiote-0.1.0.tar.gz`
  and `tt_symbiote-0.1.0-py3-none-any.whl` in `dist/`.
- [x] **`twine check dist/*` passes** — both sdist and wheel report
  `PASSED`.
- [x] **Wheel and sdist exclude internal artifacts** — `unzip -l
  dist/*.whl` shows no `_experimental` / `qwen3_moe` / `.bak`; `tar tzf
  dist/*.tar.gz` shows no `PROJECT_PROPOSAL.md`, `migration_notes.md`,
  `tests/experimental/`, `.cursor/`, `CLAUDE.md`.
- [x] **Fresh venv install of wheel + import succeeds** — with `ttnn`
  stubbed via `MagicMock` (CI has no Tenstorrent hardware), the
  pure-Python surface is reachable: `AutoModelForCausalLM`,
  `set_device`, `__version__` are all defined and the version equals
  `"0.1.0"`.
- [x] **Hardware-free test suite green** — `pytest tests/auto` →
  124/124 passing.
- [x] **`pre-commit run --all-files` is green** — black, isort,
  autoflake, yamllint, end-of-file, trailing-whitespace all clean.

## Stage C: PyPI name availability

- [x] **`tt_symbiote` / `tt-symbiote` free on PyPI** — verified by HTTP
  404 from `https://pypi.org/pypi/tt-symbiote/json` and
  `https://pypi.org/pypi/tt_symbiote/json`.
- [x] **`tt-symbiote` free on TestPyPI** — HTTP 404 from
  `https://test.pypi.org/pypi/tt-symbiote/json`.

## Stage D: PyPI / TestPyPI configuration (external, manual)

Requires PyPI/TestPyPI account access and a logged-in browser. The
exact field values are in
[`release_process.md`](release_process.md#one-time-setup-trusted-publishers-on-pypi--testpypi).

- [ ] **TestPyPI pending publisher registered** — at
  <https://test.pypi.org/manage/account/publishing/>, with project
  name `tt_symbiote`, owner = the GitHub repo's org/user, repository
  name = `tt_symbiote`, workflow filename = `release.yml`, environment
  name = `pypi`.
- [ ] **PyPI pending publisher registered** — same fields, at
  <https://pypi.org/manage/account/publishing/>.
- [ ] **GitHub Actions environment `pypi` created** — under repo
  *Settings → Environments → New environment*. Optionally add a
  required reviewer for production releases.

## Stage E: publish (external, semi-automated)

- [ ] **HEAD is the merge commit on `transformers5.9.0`** — that branch
  IS the release marker; `git log -1 --oneline` should show the merge
  bringing the cleaned `aroberge/bootstrap` work into
  `transformers5.9.0`.
- [ ] **`pyproject.toml` `version` is `0.1.0`** — no `rc` suffix for
  the stable cut; if doing an RC first, change to `0.1.0rc1`, commit,
  push, dispatch with `target=testpypi`, smoke-install, then change
  back to `0.1.0` for the stable cut.
- [ ] **Dispatch `release.yml` with `target=testpypi`** — from the
  GitHub Actions UI (`Actions → release → Run workflow`,
  branch = `transformers5.9.0`, target = `testpypi`). Watch all three
  jobs (`build`, `smoke`, `publish-testpypi`) reach green.
- [ ] **Manually smoke-install from TestPyPI**:
  ```bash
  python -m venv /tmp/tt_test
  /tmp/tt_test/bin/pip install \
    -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "tt_symbiote==0.1.0"
  /tmp/tt_test/bin/python -c "import tt_symbiote; print(tt_symbiote.__version__)"
  ```
- [ ] **Dispatch `release.yml` with `target=pypi`** — only after the
  TestPyPI smoke install reported `0.1.0`.

## Stage F: post-publish verification

- [ ] **TestPyPI release page exists** — at
  `https://test.pypi.org/project/tt-symbiote/0.1.0/`, showing the
  wheel + sdist artifacts.
- [ ] **PyPI release page exists** — at
  `https://pypi.org/project/tt-symbiote/0.1.0/`, same artifacts.
- [ ] **Clean-machine install works** — on a host with matching sfpi
  (`/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version` →
  `sfpi:7.35.3...`):
  ```bash
  python -m venv /tmp/tt010
  /tmp/tt010/bin/pip install "tt_symbiote[ttnn]==0.1.0"
  /tmp/tt010/bin/python -c "import tt_symbiote, ttnn; print(tt_symbiote.__version__, ttnn.__version__)"
  ```
- [ ] **End-to-end demo runs from the installed wheel** —
  `examples/e2e/run_ling_mini_2_0.py` on a T3K (or any
  hardware-target script appropriate to the machine you have): clone
  the repo only for the script, but `pip install
  "tt_symbiote[ttnn]==0.1.0"` provides the runtime.
