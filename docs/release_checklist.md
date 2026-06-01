# PyPI release checklist (v0.1.0)

Companion to [`release_process.md`](release_process.md). That doc is the
canonical recipe for **every** release; this checklist is a tactical,
one-time worksheet for the very first PyPI publication, tracking which
items already shipped in commits and which still require human action.

Tick items with `- [x]`. The format is:

```text
- [x] Done — what landed it, when.
- [ ] Pending — what's still needed and what unblocks it.
```

## Stage A: in-repo infrastructure

All in-repo deliverables required to build, smoke-test, and publish a
release.

- [x] **`pyproject.toml` PEP 621 metadata** — name, description, license,
  authors, URLs, dependencies, dynamic version. (Pre-existing.)
- [x] **`pyproject.toml` Trove classifiers** — `Development Status :: 4 -
  Beta`, Python 3.10/3.11/3.12, Apache-2.0, Linux, AI/ML topic. (Commit
  `e2db757`.)
- [x] **`pyproject.toml` `[ttnn]` optional extra** — pins `ttnn==0.68.0`
  to match `scripts/ttnn-pin.txt`. (Commit `e2db757`.)
- [x] **`pyproject.toml` `[dev]` extra includes `build`/`twine`** — so
  `make install` provisions the local pre-tag gate. (Commit `e2db757`.)
- [x] **`pyproject.toml` excludes `*.bak` from packaged data** — keeps
  migration-era backups out of the published wheel. (Commit `e2db757`.)
- [x] **`src/tt_symbiote/__init__.py` exposes `__version__`** — via
  `importlib.metadata`, so users can introspect the installed
  version. (Commit `e2db757`.)
- [x] **`src/tt_symbiote/utils/graph_visualization.py` lazy-imports
  matplotlib** — otherwise `pip install tt_symbiote` would fail at
  `import tt_symbiote` time because matplotlib is not in dependencies.
  (Commit `e2db757`.)
- [x] **`src/tt_symbiote/core/run_config.py` guards `from tracy import
  signpost`** — tracy is not on PyPI; the no-op fallback shim unblocks
  the happy-path pip install. (Commit `e2db757`.)
- [x] **`README.md` Installation section** — `pip install
  "tt_symbiote[ttnn]"` as the recommended path, loud sfpi caveat,
  bootstrap script as the contributor path. (Commit `e2db757`.)
- [x] **`docs/install_prerequisites.md`** — what sfpi is, how to install
  + verify it, the `(ttnn, sfpi)` compatibility table, troubleshooting.
  (Commit `e2db757`.)
- [x] **`docs/release_process.md`** — one-time Trusted Publisher setup,
  per-release recipe, rollback policy, API-token fallback. (Commit
  `e2db757`.)
- [x] **`docs/supported_models.md` links to install_prerequisites** —
  one-line callout above the model tables. (Commit `e2db757`.)
- [x] **`Makefile` cleanup** — `test` target points at `tests/auto`
  (not the nonexistent `tests/capabilities`); new `install-ttnn` and
  `dist` targets. (Commit `e2db757`.)
- [x] **`.github/workflows/release.yml`** — build → smoke (ttnn-stubbed
  import) → publish-testpypi (every tag) → publish-pypi (every non-`-rc`
  tag), all via OIDC Trusted Publishers. (Commit `e2db757`.)
- [x] **`LICENSE` present** — Apache-2.0. (Pre-existing.)
- [x] **`scripts/bootstrap_venv.sh` + `scripts/ttnn-pin.txt` consistent
  with `[ttnn]` extra** — both pin `ttnn==0.68.0` ↔ `sfpi 7.35.3`.

## Stage B: local pre-tag gate

Run these commands on the maintainer's machine before creating the tag.

- [x] **`python -m build` succeeds** — produces `tt_symbiote-<v>.tar.gz`
  and `tt_symbiote-<v>-py3-none-any.whl` in `dist/`.
- [x] **`twine check dist/*` passes** — both sdist and wheel report
  `PASSED`.
- [x] **Fresh venv install of wheel + import succeeds** — with `ttnn`
  stubbed via `MagicMock` (CI has no Tenstorrent hardware), the
  pure-Python surface is reachable: `AutoModelForCausalLM`,
  `set_device`, `__version__` are all defined.
- [x] **Hardware-free test suite green** — `pytest tests/auto
  tests/models/gemma4 tests/models/qwen3_vl` → 127/127 passing.
- [x] **Sandboxed tag dry-run** — a temporary git worktree with `v0.1.0`
  tagged at HEAD built `tt_symbiote-0.1.0-py3-none-any.whl`, installed
  cleanly in a fresh venv, reported `tt_symbiote.__version__ ==
  "0.1.0"`, and registered all 4 recipes
  (`BailingMoeV2ForCausalLM`, `Gemma4ForConditionalGeneration`,
  `Qwen3VLForConditionalGeneration`, `ResNetForImageClassification`).
  Worktree cleaned up.
- [x] **`pre-commit run --all-files` is green** — production-cleanup
  lint sweep (commit `d2f7aad`) landed all of black, isort, autoflake,
  yamllint, end-of-file, and trailing-whitespace fixes.

## Stage C: PyPI name availability

- [x] **`tt_symbiote` / `tt-symbiote` free on PyPI** — HTTP 404 from
  `https://pypi.org/pypi/tt-symbiote/json` and
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

## Stage E: tag and publish (external, semi-automated)

- [ ] **HEAD is clean and up-to-date** — `git status` shows no
  uncommitted work; `git log` shows the two infrastructure commits
  (`38dfd0a`, `e2db757`) on the integration branch.
- [ ] **Merge the integration branch to `main`** — Stage D's Trusted
  Publisher matches *workflow filename + repo*, not branch, so a
  branch-pushed tag will trigger the workflow; but tagging on `main`
  is the cleaner convention.
- [ ] **Create the annotated tag** — `git tag -a v0.1.0 -m "First
  public preview: Phase 8 Wave A + B (Gemma-4, Qwen3-VL, ResNet,
  Ling)"`.
- [ ] **Push the tag** — `git push origin v0.1.0`.
- [ ] **`release.yml` reaches green** — watch
  `https://github.com/<owner>/tt_symbiote/actions/workflows/release.yml`;
  all four jobs (`build`, `smoke`, `publish-testpypi`, `publish-pypi`)
  should succeed.

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
