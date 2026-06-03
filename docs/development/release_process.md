# Release process

This document is the canonical recipe for cutting a `tt_symbiote`
release. The goals are:

- Reproducibility: anyone with repo-admin + PyPI-owner access can cut
  a release end-to-end from this document alone.
- Safety: a typo, a missing file, or a broken import never makes it
  to PyPI — the local pre-publish gate and the CI smoke step catch
  them.
- Auditability: every published wheel is traceable to a commit on the
  release branch and an OIDC-authenticated GitHub Actions run.

## Versioning model

tt_symbiote uses **branch-driven, literal versioning** — no git tags.

- The single source of truth is the literal `version = "..."` string
  in [`pyproject.toml`](../pyproject.toml) under `[project]`.
  `importlib.metadata.version("tt_symbiote")` reads that string after
  install; `tt_symbiote.__version__` re-exports it.
- The release branch (currently `transformers5.9.0`) **is** the
  release marker. Cutting a release means: bump the version on the
  release branch, push, then trigger the publish workflow.
- Version format: `MAJOR.MINOR.PATCH`, plus an optional `rcN` suffix
  for release candidates (`0.1.0`, `0.2.0rc1`, `0.2.0`, `0.2.1`, ...).
- RCs publish to TestPyPI only. Stable versions publish to TestPyPI
  *and* PyPI. Routing is controlled by the `target` input on
  `release.yml`'s `workflow_dispatch`, not by the version string —
  but by convention you bump to an `rcN` suffix before dispatching
  with `target=testpypi` and to the plain version before dispatching
  with `target=pypi`.

## One-time setup: Trusted Publishers on PyPI / TestPyPI

PyPI's recommended publish path is [Trusted Publishers
(OIDC)](https://docs.pypi.org/trusted-publishers/) — GitHub Actions
exchanges an OIDC ID token for a short-lived API token; no long-lived
secret ever lives in the repo.

Do this *once* per index (you need both because we publish to TestPyPI
first and then PyPI):

1. Register the project name. PyPI normalizes `tt_symbiote` →
   `tt-symbiote`; both forms address the same project.
   - <https://test.pypi.org/manage/account/publishing/>
   - <https://pypi.org/manage/account/publishing/>

   If the project doesn't exist yet (first release), use the
   *pending publisher* form on the same page — PyPI will create the
   project on first successful upload.

2. Add a new "pending" (or "trusted") publisher with these values:

   | Field                | Value                                       |
   |----------------------|---------------------------------------------|
   | PyPI Project Name    | `tt_symbiote`                               |
   | Owner                | `alnah005` *(or the canonical Tenstorrent org)* |
   | Repository name      | `tt_symbiote`                               |
   | Workflow filename    | `release.yml`                               |
   | Environment name     | `pypi`                                      |

   The workflow filename and environment name **must match exactly**
   what's in [`.github/workflows/release.yml`](../.github/workflows/release.yml);
   PyPI cross-checks them against the OIDC claims at publish time.

3. On GitHub: create an Actions environment named `pypi` under
   *Settings → Environments → New environment*. No secrets / reviewers
   are strictly required for OIDC, but adding a required reviewer is a
   good belt-and-suspenders for production releases.

## Per-release recipe

### Phase 1 — local pre-publish gate

Run **on the maintainer's machine**, on a clean tree, *before*
committing the version bump. Catches packaging errors that would
otherwise leave a half-published version on PyPI (which is
irreversible — see Rollback policy).

```bash
# 1. Sanity-check the working tree.
git checkout transformers5.9.0
git pull --ff-only
git status                         # must be clean
git log -1 --oneline               # confirm head SHA

# 2. Build sdist + wheel.
python -m pip install --upgrade build twine
rm -rf dist/ build/ *.egg-info
python -m build                    # produces dist/tt_symbiote-X.Y.Z.tar.gz + .whl
ls -lh dist/
twine check dist/*                 # validates long-description / metadata

# 3. Install in a fresh venv and smoke-import.
python -m venv /tmp/tt_smoke
/tmp/tt_smoke/bin/pip install dist/*.whl
/tmp/tt_smoke/bin/python - <<'PY'
import sys
from unittest.mock import MagicMock
# /tmp/tt_smoke has no Tenstorrent hardware. MagicMock (vs
# types.ModuleType) so eager attribute accesses at module load time
# (e.g. `ttnn.float32`) get a usable, hashable placeholder.
# tracy does NOT need stubbing — core/run_config.py has a fallback
# no-op signpost when tracy is unavailable.
for n in ("ttnn", "ttnn.model_preprocessing", "ttnn.distributed"):
    sys.modules[n] = MagicMock()

import tt_symbiote
print("tt_symbiote", tt_symbiote.__version__)
assert hasattr(tt_symbiote, "AutoModelForCausalLM")
assert hasattr(tt_symbiote, "set_device")
PY
rm -rf /tmp/tt_smoke

# 4. PyPI name availability (only matters before the very first release).
pip index versions tt_symbiote 2>&1 || true
# - Output "no matching distribution" → name is free, proceed.
# - Output a version list → another project owns the name; rename in
#   pyproject.toml *before* publishing.
```

### Phase 2 — bump the `(ttnn, sfpi)` pair (skip if unchanged)

If this release ships a new `ttnn` wheel, bump these two files **in
the same commit** so the bootstrap script and the PyPI extra never
disagree:

- [`scripts/ttnn-pin.txt`](../scripts/ttnn-pin.txt)
- [`pyproject.toml`](../pyproject.toml) → `[project.optional-dependencies].ttnn`

Then rerun Phase 1.

### Phase 3 — bump the version and push

```bash
# Edit pyproject.toml: change `version = "0.1.0"` to the new value.
# For an RC: `version = "0.2.0rc1"`.
# For stable: `version = "0.2.0"`.
$EDITOR pyproject.toml

# Verify the bump shows up.
git diff pyproject.toml

# Commit on the release branch.
git add pyproject.toml
git commit -m "release: tt_symbiote 0.2.0"
git push origin transformers5.9.0
```

### Phase 4 — dispatch the publish workflow

The release workflow has **no automatic triggers** — pushing to the
release branch does NOT publish. You publish explicitly from the
GitHub Actions UI:

1. Navigate to *Actions → release → Run workflow*.
2. Choose the branch: `transformers5.9.0`.
3. Choose `target`:
   - `testpypi` — uploads to <https://test.pypi.org/>. **Always do
     this first.**
   - `pypi` — uploads to <https://pypi.org/>. Only after TestPyPI
     succeeded and you've manually smoke-installed from there
     (Phase 5).

The workflow runs:

1. `build` — `python -m build` + `twine check`.
2. `smoke` — installs the wheel in a clean ubuntu-latest venv and
   imports `tt_symbiote` with `ttnn` stubbed (CI is hardware-free).
3. `publish-testpypi` *or* `publish-pypi`, gated by the `target`
   input.

### Phase 5 — post-publish verification

```bash
# After dispatch with target=testpypi:
python -m venv /tmp/tt_test
/tmp/tt_test/bin/pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "tt_symbiote==$VERSION"
/tmp/tt_test/bin/python -c "import tt_symbiote; print(tt_symbiote.__version__)"
rm -rf /tmp/tt_test

# After dispatch with target=pypi (skip for rcN versions):
python -m venv /tmp/tt_prod
/tmp/tt_prod/bin/pip install "tt_symbiote==$VERSION"
/tmp/tt_prod/bin/python -c "import tt_symbiote, ttnn; print(tt_symbiote.__version__, ttnn.__version__)"
rm -rf /tmp/tt_prod
```

On a host with matching sfpi, run one of the end-to-end demos to
confirm the install really drives silicon:

```bash
python examples/e2e/run_ling_mini_2_0.py     # T3K, ~1 minute
```

## Rollback policy

PyPI does not allow re-uploading a version, even after a yank — you
can hide a release but you can't replace its artifacts. Treat
published versions as immutable:

- **Found a packaging bug after publish.** Yank the bad version on
  PyPI (project settings → release → yank) so resolvers stop picking
  it. Cut a `+1` patch release (bump `pyproject.toml` to the next
  patch — e.g. `0.1.2` → `0.1.3` — push, dispatch).
- **Found a hardware regression after publish.** Yank, then cut a
  patch with the fix.
- **Aborted release (CI failed before publish).** Just push a follow-up
  commit on the release branch — there is no tag to delete because we
  don't use tags. The previously bumped version string never reached
  PyPI, so it can simply be left in place or bumped further.

## Fallback: API token (only if Trusted Publishers can't be configured)

If you don't have admin on the PyPI project and can't add a Trusted
Publisher (e.g., publishing to a temporary fork's PyPI account during
a rehearsal), the workflow can fall back to an API token:

1. Create an API token on PyPI scoped to the `tt_symbiote` project.
2. Add it to the `pypi` GitHub environment as a secret named
   `PYPI_API_TOKEN`.
3. In `release.yml`, replace the OIDC publisher step's
   `permissions: { id-token: write }` block with:
   ```yaml
   - uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

This is **not** the default — long-lived tokens are a leak risk and
PyPI is actively migrating projects off them. Use only as a last
resort.

## CI workflow reference

The full pipeline lives in
[`.github/workflows/release.yml`](../.github/workflows/release.yml).
Job graph:

```text
build (sdist + wheel)
  └── smoke (stubbed ttnn import)
        ├── publish-testpypi (only when target=testpypi)
        └── publish-pypi     (only when target=pypi)
```

`id-token: write` and `environment: pypi` are both required for OIDC
Trusted Publishers to work; the smoke step is intentionally
hardware-free so the pipeline doesn't depend on Tenstorrent runners.

## Why branch-driven (not tag-driven)?

Tag-driven releases (the classic `setuptools-scm` flow) couple the
release marker to a git ref that can be deleted, retagged, or
force-overwritten — and at the same time leak interim builds with
ugly versions like `0.1.1.dev3+gabcdef0` to anyone running
`python -m build` on a non-tagged commit. The branch-driven model
removes both problems:

- The version is whatever `pyproject.toml` says, end of story.
  `python -m build` on any commit produces a clean version string.
- Releasing requires an intentional commit (the version bump) plus an
  intentional dispatch (the workflow run). There is no way to
  accidentally publish by retagging or by force-pushing a tag.

## Related docs

- [`install_prerequisites.md`](install_prerequisites.md) — the
  `(ttnn, sfpi)` story the release inherits.
- [`development/migration_notes.md`](development/migration_notes.md) — the
  phase history that the version numbers track (internal-only,
  excluded from sdist).
- [`../scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh) —
  the contributor install path that runs *against* the same pin the
  PyPI extra ships.
