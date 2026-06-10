<!-- SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Per-model tt-metal pinning (scalable to 100+ models)

Each tt_symbiote recipe is verified against a **specific tt-metal commit**. The
sharded op contracts a recipe depends on (layernorm / embedding / matmul tensor
layouts) change between commits, so running a recipe against a ttnn built from a
*different* commit can silently produce wrong numerics.

The design scales to 100+ models the same way `transformers` does — by
**separating "pin" from "install":**

| | What | Cardinality | Where |
|---|---|---|---|
| **PIN** | the tt-metal commit a recipe was verified against | **per model** (100+) | `RUNTIME_PINS` in `_runtime_pins.py` — metadata, *never* a pip dep |
| **INSTALL** | the ttnn runtime + Python deps that actually get installed | **one per release** | `RELEASE_TTNN` + `CAPABILITY_EXTRAS` → `pyproject.toml` |

## Why per-model commits are *not* pip dependencies

A Python process can import exactly **one** `ttnn` — it is a compiled extension
with a global device singleton. You cannot `pip install` 100 different ttnns (one
per model's commit) into one venv. So a single `pyproject.toml` installs **one
ttnn runtime + every model's Python/aux deps**; the per-model commit is recorded
as metadata and enforced at runtime by the gate.

This mirrors `transformers`: one shared `torch`, per-capability extras
(`[vision]`, `[audio]`, …), and the model-specific bits are metadata — not a
separate `torch` install per model.

## The pieces

| Piece | File | Role |
|---|---|---|
| Registry (source of truth) | `src/tt_symbiote/models/_runtime_pins.py` | `RUNTIME_PINS` (per-model commit + extras), `CAPABILITY_EXTRAS`, `RELEASE_TTNN`, `TTNN_VERSION_COMMITS` |
| Packaging | `pyproject.toml` | **generated** `ttnn-runtime` base pin + `[project.optional-dependencies]` capability extras + `all` |
| Sync tool | `scripts/sync_ttnn_extras.py` | regenerates both managed blocks; `--check` in pre-commit/CI |
| Runtime gate | `src/tt_symbiote/utils/runtime_compat.py` | warns / (strict) raises when the installed ttnn's commit ≠ a model's pinned commit |
| Recipe constant | `src/tt_symbiote/models/<m>/...` | `TT_METAL_COMMIT` **derives** from the registry |
| Test-time check | `tests/conftest.py` `tt_metal_commit_check` | non-blocking warning vs `$TT_METAL_HOME` |

Everything derives from `_runtime_pins.py`; nothing is hand-duplicated.

## Adding a model (the per-model, scalable path)

Adding the Nth model is one dict entry + (optionally) one capability group:

1. Add an entry to `RUNTIME_PINS` in `_runtime_pins.py`:

   ```python
   "MyModelForCausalLM": {
       "tt_metal_commit": "abc123…",   # THIS model's own verified commit
       "extras": ["vision"],            # capability groups it needs ([] if none)
   },
   ```

   Each model pins its **own** commit — models do **not** share one commit.

2. If the model needs a Python lib no existing capability group covers, add a
   group to `CAPABILITY_EXTRAS` (shared by every model that lists it):

   ```python
   CAPABILITY_EXTRAS = {"vision": ["torchvision"], "audio": ["librosa"], …}
   ```

3. Regenerate packaging: `python scripts/sync_ttnn_extras.py`. This writes the
   capability extras + the union `all` extra, and the single `ttnn` base pin.
4. The recipe's `TT_METAL_COMMIT` updates automatically (it reads the registry).
5. Keep `tests/<tree>/<model>/test_config.json:"tt_metal_commit"` equal to the
   registry value (asserted by `tests/auto/test_runtime_compat.py`).

This stays **O(capabilities), not O(models)** on the packaging side: 100 models
that all need vision share one `vision` extra.

## Choosing `RELEASE_TTNN` (the single runtime)

`RELEASE_TTNN` is the one ttnn the release ships (e.g. `"==0.69.0"`), pinned once
into the base `dependencies`. Leave it `""` to keep the package **source-build
only** (no ttnn auto-installed) until a verified version exists.

The ttnn wheel does not expose its tt-metal commit, so the compatible version is
found by test, not lookup:

1. `pip index versions ttnn`; pick candidates near the target commit's date.
2. In a throwaway venv per candidate, run the model e2e demos on real hardware
   (e.g. `examples/e2e/dots_ocr/run_dots_ocr.py --dp 8`) + Tier1 ops; accept the
   version yielding **correct output** + PCC ≥ 0.99.
3. Set `RELEASE_TTNN`, run `python scripts/sync_ttnn_extras.py`, and add the
   `version → commit` entry to `TTNN_VERSION_COMMITS` so the gate recognizes it.

Because there is one runtime per release, the ideal is that every on-device model
in a release pins the commit `RELEASE_TTNN` was built from. Models pinned to a
*different* commit still install and run, but the gate flags them — that is the
"ported but not yet re-verified against the current runtime" frontier (the
"we only go forward in time" branch model).

## Runtime gate behavior

`check_ttnn_compat(hf_class)` runs at `AutoModel*.from_pretrained` and again at
`set_device` (de-duplicated):

- **soft warning** by default on mismatch / unknown installed commit;
- **`TT_SYMBIOTE_STRICT_TTNN=1`** turns the warning into a `RuntimeError` (CI);
- no-op for any model absent from `RUNTIME_PINS`.

`installed_ttnn_commit()` resolution order: ① `ttnn.__tt_metal_commit__` if the
imported ttnn exposes it (future / source builds may), else ② the
`TTNN_VERSION_COMMITS` table keyed by the installed ttnn dist version, else
`None` ("cannot determine" — conservative warning).

## Long-term

Land an upstream `ttnn.__tt_metal_commit__` (or a `sfpi-version`-style file) so
`installed_ttnn_commit()` reads the commit directly and `TTNN_VERSION_COMMITS`
can be retired.
