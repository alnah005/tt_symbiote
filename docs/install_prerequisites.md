# Installation prerequisites

`pip install tt_symbiote` lands a pure-Python package; the **hardware
backend it drives is not pip-resolvable**. Specifically, `ttnn` (the
Tenstorrent Neural Network runtime) JIT-compiles firmware kernels for
your Tensix cores at first `open_mesh_device(...)` call, and that JIT
uses a *system-installed* RISC-V cross-compiler called `sfpi`. Each
PyPI `ttnn` wheel pins exactly one supported sfpi version inside
`ttnn/tt_metal/sfpi-version`; if the host's installed sfpi disagrees,
device open fails at runtime with `unrecognized command-line option`.

This document is the authoritative reference for that prerequisite —
what to install, how to verify it, and which `ttnn` wheel matches
which sfpi version. The README's `Installation` section links here;
contributors using [`scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh)
get the same checks enforced automatically.

## What sfpi is and why pip can't install it

`sfpi` is Tenstorrent's RISC-V GCC fork. It targets the small RISC-V
cores embedded on each Tensix tile and provides the SIMD intrinsics
(`SFPU`) that `ttnn` uses for fused matmul / norm / softmax kernels.
The C++ source for each `ttnn` op is JIT-compiled by `sfpi-g++` the
first time the op fires on device — so the toolchain has to be
present at *runtime*, not just install time.

`pip` ships Python wheels and ELF binaries; it cannot install a
RISC-V cross-compiler that lives at `/opt/tenstorrent/sfpi/`. That
path is owned by the Tenstorrent system installer
(`tt-installer` / the bundled apt packages), the same way CUDA toolkit
is a system prerequisite of `torch.cuda` rather than a pip dep.

## How to install sfpi

The official path is the **tt-installer** flow documented at
<https://docs.tenstorrent.com/>. The relevant component is the
`sfpi` apt package, which lays down:

- `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++` (the JIT)
- `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-as`
- `/opt/tenstorrent/sfpi/include/` (intrinsics headers)
- `/opt/tenstorrent/sfpi/lib/` (target libstdc++)

Direct sfpi releases live at <https://github.com/tenstorrent/sfpi/releases>
if you need to install a specific version standalone (e.g., when
bumping the `(ttnn, sfpi)` pair below before the apt repo catches up).

## How to verify sfpi is present and matches `ttnn`

```bash
/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version
```

Look for a line like:

```text
riscv-tt-elf-g++ (sfpi:7.35.3[426]) 12.2.0
```

The string after `sfpi:` (here `7.35.3`) is the canonical sfpi
version. It must match the version pinned by your installed `ttnn`
wheel (next section).

To read the pin from an installed `ttnn`:

```bash
python -c "from importlib.resources import files; print(files('ttnn.tt_metal').joinpath('sfpi-version').read_text())"
# expected output: sfpi_version='7.35.3'  (with whatever quotes ttnn uses)
```

To read the pin from a wheel before installing it:

```bash
pip download --no-deps ttnn==0.68.0 -d /tmp/ttnn-probe
unzip -p /tmp/ttnn-probe/ttnn-*.whl ttnn/tt_metal/sfpi-version
```

## (ttnn, sfpi) compatibility table

The canonical source of truth is [`scripts/ttnn-pin.txt`](../scripts/ttnn-pin.txt),
which the bootstrap script reads at venv-build time. The PyPI extra
[`tt_symbiote[ttnn]`](../pyproject.toml) is pinned to match this table.

| `ttnn` (PyPI) | required `sfpi` | Verified end-to-end | Notes                                  |
|---------------|-----------------|---------------------|----------------------------------------|
| `0.68.0`      | `7.35.3`        | May 2026 (T3K, N150) | The current happy path; see Phase 8 Wave A + B. |

When bumping the pair:

1. Find the sfpi version installed on the target hosts:
   `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version`.
2. Find the matching `ttnn` PyPI release by inspecting its wheel
   (`pip download --no-deps ttnn==<v> -d /tmp/probe; unzip -p
   /tmp/probe/ttnn-*.whl ttnn/tt_metal/sfpi-version`).
3. Update `scripts/ttnn-pin.txt` and `pyproject.toml`'s `[ttnn]`
   extra **in lockstep**. The release-process doc covers this.

## Troubleshooting

### `unrecognized command-line option ...` from `open_mesh_device(...)`

This is the canonical signature of the (ttnn, sfpi) mismatch.
Steps:

1. Print the installed sfpi version (`riscv-tt-elf-g++ --version`).
2. Print the ttnn-pinned sfpi version (`unzip -p` recipe above, or
   `python -c "from importlib.resources import files; ..."`).
3. If they disagree, *either* upgrade/downgrade the system sfpi to
   match the ttnn pin, *or* `pip install` a different ttnn version
   whose pin matches your sfpi.

### `ModuleNotFoundError: No module named 'ttnn'` from `import tt_symbiote`

`ttnn` is not in the hard dependencies of `tt_symbiote` (only in the
`[ttnn]` extra) — `pip install tt_symbiote` alone doesn't pull it.
Either `pip install "tt_symbiote[ttnn]"` or `pip install ttnn==<v>`
explicitly.

### `ModuleNotFoundError: No module named 'tracy'`

`tracy` is Tenstorrent's profiler. It is not on PyPI and only ships
via a tt-metal source build.

Since `tt_symbiote` 0.1.0 the `from tracy import signpost` import is
guarded by a `try/except` in `tt_symbiote.core.run_config` and falls
back to a no-op shim when tracy is missing. `import tt_symbiote`
therefore does *not* require tracy to be installed. The shim is only
activated when the env var `TT_SYMBIOTE_SIGNPOST_MODE` is set, so
unless you're explicitly profiling there is nothing to do.

If you previously saw this error from a tt_symbiote < 0.1.0 install,
upgrade with `pip install -U tt_symbiote`.

## Where to go from here

- [`README.md`](../README.md#installation) — the user-facing install snippet.
- [`scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh) — the
  validated end-to-end install flow for contributors.
- [`scripts/ttnn-pin.txt`](../scripts/ttnn-pin.txt) — the canonical
  `(ttnn, sfpi)` pair.
- [`docs/release_process.md`](release_process.md) — how a new
  `tt_symbiote` release rolls out, including how the `[ttnn]` extra
  pin gets bumped.
