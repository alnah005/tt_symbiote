# Installation prerequisites

`pip install tt_symbiote` lands a pure-Python package; the **hardware
backend it drives is not pip-resolvable**. Specifically, `ttnn` (the
Tenstorrent Neural Network runtime) JIT-compiles firmware kernels for
your Tensix cores at first `open_mesh_device(...)` call, and that JIT
uses a *system-installed* RISC-V cross-compiler called `sfpi`. `ttnn`
is built from the tt-metal source tree at `$TT_METAL_HOME` (no PyPI
wheel, no global commit pin); its required sfpi version is recorded in
`$TT_METAL_HOME/tt_metal/sfpi-version`. If the host's installed sfpi
disagrees, device open fails at runtime with
`unrecognized command-line option`.

This document is the deeper sfpi/host-prerequisites reference — what
sfpi is, how to install it, and how to verify it matches your
`$TT_METAL_HOME` build. The root [`README.md`](../README.md#installation)
owns the user-facing install flow and links here; contributors using
[`scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh) get the
same checks enforced automatically.

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
if you need to install a specific version standalone (e.g., when your
host sfpi predates the version your `$TT_METAL_HOME` build requires).

## How to verify sfpi is present and matches your tt-metal build

```bash
/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ --version
```

Look for a line like:

```text
riscv-tt-elf-g++ (sfpi:7.52.0[426]) 12.2.0
```

The string after `sfpi:` (here `7.52.0`) is the canonical sfpi
version. It must match the version your `$TT_METAL_HOME` build
requires.

To read the required version from your tt-metal source build:

```bash
cat $TT_METAL_HOME/tt_metal/sfpi-version
# e.g. sfpi_version='7.52.0'
```

## Which sfpi version applies

sfpi is **auto-derived** from `$TT_METAL_HOME/tt_metal/sfpi-version`
(format `sfpi_version='X.Y.Z'`). ttnn is a source build, not a PyPI
package, and there is no global tt-metal commit pin: each model records
its own `tt_metal_commit`, and the source build at `$TT_METAL_HOME`
determines the required sfpi.

[`scripts/ttnn-pin.txt`](../scripts/ttnn-pin.txt) holds the **optional
SFPI override** only (`SFPI_REQUIRED`, empty by default = auto-derive);
it pins no ttnn commit. Set `SFPI_REQUIRED` only to assert or override
the auto-derived value.

## Troubleshooting

### `unrecognized command-line option ...` from `open_mesh_device(...)`

This is the canonical signature of the sfpi mismatch. Steps:

1. Print the installed sfpi version (`riscv-tt-elf-g++ --version`).
2. Print the version your build requires
   (`cat $TT_METAL_HOME/tt_metal/sfpi-version`).
3. If they disagree, install the system sfpi version that matches your
   `$TT_METAL_HOME` build (apt package or a standalone sfpi release).

### `ModuleNotFoundError: No module named 'ttnn'` from `import tt_symbiote`

`ttnn` is not a PyPI dependency — it comes from a tt-metal source build
at `$TT_METAL_HOME`. `import tt_symbiote` auto-wires that source build
onto `sys.path`. This error means `$TT_METAL_HOME` is unset, or its
checkout has no built `ttnn`. Set `$TT_METAL_HOME` to a built tt-metal
checkout (or run [`scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh),
which writes a persistent `.pth`).

### `ImportError: ... requires the Torchvision library but it was not found`

Symptom on `AutoProcessor.from_pretrained("google/gemma-4-...-it")`
or any other HuggingFace VLM:

```
ImportError: Gemma4VideoProcessor requires the Torchvision library but it
was not found in your environment.
```

HF's multimodal `AutoProcessor` constructs a `*VideoProcessor`
subclass that unconditionally imports `torchvision`. Torchvision is
NOT a mandatory `transformers` dep — HF themselves put it in their
own `[vision]` extra — so `tt_symbiote` follows that pattern.

Fix: reinstall with the `[vision]` extra.

```bash
pip install "tt_symbiote[vision]"
```

The extra simply pulls `torchvision`; everything else stays identical.
Text-only causal LMs do NOT need this extra.

### `ModuleNotFoundError: No module named 'tracy'`

`tracy` is Tenstorrent's profiler. It is not on PyPI and only ships
via a tt-metal source build.

The `from tracy import signpost` import is guarded by a `try/except` in
`tt_symbiote.core.run_config` and falls back to a no-op shim when tracy
is missing. `import tt_symbiote` therefore does *not* require tracy to
be installed. The shim is only activated when the env var
`TT_SYMBIOTE_SIGNPOST_MODE` is set, so unless you're explicitly
profiling there is nothing to do.

## Where to go from here

- [`README.md`](../README.md#installation) — the user-facing install snippet.
- [`scripts/bootstrap_venv.sh`](../scripts/bootstrap_venv.sh) — the
  validated end-to-end install flow for contributors.
- [`scripts/ttnn-pin.txt`](../scripts/ttnn-pin.txt) — the optional SFPI
  override (pins no ttnn commit).
- [`docs/development/release_process.md`](development/release_process.md) — how a new
  `tt_symbiote` release rolls out.
