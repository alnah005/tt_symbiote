#!/usr/bin/env python3
"""One-shot file merger for the Phase 2 migration.

Concatenates multiple source files into a single `modeling_<model>.py`
under `src/tt_symbiote/models/<model>/`, deduplicating import lines and
inserting clear section markers so the origin of each block is traceable.

Layout:
  - First file's leading header (SPDX block) is kept verbatim.
  - Top-of-file imports across all inputs are collected, deduplicated by
    exact-string equality, and emitted in a single block right after the
    header.
  - Each file's body (non-import, non-header content) is emitted after a
    `# === content from <original-relative-path> ===` divider.

Idempotent: re-running with the same inputs produces the same output.

Invoke from the repo root.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
STAGING = REPO / "_staging"
DEST = REPO / "src" / "tt_symbiote" / "models"


# (model_dir_name, list of staging-relative source files, in concatenation order)
MERGES: list[tuple[str, list[str]]] = [
    (
        "bailing_moe_v2",
        ["models/bailing_moe_v2.py", "modules/decoder_layer.py"],
    ),
    (
        "gemma4",
        [
            "models/gemma4_text.py",
            "modules/gemma4_modules.py",
            "modules/gemma4_attention.py",
            "modules/gemma4_mlp.py",
        ],
    ),
    (
        "qwen3_moe",
        ["modules/qwen_attention.py", "modules/qwen_moe.py"],
    ),
]


# Top-level imports only (no leading whitespace). Indented imports inside
# function bodies / classes must stay where they are.
IMPORT_LINE = re.compile(r"^(?:from\s+\S+\s+import\s+|import\s+\S+)")
SPDX_LINE = re.compile(r"^#\s*(SPDX-|Vendored\b|Copyright\b)")


def split_file(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Split a Python source file into (header, imports, body) line lists.

    - header: leading comment/SPDX/blank lines, up to the first non-comment
      non-blank line.
    - imports: each top-of-line `from X import ...` / `import X`, including
      multi-line variants like `from X import (\\n  a,\\n  b,\\n)` — the entire
      bracketed continuation is captured as a single import "block" stored as
      one joined string.
    - body: everything else, in original order.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    # Detect the header block: leading blank lines + comment lines.
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            header_end = i + 1
        else:
            break
    header = lines[:header_end]

    imports: list[str] = []
    body: list[str] = []
    i = header_end
    while i < len(lines):
        line = lines[i]
        if IMPORT_LINE.match(line):
            # Consume continuation lines if this is a bracketed import or
            # uses a backslash continuation.
            depth = line.count("(") - line.count(")")
            block = line
            j = i + 1
            while depth > 0 or block.rstrip().endswith("\\"):
                if j >= len(lines):
                    break
                block += lines[j]
                depth += lines[j].count("(") - lines[j].count(")")
                if depth == 0 and not lines[j].rstrip().endswith("\\"):
                    j += 1
                    break
                j += 1
            imports.append(block)
            i = j
        else:
            body.append(line)
            i += 1
    return header, imports, body


def dedupe_preserving_order(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def merge(model_dir: str, sources: list[str]) -> None:
    abs_sources = [STAGING / s for s in sources]
    for p in abs_sources:
        if not p.is_file():
            raise FileNotFoundError(p)

    out_dir = DEST / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "__init__.py").touch()
    out_path = out_dir / f"modeling_{model_dir}.py"

    # Use the first source's header verbatim, then add a Phase-2 merge note.
    first_header, _, _ = split_file(abs_sources[0])
    pieces: list[str] = []
    pieces.extend(first_header)
    if first_header and first_header[-1].strip() != "":
        pieces.append("\n")
    pieces.append("# This file was assembled during the Phase 2 mechanical migration\n")
    pieces.append("# (see scripts/merge_model_files.py). It concatenates the following\n")
    pieces.append("# original sources in order, deduplicating top-of-file imports:\n")
    for s in sources:
        pieces.append(f"#   - models/experimental/tt_symbiote/{s}\n")
    pieces.append("\n")

    # The dotted module path of the file being assembled. Any top-level
    # import that targets this same module is dropped (it's a self-import
    # produced by the codemod rewriting sibling imports to the merge target;
    # after merging, the symbols are defined locally).
    self_module = f"tt_symbiote.models.{model_dir}.modeling_{model_dir}"
    self_import_re = re.compile(rf"^from\s+{re.escape(self_module)}\s+import\b|^import\s+{re.escape(self_module)}\b")

    # Collect + dedupe imports across all sources, dropping self-imports.
    all_imports: list[str] = []
    bodies: list[tuple[str, list[str]]] = []
    dropped_self = 0
    for s, p in zip(sources, abs_sources):
        _, imports, body = split_file(p)
        for imp in imports:
            if self_import_re.match(imp):
                dropped_self += 1
                continue
            all_imports.append(imp)
        bodies.append((s, body))
    pieces.extend(dedupe_preserving_order(all_imports))
    pieces.append("\n")

    # Bodies with section markers.
    for s, body in bodies:
        pieces.append(f"# === content from models/experimental/tt_symbiote/{s} ===\n")
        # Strip leading blanks from each body to avoid noisy whitespace blocks.
        while body and body[0].strip() == "":
            body.pop(0)
        pieces.extend(body)
        if pieces and not pieces[-1].endswith("\n"):
            pieces.append("\n")
        pieces.append("\n")

    out_path.write_text("".join(pieces))
    print(
        f"merged -> {out_path.relative_to(REPO)}  (from {len(sources)} sources, "
        f"{len(dedupe_preserving_order(all_imports))} unique imports, "
        f"{dropped_self} self-imports dropped)"
    )


def main() -> int:
    if not STAGING.is_dir():
        print(f"error: {STAGING} does not exist")
        return 2
    for model_dir, sources in MERGES:
        merge(model_dir, sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
