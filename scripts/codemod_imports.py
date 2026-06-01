#!/usr/bin/env python3
"""One-shot import rewriter for the Phase 2 migration.

Rewrites tt-metal-style imports in a target directory tree to tt_symbiote-style
imports, in preparation for Phase 2's reorganization.

Rules (applied per line, in order):
  1. Drop any `from models.experimental.tt_symbiote.core.dispatchers...` line
     entirely. Phase 3 deletes the dispatcher subsystem; we anticipate by
     stripping the imports here so the tree compiles after Phase 2.
  2. Rewrite `models.experimental.tt_symbiote` -> `tt_symbiote` everywhere.
  3. Rewrite `models.tt_transformers.tt.ccl` -> `tt_symbiote.core.ccl`.
  4. Rewrite `models.tt_cnn.tt` -> `tt_symbiote.integrations.tt_cnn`
     (tt_cnn is also vendored; see src/tt_symbiote/integrations/tt_cnn/).

Invariants:
  - Files outside `*.py` are not touched.
  - The script edits in place; intended to run against the `_staging/` directory.
  - Reports per-file change counts to stdout.

Usage:
    python scripts/codemod_imports.py _staging/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DROP_PATTERN = re.compile(r"^\s*from\s+models\.experimental\.tt_symbiote\.core\.dispatchers")
TTSYMB_SUB = (
    re.compile(r"\bmodels\.experimental\.tt_symbiote\b"),
    "tt_symbiote",
)
TTCCL_SUB = (
    re.compile(r"\bmodels\.tt_transformers\.tt\.ccl\b"),
    "tt_symbiote.core.ccl",
)
TTCNN_SUB = (
    re.compile(r"\bmodels\.tt_cnn\.tt\b"),
    "tt_symbiote.integrations.tt_cnn",
)

# Intra-package remapping: the flat `modules/` and `models/<file>.py` layout
# becomes HF-mirrored. Generic primitives go to `integrations/ttnn_*.py`;
# model-specific code is merged into `models/<model>/modeling_<model>.py`.
# Each entry is (regex, replacement). Applied in order.
MODULE_REMAP: list[tuple[re.Pattern, str]] = [
    # Generic primitives → integrations/
    (re.compile(r"\btt_symbiote\.modules\.linear_intelligent\b"), "tt_symbiote.integrations.ttnn_linear_intelligent"),
    (re.compile(r"\btt_symbiote\.modules\.linear\b"), "tt_symbiote.integrations.ttnn_linear"),
    (re.compile(r"\btt_symbiote\.modules\.normalization\b"), "tt_symbiote.integrations.ttnn_normalization"),
    (re.compile(r"\btt_symbiote\.modules\.activation\b"), "tt_symbiote.integrations.ttnn_activation"),
    (re.compile(r"\btt_symbiote\.modules\.embedding\b"), "tt_symbiote.integrations.ttnn_embedding"),
    (re.compile(r"\btt_symbiote\.modules\.rope\b"), "tt_symbiote.integrations.ttnn_rope"),
    (re.compile(r"\btt_symbiote\.modules\.attention\b"), "tt_symbiote.integrations.ttnn_attention"),
    (re.compile(r"\btt_symbiote\.modules\.moe\b"), "tt_symbiote.integrations.ttnn_moe"),
    (re.compile(r"\btt_symbiote\.modules\.conv\b"), "tt_symbiote.integrations.ttnn_conv"),
    (re.compile(r"\btt_symbiote\.modules\.tensor\b"), "tt_symbiote.integrations.ttnn_tensor"),
    # Model-specific modules → models/<model>/modeling_<model>.py
    (
        re.compile(r"\btt_symbiote\.modules\.decoder_layer\b"),
        "tt_symbiote.models.bailing_moe_v2.modeling_bailing_moe_v2",
    ),
    (re.compile(r"\btt_symbiote\.modules\.gemma4_attention\b"), "tt_symbiote.models.gemma4.modeling_gemma4"),
    (re.compile(r"\btt_symbiote\.modules\.gemma4_mlp\b"), "tt_symbiote.models.gemma4.modeling_gemma4"),
    (re.compile(r"\btt_symbiote\.modules\.gemma4_modules\b"), "tt_symbiote.models.gemma4.modeling_gemma4"),
    (re.compile(r"\btt_symbiote\.modules\.qwen_attention\b"), "tt_symbiote.models.qwen3_moe.modeling_qwen3_moe"),
    (re.compile(r"\btt_symbiote\.modules\.qwen_moe\b"), "tt_symbiote.models.qwen3_moe.modeling_qwen3_moe"),
    # Top-level model files → per-model subpackage.
    # Negative lookahead avoids double-substitution: if `models.bailing_moe_v2`
    # is already followed by `.modeling_bailing_moe_v2`, skip it.
    (
        re.compile(r"\btt_symbiote\.models\.bailing_moe_v2\b(?!\.modeling_)"),
        "tt_symbiote.models.bailing_moe_v2.modeling_bailing_moe_v2",
    ),
    (re.compile(r"\btt_symbiote\.models\.gemma4_text\b"), "tt_symbiote.models.gemma4.modeling_gemma4"),
]


def rewrite_file(path: Path) -> tuple[int, int]:
    """Rewrite a single .py file. Returns (lines_changed, lines_dropped).

    Rule 1 (drop dispatcher imports) handles multi-line imports correctly:
    when a line matching DROP_PATTERN starts a `(...)` import block, the
    entire block (including the closing `)`) is dropped. Backslash
    continuations are also consumed.
    """
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    out_lines: list[str] = []
    changed = 0
    dropped = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        if DROP_PATTERN.match(line):
            # Consume the rest of a possibly multi-line import block.
            depth = line.count("(") - line.count(")")
            dropped += 1
            i += 1
            while depth > 0 or (i > 0 and lines[i - 1].rstrip().endswith("\\")):
                if i >= len(lines):
                    break
                depth += lines[i].count("(") - lines[i].count(")")
                dropped += 1
                i += 1
                if depth <= 0 and not (i < len(lines) and lines[i - 1].rstrip().endswith("\\")):
                    break
            continue
        new_line = line
        n_total = 0
        new_line, n1 = TTSYMB_SUB[0].subn(TTSYMB_SUB[1], new_line)
        new_line, n2 = TTCCL_SUB[0].subn(TTCCL_SUB[1], new_line)
        new_line, n3 = TTCNN_SUB[0].subn(TTCNN_SUB[1], new_line)
        n_total = n1 + n2 + n3
        for pat, repl in MODULE_REMAP:
            new_line, n = pat.subn(repl, new_line)
            n_total += n
        if n_total:
            changed += 1
        out_lines.append(new_line)
        i += 1

    new_text = "".join(out_lines)
    if new_text != original:
        path.write_text(new_text)
    return changed, dropped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root directory to rewrite (e.g. _staging/).")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    total_files = 0
    total_changed_files = 0
    total_changed = 0
    total_dropped = 0
    for py in sorted(args.root.rglob("*.py")):
        total_files += 1
        changed, dropped = rewrite_file(py)
        if changed or dropped:
            total_changed_files += 1
            total_changed += changed
            total_dropped += dropped
            print(f"  {py.relative_to(args.root)}: {changed} import lines rewritten, {dropped} dropped")

    print(
        f"\nScanned {total_files} .py files; modified {total_changed_files}; "
        f"{total_changed} import lines rewritten; {total_dropped} dispatcher-import lines dropped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
