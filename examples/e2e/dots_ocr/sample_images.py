# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic sample-document generation for the dots.ocr batched OCR demo.

Renders distinct, OCR-able legal documents (a single paragraph + effective /
execution dates + a signature block, unique per page) at a FIXED size so every
page in a batch yields the SAME vision patch grid -> one shared trace, no aspect
distortion. Used by ``run_dots_ocr.py``'s ``--batched`` default when no images
are supplied; each page carries a unique reference code ``AGR-<n>-<idx>`` so the
per-stream OCR output is individually verifiable.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

_FONT_CACHE: dict = {}

LEGAL_PARTIES = [
    ("Acme Holdings Ltd.", "Globex Industries LLC"),
    ("Initech Corporation", "Umbrella Group Inc."),
    ("Stark Logistics S.A.", "Wayne Enterprises Ltd."),
    ("Soylent Foods Co.", "Hooli Technologies Inc."),
    ("Wonka Confectionery", "Tyrell Biotech LLC"),
    ("Cyberdyne Systems", "Aperture Science Ltd."),
    ("Nakatomi Trading Co.", "Oscorp Ventures Inc."),
    ("Massive Dynamic LLC", "Pied Piper Holdings"),
]


def _find_font(size: int, bold: bool = False):
    import glob

    from PIL import ImageFont

    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]
    font = None
    for nm in names:
        for p in glob.glob(f"/usr/share/fonts/**/{nm}", recursive=True):
            try:
                font = ImageFont.truetype(p, size)
                break
            except Exception:
                continue
        if font is not None:
            break
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def synthetic_doc_image(idx: int, size):
    """Render a distinct, OCR-able legal sample document (unique per idx).

    ``size`` is the (W, H) canvas; the caller picks it so the page lands on the
    desired vision patch grid. A single legal paragraph + dates + signature block,
    sized so dots.ocr generates roughly 150-200 tokens.
    """
    from PIL import Image, ImageDraw

    w, h = size
    party_a, party_b = LEGAL_PARTIES[idx % len(LEGAL_PARTIES)]
    ref = f"AGR-{1000 + idx * 7}-{idx:02d}"
    amount = f"{(idx + 1) * 125_000:,}"
    eff_date = f"{(idx % 28) + 1:02d} March 2026"
    exec_date = f"{(idx % 27) + 2:02d} March 2026"

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_f, head_f, body_f = _find_font(46, bold=True), _find_font(30, bold=True), _find_font(28)

    margin = 64
    x, y = margin, 52
    d.text((x, y), f"MEMORANDUM OF AGREEMENT  No. {ref}", fill="black", font=title_f)
    y += 70
    d.line((x, y, w - margin, y), fill="black", width=3)
    y += 30

    paragraph = (
        f"This Memorandum of Agreement is made and entered into on {eff_date}, by and between "
        f'{party_a} ("Party A") and {party_b} ("Party B"). The parties hereby agree that the '
        f"obligations set forth in Schedule {idx + 1} shall be performed in full, in consideration "
        f"of the sum of USD {amount}, payable within thirty (30) days of the Effective Date. Each "
        f"party represents and warrants that it has full corporate authority to execute this "
        f"instrument, that the terms herein constitute the entire agreement between the parties, and "
        f"that they supersede all prior negotiations, understandings, and representations, whether "
        f"written or oral. This Agreement shall be governed by the laws of the State of Delaware."
    )
    # Wrap to the page width (landscape -> ~96 chars/line at this font).
    for line in textwrap.wrap(paragraph, width=96):
        d.text((x, y), line, fill="black", font=body_f)
        y += 38
    y += 18

    d.text((x, y), f"Effective Date: {eff_date}.        Executed on: {exec_date}.", fill="black", font=head_f)
    y += 64
    d.text(
        (x, y),
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date above.",
        fill="black",
        font=body_f,
    )
    y += 60
    # Signature block.
    d.text((x, y), "Signed: ______________________", fill="black", font=body_f)
    d.text((x + 880, y), "Signed: ______________________", fill="black", font=body_f)
    y += 40
    d.text((x, y), f"Name: {party_a}, Party A", fill="black", font=body_f)
    d.text((x + 880, y), f"Name: {party_b}, Party B", fill="black", font=body_f)
    return img


def default_batch_specs(n: int, cache_dir, size) -> list[str]:
    """Generate n same-size synthetic sample documents; return their file paths."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = cache_dir / f"sample_doc_{i + 1:02d}.png"
        synthetic_doc_image(i, size).save(p)
        paths.append(str(p))
    print(f"  (no --images given; generated {n} same-size synthetic sample docs in {cache_dir})")
    return paths
