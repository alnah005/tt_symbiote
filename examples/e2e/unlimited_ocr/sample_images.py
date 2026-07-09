# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic sample-document generation for the Unlimited-OCR demo.

Renders GENUINELY DIFFERENT readable, OCR-able documents so the demo can run
end-to-end with no network access and no user-supplied image, and so each image
produces a DISTINCT OCR output. Each ``idx`` maps to a different document *type*
whose big header/title is the FIRST prominent line on the page -- so the leading
OCR tokens differ immediately (not buried past token 24):

    idx 0 -> INVOICE          (invoice no, bill-to, line items, total)
    idx 1 -> PURCHASE ORDER   (PO number, vendor, ordered items)
    idx 2 -> MEMORANDUM       (to / from / date / subject + body)
    idx 3 -> LEASE AGREEMENT  (landlord / tenant / term / rent)

``idx`` values beyond 3 cycle through the four templates with distinct numbers.
Every page is rendered at 1024x1024; ``run_unlimited_ocr.py`` pads/normalizes it
to the model's 1024x1024 global view at preprocessing time.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

_FONT_CACHE: dict = {}


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


# ---------------------------------------------------------------------------
# Per-document-type renderers. Each takes the ImageDraw, canvas size, a run
# number ``n`` (to vary IDs/amounts across cycles) and the three fonts, and
# draws a distinct document whose FIRST prominent line is its big title.
# ---------------------------------------------------------------------------
def _title(d, x, y, w, margin, text, title_f):
    """Draw the big header title + an underline rule; return the new y."""
    d.text((x, y), text, fill="black", font=title_f)
    y += 66
    d.line((x, y, w - margin, y), fill="black", width=3)
    return y + 26


def _lines(d, x, y, rows, font, step=38):
    for row in rows:
        d.text((x, y), row, fill="black", font=font)
        y += step
    return y


def _para(d, x, y, text, font, width=54, step=34):
    for line in textwrap.wrap(text, width=width):
        d.text((x, y), line, fill="black", font=font)
        y += step
    return y


def _draw_invoice(d, w, h, n, margin, title_f, head_f, body_f):
    x, y = margin, 56
    y = _title(d, x, y, w, margin, "INVOICE", title_f)
    inv_no = f"INV-2026-{4100 + n * 13:04d}"
    total = f"{(n + 1) * 4820:,}.00"
    y = _lines(d, x, y, [
        f"Invoice No: {inv_no}",
        "Invoice Date: 06 March 2026",
        "Due Date: 05 April 2026",
    ], head_f, step=40)
    y += 10
    y = _lines(d, x, y, [
        "Bill To: Acme Holdings Ltd.",
        "         480 Riverside Parkway, Suite 210",
        "         Portland, OR 97201",
    ], body_f, step=36)
    y += 16
    y = _lines(d, x, y, [
        "Description                 Qty     Amount",
        "---------------------------------------------",
        "Consulting services         40      3,200.00",
        "Cloud compute credits       12        960.00",
        "On-site support             4         660.00",
        "---------------------------------------------",
    ], body_f, step=36)
    y += 12
    d.text((x, y), f"TOTAL DUE: USD {total}", fill="black", font=head_f)
    y += 56
    y = _lines(d, x, y, [
        "Remit payment to Acme Holdings Ltd.",
        "Bank: First National, Acct 0042-19837.",
        "Thank you for your business.",
    ], body_f, step=36)


def _draw_purchase_order(d, w, h, n, margin, title_f, head_f, body_f):
    x, y = margin, 56
    y = _title(d, x, y, w, margin, "PURCHASE ORDER", title_f)
    po_no = f"PO-{88200 + n * 31:05d}"
    total = f"{(n + 1) * 15650:,}.00"
    y = _lines(d, x, y, [
        f"PO Number: {po_no}",
        "Order Date: 06 March 2026",
        "Ship Via: Freight, FOB Destination",
    ], head_f, step=40)
    y += 10
    y = _lines(d, x, y, [
        "Vendor: Globex Industries LLC",
        "        1200 Commerce Drive",
        "        Austin, TX 78701",
    ], body_f, step=36)
    y += 16
    y = _lines(d, x, y, [
        "Item   Description              Qty    Unit",
        "---------------------------------------------",
        "A-100  Steel brackets           500    2.40",
        "B-220  Hex bolts, M8            2000   0.15",
        "C-305  Industrial sealant       80     18.00",
        "---------------------------------------------",
    ], body_f, step=36)
    y += 12
    d.text((x, y), f"ORDER TOTAL: USD {total}", fill="black", font=head_f)
    y += 56
    y = _lines(d, x, y, [
        "Approved by: Purchasing Department",
        "Deliver no later than 20 March 2026.",
    ], body_f, step=36)


def _draw_memorandum(d, w, h, n, margin, title_f, head_f, body_f):
    x, y = margin, 56
    y = _title(d, x, y, w, margin, "MEMORANDUM", title_f)
    ref = f"MEMO-{700 + n * 5:03d}"
    y = _lines(d, x, y, [
        "TO:      All Engineering Staff",
        "FROM:    Dana Whitfield, VP of Operations",
        "DATE:    06 March 2026",
        "SUBJECT: Quarterly Maintenance Window",
        f"REF:     {ref}",
    ], head_f, step=42)
    y += 18
    body = (
        "Please be advised that the scheduled quarterly maintenance window "
        "will begin at 22:00 on Friday and conclude by 06:00 Saturday. All "
        "non-critical services will be taken offline during this period. "
        "Engineering teams should complete and merge outstanding changes "
        "before the freeze, and confirm on-call coverage with their leads."
    )
    y = _para(d, x, y, body, body_f, width=52, step=36)
    y += 18
    y = _lines(d, x, y, [
        "Questions may be directed to the Operations desk.",
        "Regards, Dana Whitfield.",
    ], body_f, step=36)


def _draw_lease(d, w, h, n, margin, title_f, head_f, body_f):
    x, y = margin, 56
    y = _title(d, x, y, w, margin, "LEASE AGREEMENT", title_f)
    rent = f"{2400 + n * 150:,}"
    y = _lines(d, x, y, [
        "Landlord: Riverside Property Group",
        "Tenant:   Jordan Ellis",
        "Premises: 88 Maple Court, Unit 4B",
        "Term:     12 months, from 01 April 2026",
        f"Rent:     USD {rent} per month",
    ], head_f, step=42)
    y += 18
    body = (
        "This Lease Agreement is entered into between the Landlord and the "
        "Tenant for the residential premises described above. The Tenant "
        "agrees to pay the monthly rent on the first day of each month and "
        "to maintain the premises in good condition. A security deposit "
        "equal to one month's rent is due upon signing of this agreement."
    )
    y = _para(d, x, y, body, body_f, width=52, step=36)
    y += 18
    y = _lines(d, x, y, [
        "Signed (Landlord): ____________________",
        "Signed (Tenant):   ____________________",
    ], body_f, step=40)


_DOC_RENDERERS = (_draw_invoice, _draw_purchase_order, _draw_memorandum, _draw_lease)
_DOC_STEMS = ("invoice", "purchase_order", "memorandum", "lease_agreement")


def synthetic_doc_image(idx: int = 0, size=(1024, 1024)):
    """Render a single distinct, OCR-able sample document for ``idx``.

    ``idx`` selects one of four genuinely different document types (INVOICE,
    PURCHASE ORDER, MEMORANDUM, LEASE AGREEMENT). The big title is the first
    prominent line, so distinct titles => distinct leading OCR tokens. ``idx``
    values beyond 3 cycle through the templates with distinct numbers.
    """
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_f = _find_font(56, bold=True)
    head_f = _find_font(28, bold=True)
    body_f = _find_font(26)

    which = idx % len(_DOC_RENDERERS)
    cycle = idx // len(_DOC_RENDERERS)  # 0 for the first pass; varies IDs on repeats
    n = idx  # run number drives per-image IDs/amounts (unique even within a type)
    _DOC_RENDERERS[which](d, w, h, n, 56, title_f, head_f, body_f)
    return img


def _stem_for(idx: int) -> str:
    """Human-readable file stem for ``idx`` (e.g. ``01_invoice``)."""
    return f"{idx + 1:02d}_{_DOC_STEMS[idx % len(_DOC_STEMS)]}"


def default_sample_spec(cache_dir, size=(1024, 1024)) -> str:
    """Generate a single synthetic sample document; return its file path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"sample_doc_{_stem_for(0)}.png"
    synthetic_doc_image(0, size).save(p)
    print(f"  (no --image given; generated a synthetic sample doc at {p})")
    return str(p)


def default_batch_specs(n: int, cache_dir, size=(1024, 1024)) -> list[str]:
    """Generate ``n`` genuinely DIFFERENT synthetic sample documents; return paths.

    Each page is rendered by ``synthetic_doc_image(idx)`` as a distinct document
    *type* (INVOICE / PURCHASE ORDER / MEMORANDUM / LEASE AGREEMENT, cycling for
    ``idx > 3``) whose big title is the FIRST prominent line -- so every image
    OCRs to a DISTINCT output. Mirrors the dots.ocr demo's ``default_batch_specs``
    (the Unlimited-OCR demo OCRs these SEQUENTIALLY on a single P150; every page
    is padded to the model's 1024x1024 global view at preprocessing time).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(max(1, int(n))):
        p = cache_dir / f"sample_doc_{_stem_for(i)}.png"
        synthetic_doc_image(i, size).save(p)
        paths.append(str(p))
    print(f"  (no --image given; generated {len(paths)} distinct synthetic sample docs in {cache_dir})")
    return paths
