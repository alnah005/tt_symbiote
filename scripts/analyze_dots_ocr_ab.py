# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Score and compare dots.ocr vision-precision A/B/C demo runs.

Reads one results dir per tier (produced by run_dots_ocr_ab.sh / the demo), scores
each page's OCR text against the ground-truth silver reference (CER / WER / similarity,
raw and whitespace-normalized), and pulls latency/throughput/token metrics from each
run's results.json. Emits:

    <out>/summary.json   machine-readable: per-tier aggregate + per-page metrics
    <out>/summary.md     human-readable comparison tables
    <out>/diffs/         per-page unified diffs (baseline vs each other tier)

No third-party deps: CER/WER use a self-contained edit-distance DP; similarity uses
difflib. Absolute CER vs the silver reference is not the point -- the DELTA between
tiers (and per-page regressions vs baseline) is the accuracy signal.

Usage:
    python scripts/analyze_dots_ocr_ab.py \
        --results-root /home/aroberge/t1_pages/results \
        --ground-truth /home/aroberge/t1_pages/ground_truth \
        --tiers baseline hifi4 bf16 \
        --out /home/aroberge/t1_pages/results/ab_summary
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #
def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance over two sequences (chars or tokens), O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[-1]


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return re.sub(r"\s+", " ", s).strip()


def score_pair(hyp: str, ref: str) -> dict:
    """CER / WER / similarity for one (hypothesis, reference) pair, raw + normalized."""
    hyp = hyp or ""
    ref = ref or ""
    hn, rn = _normalize(hyp), _normalize(ref)

    def cer(h, r):
        return (_edit_distance(list(h), list(r)) / len(r)) if r else (0.0 if not h else 1.0)

    def wer(h, r):
        ht, rt = h.split(), r.split()
        return (_edit_distance(ht, rt) / len(rt)) if rt else (0.0 if not ht else 1.0)

    return {
        "cer_raw": round(cer(hyp, ref), 4),
        "cer_norm": round(cer(hn, rn), 4),
        "wer_norm": round(wer(hn, rn), 4),
        "similarity_norm": round(difflib.SequenceMatcher(None, hn, rn).ratio(), 4),
        "hyp_chars": len(hyp),
        "ref_chars": len(ref),
    }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ground_truth(gt_dir: Path) -> dict:
    gt = {}
    for p in sorted(gt_dir.glob("*.txt")):
        gt[p.stem] = p.read_text(encoding="utf-8", errors="replace")
    return gt


def _page_key(image_name: str) -> str:
    return Path(image_name).stem


def load_tier(results_root: Path, tier: str) -> dict:
    """Return {status, meta, pages:{stem:{text, perf...}}} for one tier dir."""
    d = results_root / f"ab_{tier}"
    if (d / "INFEASIBLE.json").exists():
        info = json.loads((d / "INFEASIBLE.json").read_text())
        return {"status": info.get("status", "infeasible"), "meta": {}, "pages": {}}
    rj = d / "results.json"
    if not rj.exists():
        return {"status": "missing", "meta": {}, "pages": {}}
    data = json.loads(rj.read_text(encoding="utf-8", errors="replace"))
    pages = {}
    for r in data.get("results", []):
        stem = _page_key(r.get("image", ""))
        pages[stem] = {
            "text": r.get("text", "") or "",
            "ok": r.get("ok", None),
            "finish_reason": r.get("finish_reason"),
            "latency_s": r.get("latency_s"),
            "tokens_per_s": r.get("tokens_per_s"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
        }
    return {"status": "ok", "meta": data.get("meta", {}), "pages": pages}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def analyze(results_root: Path, gt_dir: Path, tiers: list[str]) -> dict:
    gt = load_ground_truth(gt_dir)
    loaded = {t: load_tier(results_root, t) for t in tiers}
    all_stems = sorted(gt.keys())

    per_tier = {}
    for tier in tiers:
        info = loaded[tier]
        if info["status"] != "ok":
            per_tier[tier] = {"status": info["status"]}
            continue
        pages = {}
        for stem in all_stems:
            pg = info["pages"].get(stem)
            if pg is None:
                pages[stem] = {"status": "no_output"}
                continue
            sc = score_pair(pg["text"], gt.get(stem, ""))
            sc.update(
                {
                    "ok": pg["ok"],
                    "finish_reason": pg["finish_reason"],
                    "latency_s": pg["latency_s"],
                    "tokens_per_s": pg["tokens_per_s"],
                    "completion_tokens": pg["completion_tokens"],
                }
            )
            pages[stem] = sc
        scored = [p for p in pages.values() if "cer_norm" in p]
        agg = {
            "pages_scored": len(scored),
            "cer_raw_mean": _mean([p["cer_raw"] for p in scored]),
            "cer_norm_mean": _mean([p["cer_norm"] for p in scored]),
            "wer_norm_mean": _mean([p["wer_norm"] for p in scored]),
            "similarity_norm_mean": _mean([p["similarity_norm"] for p in scored]),
            "avg_latency_s": info["meta"].get("avg_latency") or _mean([p.get("latency_s") for p in scored]),
            "avg_tps": info["meta"].get("avg_tps") or _mean([p.get("tokens_per_s") for p in scored]),
            "aggregate_tps": info["meta"].get("aggregate_tps"),
            "wall_clock_s": info["meta"].get("wall_clock_s"),
            "completion_tokens_mean": _mean([p.get("completion_tokens") for p in scored]),
            "finish_reasons": _fr_counts(scored),
        }
        per_tier[tier] = {"status": "ok", "aggregate": agg, "pages": pages}

    return {"tiers": tiers, "stems": all_stems, "per_tier": per_tier}


def _fr_counts(scored):
    c = {}
    for p in scored:
        fr = p.get("finish_reason") or "none"
        c[fr] = c.get(fr, 0) + 1
    return c


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(v, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def render_markdown(res: dict, baseline: str) -> str:
    tiers = res["tiers"]
    pt = res["per_tier"]
    lines = ["# dots.ocr Vision-Encoder Precision A/B/C Results", ""]

    lines += ["## Per-tier aggregate", ""]
    hdr = "| Tier | Status | CER(norm)↓ | WER(norm)↓ | Similarity↑ | Avg latency (s)↓ | Avg TPS↑ | Agg TPS↑ | Wall (s) | Compl.tok |"
    lines += [hdr, "|" + "---|" * 10]
    for t in tiers:
        e = pt[t]
        if e.get("status") != "ok":
            lines.append(f"| {t} | **{e.get('status')}** | — | — | — | — | — | — | — | — |")
            continue
        a = e["aggregate"]
        lines.append(
            f"| {t} | ok | {_fmt(a['cer_norm_mean'], pct=True)} | {_fmt(a['wer_norm_mean'], pct=True)} | "
            f"{_fmt(a['similarity_norm_mean'], pct=True)} | {_fmt(a['avg_latency_s'])} | {_fmt(a['avg_tps'])} | "
            f"{_fmt(a['aggregate_tps'])} | {_fmt(a['wall_clock_s'])} | {_fmt(a['completion_tokens_mean'])} |"
        )
    lines.append("")

    # Deltas vs baseline
    if pt.get(baseline, {}).get("status") == "ok":
        b = pt[baseline]["aggregate"]
        lines += ["", f"## Change vs baseline (`{baseline}`)", ""]
        lines += ["| Tier | ΔCER(norm) | ΔSimilarity | ΔAvg latency (s) | ΔAvg TPS |", "|---|---|---|---|---|"]
        for t in tiers:
            if t == baseline or pt[t].get("status") != "ok":
                continue
            a = pt[t]["aggregate"]

            def d(k, better_lower):
                if a.get(k) is None or b.get(k) is None:
                    return "—"
                dv = a[k] - b[k]
                arrow = ("✓" if (dv < 0) == better_lower else "✗") if abs(dv) > 1e-9 else "="
                return f"{dv:+.4f} {arrow}"

            lines.append(
                f"| {t} | {d('cer_norm_mean', True)} | {d('similarity_norm_mean', False)} | "
                f"{d('avg_latency_s', True)} | {d('avg_tps', False)} |"
            )
        lines.append("")

    # Per-page CER(norm)
    lines += ["## Per-page CER (normalized)", ""]
    lines += ["| Page | " + " | ".join(tiers) + " |", "|" + "---|" * (len(tiers) + 1)]
    for stem in res["stems"]:
        cells = []
        for t in tiers:
            e = pt[t]
            if e.get("status") != "ok":
                cells.append("—")
            else:
                p = e["pages"].get(stem, {})
                cells.append(_fmt(p.get("cer_norm"), pct=True) if "cer_norm" in p else "n/a")
        lines.append(f"| {stem} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_diffs(res: dict, results_root: Path, gt_dir: Path, baseline: str, out: Path):
    diffs = out / "diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    tiers = res["tiers"]
    texts = {t: load_tier(results_root, t)["pages"] for t in tiers}
    for stem in res["stems"]:
        base_txt = (texts.get(baseline, {}).get(stem) or {}).get("text", "")
        for t in tiers:
            if t == baseline:
                continue
            other = (texts.get(t, {}).get(stem) or {}).get("text", "")
            if not base_txt and not other:
                continue
            ud = difflib.unified_diff(
                base_txt.splitlines(), other.splitlines(),
                fromfile=f"{baseline}/{stem}", tofile=f"{t}/{stem}", lineterm="",
            )
            (diffs / f"{stem}__{baseline}_vs_{t}.diff").write_text("\n".join(ud), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score/compare dots.ocr precision A/B/C runs")
    ap.add_argument("--results-root", default="/home/aroberge/t1_pages/results")
    ap.add_argument("--ground-truth", default="/home/aroberge/t1_pages/ground_truth")
    ap.add_argument("--tiers", nargs="+", default=["baseline", "hifi4", "bf16"])
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", default="/home/aroberge/t1_pages/results/ab_summary")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res = analyze(results_root, Path(args.ground_truth), args.tiers)
    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(res, args.baseline)
    (out / "summary.md").write_text(md, encoding="utf-8")
    write_diffs(res, results_root, Path(args.ground_truth), args.baseline, out)

    print(md)
    print(f"\nWrote: {out}/summary.json, {out}/summary.md, {out}/diffs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
