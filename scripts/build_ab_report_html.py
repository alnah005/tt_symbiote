# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Render the dots.ocr precision A/B/C summary.json into a self-contained HTML report.

Reads <summary_dir>/summary.json (from analyze_dots_ocr_ab.py) and writes
<summary_dir>/report.html: a theme-aware, dependency-free page with the per-tier
aggregate table, change-vs-baseline table, per-page CER heat table, finish-reason
badges, and an auto-generated conclusion. Intended to be published as an Artifact.

Usage: python scripts/build_ab_report_html.py --summary-dir /home/aroberge/t1_pages/results/ab_summary
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _pct(v):
    return "—" if v is None else f"{v * 100:.2f}%"


def _num(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def _cer_color(v):
    if v is None:
        return "transparent"
    # green (low CER) -> red (high CER); clamp at 0.30
    t = max(0.0, min(1.0, v / 0.30))
    r = int(40 + t * 180)
    g = int(170 - t * 120)
    return f"rgba({r},{g},70,0.28)"


def build(summary_dir: Path) -> Path:
    res = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    tiers = res["tiers"]
    pt = res["per_tier"]
    stems = res["stems"]
    baseline = "baseline" if "baseline" in tiers else tiers[0]

    def agg(t, k):
        e = pt.get(t, {})
        return e.get("aggregate", {}).get(k) if e.get("status") == "ok" else None

    # ---- aggregate rows ----
    agg_rows = ""
    for t in tiers:
        e = pt.get(t, {})
        if e.get("status") != "ok":
            agg_rows += f'<tr><td>{html.escape(t)}</td><td class="bad">{html.escape(e.get("status","?"))}</td>' + "<td>—</td>" * 8 + "</tr>"
            continue
        a = e["aggregate"]
        agg_rows += (
            f"<tr><td><b>{html.escape(t)}</b></td><td class='ok'>ok</td>"
            f"<td>{_pct(a['cer_norm_mean'])}</td><td>{_pct(a['wer_norm_mean'])}</td>"
            f"<td>{_pct(a['similarity_norm_mean'])}</td><td>{_num(a['avg_latency_s'])}</td>"
            f"<td>{_num(a['avg_tps'])}</td><td>{_num(a['aggregate_tps'])}</td>"
            f"<td>{_num(a['wall_clock_s'])}</td><td>{_num(a['completion_tokens_mean'],0)}</td></tr>"
        )

    # ---- delta rows ----
    delta_rows = ""
    b_cer, b_sim, b_lat, b_tps = (agg(baseline, k) for k in ("cer_norm_mean", "similarity_norm_mean", "avg_latency_s", "avg_tps"))
    for t in tiers:
        if t == baseline or pt.get(t, {}).get("status") != "ok":
            continue

        def delta(cur, base, better_lower):
            if cur is None or base is None:
                return "—", ""
            dv = cur - base
            good = (dv < 0) if better_lower else (dv > 0)
            cls = "good" if abs(dv) > 1e-9 and good else ("bad" if abs(dv) > 1e-9 else "")
            return f"{dv:+.4f}", cls

        c_cer = delta(agg(t, "cer_norm_mean"), b_cer, True)
        c_sim = delta(agg(t, "similarity_norm_mean"), b_sim, False)
        c_lat = delta(agg(t, "avg_latency_s"), b_lat, True)
        c_tps = delta(agg(t, "avg_tps"), b_tps, False)
        delta_rows += (
            f"<tr><td><b>{html.escape(t)}</b></td>"
            f"<td class='{c_cer[1]}'>{c_cer[0]}</td><td class='{c_sim[1]}'>{c_sim[0]}</td>"
            f"<td class='{c_lat[1]}'>{c_lat[0]}</td><td class='{c_tps[1]}'>{c_tps[0]}</td></tr>"
        )
    delta_section = (
        f"<h2>Change vs baseline (<code>{html.escape(baseline)}</code>)</h2>"
        "<table><thead><tr><th>Tier</th><th>ΔCER(norm)</th><th>ΔSimilarity</th>"
        "<th>ΔAvg latency (s)</th><th>ΔAvg TPS</th></tr></thead><tbody>"
        f"{delta_rows}</tbody></table>"
        if delta_rows else ""
    )

    # ---- per-page CER table ----
    head_cells = "".join(f"<th>{html.escape(t)}</th>" for t in tiers)
    page_rows = ""
    for stem in stems:
        cells = ""
        for t in tiers:
            e = pt.get(t, {})
            if e.get("status") != "ok":
                cells += "<td>—</td>"
                continue
            p = e["pages"].get(stem, {})
            if "cer_norm" not in p:
                cells += "<td class='bad'>n/a</td>"
            else:
                v = p["cer_norm"]
                fr = p.get("finish_reason") or ""
                badge = f"<span class='fr'>{html.escape(str(fr))}</span>" if fr and fr != "stop" else ""
                cells += f"<td style='background:{_cer_color(v)}'>{_pct(v)}{badge}</td>"
        page_rows += f"<tr><td>{html.escape(stem)}</td>{cells}</tr>"

    # ---- conclusion ----
    ok_tiers = [t for t in tiers if pt.get(t, {}).get("status") == "ok"]
    concl = []
    if ok_tiers:
        best_acc = min(ok_tiers, key=lambda t: (agg(t, "cer_norm_mean") if agg(t, "cer_norm_mean") is not None else 9))
        fastest = min(ok_tiers, key=lambda t: (agg(t, "avg_latency_s") if agg(t, "avg_latency_s") is not None else 9e9))
        concl.append(f"Lowest CER (best accuracy): <b>{html.escape(best_acc)}</b> at {_pct(agg(best_acc,'cer_norm_mean'))}.")
        concl.append(f"Lowest avg latency: <b>{html.escape(fastest)}</b> at {_num(agg(fastest,'avg_latency_s'))} s/image.")
        if agg(baseline, "cer_norm_mean") is not None and agg(best_acc, "cer_norm_mean") is not None:
            d = (agg(best_acc, "cer_norm_mean") - agg(baseline, "cer_norm_mean")) * 100
            verdict = ("no measurable accuracy gain" if abs(d) < 0.05
                       else f"{'reduces' if d < 0 else 'increases'} CER by {abs(d):.2f} pts vs baseline")
            concl.append(f"Highest-precision-vs-baseline accuracy effect: <b>{verdict}</b>.")
    infeasible = [t for t in tiers if pt.get(t, {}).get("status") not in ("ok", None)]
    if infeasible:
        concl.append("Infeasible/failed arms: " + ", ".join(f"<code>{html.escape(t)}</code> ({html.escape(pt[t]['status'])})" for t in infeasible) + ".")
    concl_html = "".join(f"<li>{c}</li>" for c in concl)

    page = f"""<title>dots.ocr Vision Precision A/B/C</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --head:#f4f4f6; --code:#f0f0f3; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15171c; --fg:#e8e8ea; --muted:#9aa; --line:#333; --head:#20242c; --code:#20242c; }} }}
:root[data-theme=dark] {{ --bg:#15171c; --fg:#e8e8ea; --muted:#9aa; --line:#333; --head:#20242c; --code:#20242c; }}
:root[data-theme=light] {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --head:#f4f4f6; --code:#f0f0f3; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:2rem; }}
.wrap {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .3rem; }} h2 {{ font-size:1.15rem; margin:2rem 0 .6rem; }}
p.sub {{ color:var(--muted); margin:0 0 1rem; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:.5rem 0; }}
th,td {{ border:1px solid var(--line); padding:.4rem .6rem; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
thead th {{ background:var(--head); position:sticky; top:0; }}
code {{ background:var(--code); padding:.1rem .3rem; border-radius:4px; }}
.ok {{ color:#2a8a2a; }} .bad {{ color:#c0392b; font-weight:600; }} .good {{ color:#2a8a2a; font-weight:600; }}
.fr {{ display:inline-block; margin-left:.3rem; font-size:11px; color:var(--bg); background:#c0392b; padding:0 .3rem; border-radius:3px; }}
ul.concl li {{ margin:.25rem 0; }}
</style>
<div class="wrap">
<h1>dots.ocr — Vision-Encoder Precision A/B/C</h1>
<p class="sub">OCR accuracy (vs Turkish silver reference) and latency/throughput per precision tier. CER/WER lower is better; similarity/TPS higher is better.</p>

<h2>Per-tier aggregate</h2>
<div class="scroll"><table><thead><tr>
<th>Tier</th><th>Status</th><th>CER(norm)↓</th><th>WER(norm)↓</th><th>Similarity↑</th>
<th>Avg latency (s)↓</th><th>Avg TPS↑</th><th>Agg TPS↑</th><th>Wall (s)</th><th>Compl. tok</th>
</tr></thead><tbody>{agg_rows}</tbody></table></div>

{delta_section}

<h2>Per-page CER (normalized)</h2>
<p class="sub">Cell shading: green = low error, red = high error. A red badge marks a non-<code>stop</code> finish reason (truncation / error).</p>
<div class="scroll"><table><thead><tr><th>Page</th>{head_cells}</tr></thead><tbody>{page_rows}</tbody></table></div>

<h2>Conclusion</h2>
<ul class="concl">{concl_html}</ul>
</div>
"""
    out = summary_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="/home/aroberge/t1_pages/results/ab_summary")
    args = ap.parse_args()
    out = build(Path(args.summary_dir))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
