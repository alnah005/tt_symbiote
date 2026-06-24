# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""dots.ocr serving determinism / corruption validator (deep-plan_0 §9.1).

Fires identical greedy (temperature=0) OCR requests at a running OpenAI-compatible
dots.ocr endpoint, records the EXACT bytes returned per (page, repeat, concurrency),
and asserts:
  * byte-identity across repeats of the same page (per-request determinism)
  * (optional) byte-identity vs a golden directory
  * absence of garbage markers ("!!", "!ororv!", long runs of "!")
Also tails+greps the server log for ``allocator.cpp:105`` (the AUTHORITATIVE oracle).

LIGHT mode drives the §8 recompaction CONTROL scenario; FULL drives the §11 milestone.

tt-metal commit pinned: c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195
"""
import argparse
import base64
import json
import mimetypes
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DEFAULT_PROMPT = "Extract all the text content from this image, preserving the reading order."
GARBAGE_PATTERNS = [
    re.compile(r"!!+"),  # the observed "!!" corruption
    re.compile(r"!ororv!"),
    re.compile(r"(?:!\s*){6,}"),  # long runs of bangs
]


def to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def detect_garbage(text: str):
    hits = []
    for pat in GARBAGE_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


def ocr_one(base_url, api_key, model, prompt, max_tokens, path, timeout):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": to_data_uri(path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": round(elapsed, 3),
    }


def run_batch(jobs, base_url, api_key, model, prompt, max_tokens, timeout, concurrency):
    """jobs: list of (job_id, Path). Returns dict job_id -> result."""
    out = {}

    def _one(job):
        jid, path = job
        try:
            r = ocr_one(base_url, api_key, model, prompt, max_tokens, path, timeout)
            r["ok"] = True
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": str(e)}
        out[jid] = r

    if concurrency == 1:
        for j in jobs:
            _one(j)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(_one, j) for j in jobs]
            for f in as_completed(futs):
                f.result()
    return out


def grep_log(log_path, patterns):
    counts = {}
    if not log_path or not Path(log_path).exists():
        return {p: None for p in patterns}
    txt = Path(log_path).read_text(errors="replace")
    for p in patterns:
        counts[p] = len(re.findall(re.escape(p), txt))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8005/v1")
    ap.add_argument("--model", default="rednote-hilab/dots.ocr")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--image-dir", default="/home/aroberge/t1_pages")
    ap.add_argument("--pages", default="1,2", help="comma-separated page numbers")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--mode", choices=["light", "full", "recompaction"], default="light")
    ap.add_argument("--out", default="/home/aroberge/results/validate_run")
    ap.add_argument("--golden-dir", default=None, help="dir with pageN.txt goldens to byte-compare against")
    ap.add_argument("--log", default=None, help="server log to grep for allocator.cpp:105")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    pages = [int(x) for x in args.pages.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    page_path = {p: image_dir / f"t1_page{p}.png" for p in pages}
    for p, pp in page_path.items():
        if not pp.exists():
            print(f"MISSING image {pp}", file=sys.stderr)
            sys.exit(2)

    # Build the job set per repeat.
    # recompaction: asymmetric max_tokens forces vLLM batch recompaction mid-stream.
    per_page_text = {p: [] for p in pages}  # list over repeats
    all_results = []
    for rep in range(args.repeats):
        if args.mode == "recompaction":
            # page1 short (finishes early), others longer -> recompaction.
            jobs = []
            mt = {}
            for p in pages:
                jobs.append((p, page_path[p]))
                mt[p] = 64 if p == pages[0] else args.max_tokens
            # custom per-job max tokens: run sequentially-issued but concurrent batch
            out_map = {}

            def _one(job):
                jid, path = job
                try:
                    r = ocr_one(args.base_url, args.api_key, args.model, args.prompt, mt[jid], path, args.timeout)
                    r["ok"] = True
                except Exception as e:  # noqa: BLE001
                    r = {"ok": False, "error": str(e)}
                out_map[jid] = r

            with ThreadPoolExecutor(max_workers=max(args.concurrency, len(jobs))) as pool:
                futs = [pool.submit(_one, j) for j in jobs]
                for f in as_completed(futs):
                    f.result()
            batch = out_map
        else:
            jobs = [(p, page_path[p]) for p in pages]
            batch = run_batch(
                jobs,
                args.base_url,
                args.api_key,
                args.model,
                args.prompt,
                args.max_tokens,
                args.timeout,
                args.concurrency,
            )
        for p in pages:
            r = batch.get(p, {"ok": False, "error": "missing"})
            txt = r.get("text", "") if r.get("ok") else ""
            per_page_text[p].append(txt)
            all_results.append({"page": p, "repeat": rep, **r})
        print(f"[{args.label}] repeat {rep+1}/{args.repeats} done " f"(conc={args.concurrency}, pages={pages})")

    # Analysis
    report = {
        "label": args.label,
        "mode": args.mode,
        "pages": pages,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "per_page": {},
    }
    overall_ok = True
    for p in pages:
        texts = per_page_text[p]
        ok_texts = [t for t in texts if t != ""]
        determ = len(set(texts)) == 1 and len(texts) > 0
        garbage = {}
        for i, t in enumerate(texts):
            g = detect_garbage(t)
            if g:
                garbage[i] = g
        golden_match = None
        if args.golden_dir:
            gp = Path(args.golden_dir) / f"page{p}.txt"
            if gp.exists():
                golden = gp.read_text()
                golden_match = all(t == golden for t in texts if t)
        # write per-page artifact (first repeat)
        if texts and texts[0]:
            (out / f"page{p}.txt").write_text(texts[0], encoding="utf-8")
        empties = sum(1 for t in texts if t == "")
        rp = {
            "deterministic": determ,
            "unique_outputs": len(set(texts)),
            "garbage": garbage,
            "golden_match": golden_match,
            "empty_responses": empties,
            "lengths": [len(t) for t in texts],
            "preview": (ok_texts[0][:120] if ok_texts else ""),
        }
        report["per_page"][str(p)] = rp
        if not determ or garbage or empties:
            overall_ok = False
        print(
            f"  page{p}: determ={determ} uniq={len(set(texts))} "
            f"garbage={'YES' if garbage else 'no'} empty={empties} "
            f"lens={rp['lengths']} golden={golden_match}"
        )

    if args.log:
        log_counts = grep_log(args.log, ["allocator.cpp:105", "active trace"])
        report["log_counts"] = log_counts
        print(
            f"  LOG: allocator.cpp:105={log_counts.get('allocator.cpp:105')} "
            f"active-trace={log_counts.get('active trace')}"
        )

    report["overall_determinism_garbage_ok"] = overall_ok
    (out / f"{args.label}_report.json").write_text(json.dumps(report, indent=2))
    json.dump({"meta": report, "results": all_results}, open(out / f"{args.label}_full.json", "w"), indent=2)
    print(f"\nReport: {out / (args.label + '_report.json')}")
    print(f"OVERALL determinism+garbage OK: {overall_ok}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
