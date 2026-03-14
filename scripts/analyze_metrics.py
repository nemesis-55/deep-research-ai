#!/usr/bin/env python3
"""
Metrics log analyser — Deep Research AI

Reads the JSONL metrics logs produced by MetricsLogger and prints
a human-readable summary.

Usage:
    python scripts/analyze_metrics.py                        # today's log
    python scripts/analyze_metrics.py --date 2026-03-14      # specific day
    python scripts/analyze_metrics.py --all                  # all log files
    python scripts/analyze_metrics.py --type inference       # filter type
    python scripts/analyze_metrics.py --tail 20              # last N records
    python scripts/analyze_metrics.py --bench-interactions   # benchmark interaction log
    python scripts/analyze_metrics.py --bench-interactions --date 2026-03-15

Record types in the runtime log:
    hw        — hardware snapshot (every 30 s)
    inference — one entry per model generation
    startup / shutdown — session events

Record types in the benchmark interaction log:
    session_start / session_end — benchmark session bookends
    Every other record has: section, op, elapsed_s, ok, detail, mem_rss_mb, mem_free_gb
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

ROOT    = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs" / "metrics"


# ── Formatting helpers ────────────────────────────────────────────────────────

def _gb(b) -> str:
    return f"{b / 1e9:.2f} GB" if b is not None else "—"

def _pct(v) -> str:
    return f"{v:.1f}%" if v is not None else "—"

def _ts(iso) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%H:%M:%S")
    except Exception:
        return iso or "—"

SEP = "─" * 72


# ── Load records ──────────────────────────────────────────────────────────────

def _load_file(path: Path, type_filter: str = "") -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if type_filter and obj.get("type") != type_filter:
                    continue
                records.append(obj)
            except json.JSONDecodeError:
                pass
    return records


# ── Printers ─────────────────────────────────────────────────────────────────

def print_hw_summary(records: List[Dict]) -> None:
    hw = [r for r in records if r.get("type") == "hw"]
    if not hw:
        print("  No hardware snapshots found.")
        return

    cpu_vals  = [r["cpu_pct"]       for r in hw if r.get("cpu_pct")       is not None]
    ram_vals  = [r["ram_pct"]       for r in hw if r.get("ram_pct")       is not None]
    gpu_vals  = [r["gpu_pct"]       for r in hw if r.get("gpu_pct")       is not None]
    vram_vals = [r["vram_used_gb"]  for r in hw if r.get("vram_used_gb")  is not None]
    rss_vals  = [r["proc_rss_gb"]   for r in hw if r.get("proc_rss_gb")   is not None]

    def _stat(vals, fmt=".1f"):
        if not vals:
            return "—"
        return (f"avg {statistics.mean(vals):{fmt}}  "
                f"max {max(vals):{fmt}}  "
                f"min {min(vals):{fmt}}")

    print(f"  Samples    : {len(hw)}")
    print(f"  CPU  %     : {_stat(cpu_vals)}")
    print(f"  RAM  %     : {_stat(ram_vals)}")
    print(f"  GPU  %     : {_stat(gpu_vals)}")
    print(f"  VRAM used  : {_stat(vram_vals, '.2f')} GB")
    print(f"  Server RSS : {_stat(rss_vals,  '.2f')} GB")

    # Session totals from last hw record
    last = hw[-1]
    print(f"\n  Session requests : {last.get('session_requests', '—')}")
    print(f"  Session tokens   : {last.get('session_tokens', '—')}")
    uptime = last.get("session_uptime_s")
    if uptime:
        h, rem = divmod(int(uptime), 3600)
        m, s   = divmod(rem, 60)
        print(f"  Uptime           : {h:02d}:{m:02d}:{s:02d}")


def print_inference_summary(records: List[Dict]) -> None:
    inf = [r for r in records if r.get("type") == "inference"]
    if not inf:
        print("  No inference records found.")
        return

    elapsed_vals = [r["elapsed_s"]   for r in inf if r.get("elapsed_s")   is not None]
    tok_s_vals   = [r["tok_per_s"]   for r in inf if r.get("tok_per_s")   is not None]
    tok_vals     = [r["output_tokens"] for r in inf if r.get("output_tokens") is not None]

    print(f"  Total generations : {len(inf)}")
    if elapsed_vals:
        print(f"  Latency (s)       : "
              f"avg {statistics.mean(elapsed_vals):.2f}  "
              f"max {max(elapsed_vals):.2f}  "
              f"min {min(elapsed_vals):.2f}  "
              f"p50 {statistics.median(elapsed_vals):.2f}")
    if tok_s_vals:
        print(f"  Throughput (tok/s): "
              f"avg {statistics.mean(tok_s_vals):.1f}  "
              f"max {max(tok_s_vals):.1f}  "
              f"min {min(tok_s_vals):.1f}")
    if tok_vals:
        print(f"  Tokens/generation : "
              f"avg {statistics.mean(tok_vals):.0f}  "
              f"max {max(tok_vals)}  "
              f"min {min(tok_vals)}")

    # Breakdown by complexity
    complexities = {}
    for r in inf:
        c = r.get("complexity") or "unknown"
        complexities.setdefault(c, []).append(r)
    if len(complexities) > 1:
        print("\n  By complexity:")
        for c, items in sorted(complexities.items()):
            els = [x["elapsed_s"] for x in items if x.get("elapsed_s") is not None]
            tks = [x["tok_per_s"] for x in items if x.get("tok_per_s") is not None]
            print(f"    {c:<15} n={len(items):<4} "
                  f"avg {statistics.mean(els):.2f}s  "
                  f"{statistics.mean(tks):.1f} tok/s" if els else f"    {c}")

    # Breakdown by role
    roles = {}
    for r in inf:
        roles.setdefault(r.get("role", "?"), []).append(r)
    if len(roles) > 1:
        print("\n  By role:")
        for role, items in sorted(roles.items()):
            els = [x["elapsed_s"] for x in items if x.get("elapsed_s") is not None]
            print(f"    {role:<12} n={len(items):<4} "
                  f"avg {statistics.mean(els):.2f}s" if els else f"    {role}")


def print_raw(records: List[Dict], n: int) -> None:
    for r in records[-n:]:
        ts  = _ts(r.get("ts", ""))
        typ = r.get("type", "?")
        if typ == "hw":
            print(f"  {ts}  hw    "
                  f"cpu={_pct(r.get('cpu_pct'))}  "
                  f"ram={_pct(r.get('ram_pct'))}  "
                  f"gpu={_pct(r.get('gpu_pct'))}  "
                  f"vram={r.get('vram_used_gb','—'):.2f}GB  "
                  f"rss={r.get('proc_rss_gb','—')}")
        elif typ == "inference":
            print(f"  {ts}  infer "
                  f"role={r.get('role','?'):<8} "
                  f"cplx={r.get('complexity','?'):<15} "
                  f"{r.get('output_tokens','?'):>5}tok  "
                  f"{r.get('elapsed_s','?'):.2f}s  "
                  f"{r.get('tok_per_s','?'):.1f}tok/s")
        else:
            print(f"  {ts}  {typ:<8} {json.dumps({k:v for k,v in r.items() if k not in ('ts','type')})}")


# ── Benchmark interaction log printer ─────────────────────────────────────────

def print_bench_interactions(records: List[Dict]) -> None:
    """Print a detailed table of every timed interaction from a benchmark run."""
    # Exclude session bookend records from the table
    interactions = [r for r in records if r.get("section") not in (None,) and
                    r.get("op") not in ("benchmark_start", "benchmark_end")]

    if not interactions:
        print("  No interaction records found.")
        return

    # ── Per-section summary ───────────────────────────────────────────────────
    sections: Dict[str, list] = {}
    for r in interactions:
        sec = r.get("section", "?")
        sections.setdefault(sec, []).append(r)

    SECTION_ORDER = ["search", "scrape", "embed", "llm", "evidence", "memory"]
    ordered_sections = [s for s in SECTION_ORDER if s in sections]
    ordered_sections += [s for s in sections if s not in SECTION_ORDER]

    total_elapsed = sum(r.get("elapsed_s", 0) for r in interactions)
    ok_count   = sum(1 for r in interactions if r.get("ok", True))
    fail_count = len(interactions) - ok_count

    print(f"\n  Total interactions : {len(interactions)}")
    print(f"  Succeeded          : {ok_count}")
    print(f"  Failed             : {fail_count}")
    print(f"  Total elapsed      : {total_elapsed:.2f}s")

    for sec in ordered_sections:
        recs = sections[sec]
        elapsed_vals = [r["elapsed_s"] for r in recs if r.get("elapsed_s") is not None]
        ok_recs   = [r for r in recs if r.get("ok", True)]
        fail_recs = [r for r in recs if not r.get("ok", True)]
        rss_vals  = [r["mem_rss_mb"] for r in recs if r.get("mem_rss_mb") is not None]

        print(f"\n  ── {sec.upper()} ({'─' * (60 - len(sec))})")
        print(f"  {'Op':<30} {'Elapsed':>9}  {'OK':<5}  {'RSS MB':>8}  Detail")
        print(f"  {'─'*30} {'─'*9}  {'─'*5}  {'─'*8}  {'─'*20}")

        for r in recs:
            op       = r.get("op", "?")
            elapsed  = r.get("elapsed_s", 0)
            ok       = r.get("ok", True)
            rss      = r.get("mem_rss_mb")
            ts_str   = _ts(r.get("ts", ""))
            detail   = r.get("detail", {})

            # Build a compact detail string from the most informative fields
            skip_keys = {"elapsed_s", "ok", "error"}
            detail_str = "  ".join(
                f"{k}={v}" for k, v in detail.items()
                if k not in skip_keys and v not in (None, "", {})
            )[:60]
            if not ok:
                err = detail.get("error", "")
                detail_str = f"ERROR: {err}"[:60]

            rss_s = f"{rss:.0f}" if rss is not None else "—"
            ok_s  = "✓" if ok else "✗"
            print(f"  {op:<30} {elapsed:>8.3f}s  {ok_s:<5}  {rss_s:>8}  {detail_str}")

        # Section stats
        if elapsed_vals:
            avg = statistics.mean(elapsed_vals)
            mx  = max(elapsed_vals)
            mn  = min(elapsed_vals)
            tot = sum(elapsed_vals)
            print(f"\n  → {sec}: {len(recs)} calls | "
                  f"total={tot:.3f}s  avg={avg:.3f}s  max={mx:.3f}s  min={mn:.3f}s  "
                  f"errors={len(fail_recs)}")
            if rss_vals:
                print(f"     RSS  avg={statistics.mean(rss_vals):.0f} MB  "
                      f"max={max(rss_vals):.0f} MB  "
                      f"min={min(rss_vals):.0f} MB")

    # ── Slowest interactions ──────────────────────────────────────────────────
    slowest = sorted(interactions, key=lambda r: r.get("elapsed_s", 0), reverse=True)[:5]
    print(f"\n  ── TOP 5 SLOWEST INTERACTIONS {'─'*40}")
    for i, r in enumerate(slowest, 1):
        print(f"  {i}. [{r.get('section','?')}] {r.get('op','?'):<30}  "
              f"{r.get('elapsed_s',0):.3f}s  ok={r.get('ok',True)}")

    # ── Failed interactions ───────────────────────────────────────────────────
    failed = [r for r in interactions if not r.get("ok", True)]
    if failed:
        print(f"\n  ── FAILURES ({len(failed)}) {'─'*50}")
        for r in failed:
            err = r.get("detail", {}).get("error", "?")
            print(f"  [{r.get('section','?')}] {r.get('op','?'):<30}  "
                  f"{r.get('elapsed_s',0):.3f}s  {err[:60]}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deep Research AI — metrics log analyser")
    ap.add_argument("--date",  default="",      help="YYYY-MM-DD (default: today)")
    ap.add_argument("--all",   action="store_true", help="Load all log files")
    ap.add_argument("--type",  default="",      help="Record type filter: hw|inference|startup|shutdown")
    ap.add_argument("--tail",  type=int, default=0,  help="Print last N raw records")
    ap.add_argument("--raw",   action="store_true",  help="Print raw records (no summary)")
    ap.add_argument(
        "--bench-interactions", action="store_true",
        help="Analyse benchmark interaction log(s) (benchmark_*_interactions.jsonl)",
    )
    args = ap.parse_args()

    # ── Benchmark interaction mode ────────────────────────────────────────────
    if args.bench_interactions:
        if args.all:
            files = sorted(LOG_DIR.glob("benchmark_*_interactions.jsonl"))
        elif args.date:
            files = sorted(LOG_DIR.glob(f"benchmark_{args.date}*_interactions.jsonl"))
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            files = sorted(LOG_DIR.glob(f"benchmark_{today}*_interactions.jsonl"))

        if not files:
            print(f"No benchmark interaction logs found in {LOG_DIR}")
            print("  Tip: run  python scripts/benchmark.py  first.")
            return

        all_records: List[Dict] = []
        for f in files:
            recs = _load_file(f)
            all_records.extend(recs)

        print(f"\n{SEP}")
        print(f"  Deep Research AI — Benchmark Interaction Analysis")
        print(f"  Log dir  : {LOG_DIR}")
        print(f"  Files    : {', '.join(f.name for f in files)}")
        print(f"  Records  : {len(all_records)}")
        print(SEP)

        print_bench_interactions(all_records)
        print(f"\n{SEP}\n")
        return

    # ── Runtime metrics mode (original behaviour) ─────────────────────────────
    if args.all:
        files = sorted(LOG_DIR.glob("metrics_*.jsonl"))
    elif args.date:
        files = [LOG_DIR / f"metrics_{args.date}.jsonl"]
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        files = [LOG_DIR / f"metrics_{today}.jsonl"]

    if not files:
        print(f"No log files found in {LOG_DIR}")
        return

    all_records_rt: List[Dict] = []
    for f in files:
        recs = _load_file(f, type_filter=args.type)
        all_records_rt.extend(recs)
        if not recs:
            print(f"  (empty or missing: {f.name})")

    print(f"\n{SEP}")
    print(f"  Deep Research AI — Metrics Analysis")
    print(f"  Log dir  : {LOG_DIR}")
    print(f"  Files    : {len(files)}  |  Records: {len(all_records_rt)}")
    print(SEP)

    if args.tail or args.raw:
        n = args.tail if args.tail else len(all_records_rt)
        print(f"\n  Last {n} records:\n")
        print_raw(all_records_rt, n)
        return

    if not args.type or args.type == "hw":
        print("\n  ── Hardware Snapshots ─────────────────────────────────────────")
        print_hw_summary(all_records_rt)

    if not args.type or args.type == "inference":
        print("\n  ── Inference Performance ──────────────────────────────────────")
        print_inference_summary(all_records_rt)

    # Session events
    events = [r for r in all_records_rt if r.get("type") in ("startup", "shutdown")]
    if events:
        print("\n  ── Session Events ─────────────────────────────────────────────")
        for e in events:
            print(f"    {_ts(e.get('ts',''))}  {e.get('type')}  "
                  + ("  ".join(f"{k}={v}" for k,v in e.items()
                                if k not in ("ts", "type"))))

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
