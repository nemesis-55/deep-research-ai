#!/usr/bin/env python3
"""
Metrics log analyser — Deep Research AI

Reads the JSONL metrics logs produced by MetricsLogger and prints
a human-readable summary.

Usage:
    python scripts/analyze_metrics.py                   # today's log
    python scripts/analyze_metrics.py --date 2026-03-14 # specific day
    python scripts/analyze_metrics.py --all             # all log files
    python scripts/analyze_metrics.py --type inference  # filter type
    python scripts/analyze_metrics.py --tail 20         # last N records

Record types in the log:
    hw        — hardware snapshot (every 30 s)
    inference — one entry per model generation
    startup / shutdown — session events
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deep Research AI — metrics log analyser")
    ap.add_argument("--date",  default="",      help="YYYY-MM-DD (default: today)")
    ap.add_argument("--all",   action="store_true", help="Load all log files")
    ap.add_argument("--type",  default="",      help="Record type filter: hw|inference|startup|shutdown")
    ap.add_argument("--tail",  type=int, default=0,  help="Print last N raw records")
    ap.add_argument("--raw",   action="store_true",  help="Print raw records (no summary)")
    args = ap.parse_args()

    # Collect files
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

    all_records: List[Dict] = []
    for f in files:
        recs = _load_file(f, type_filter=args.type)
        all_records.extend(recs)
        if not recs:
            print(f"  (empty or missing: {f.name})")

    print(f"\n{SEP}")
    print(f"  Deep Research AI — Metrics Analysis")
    print(f"  Log dir  : {LOG_DIR}")
    print(f"  Files    : {len(files)}  |  Records: {len(all_records)}")
    print(SEP)

    if args.tail or args.raw:
        n = args.tail if args.tail else len(all_records)
        print(f"\n  Last {n} records:\n")
        print_raw(all_records, n)
        return

    if not args.type or args.type == "hw":
        print("\n  ── Hardware Snapshots ─────────────────────────────────────────")
        print_hw_summary(all_records)

    if not args.type or args.type == "inference":
        print("\n  ── Inference Performance ──────────────────────────────────────")
        print_inference_summary(all_records)

    # Session events
    events = [r for r in all_records if r.get("type") in ("startup", "shutdown")]
    if events:
        print("\n  ── Session Events ─────────────────────────────────────────────")
        for e in events:
            print(f"    {_ts(e.get('ts',''))}  {e.get('type')}  "
                  + ("  ".join(f"{k}={v}" for k,v in e.items()
                                if k not in ("ts", "type"))))

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
