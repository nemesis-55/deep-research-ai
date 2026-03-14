#!/usr/bin/env python3
"""
benchmark.py — Deep Research AI Pipeline Benchmarker
=====================================================
Measures throughput at every layer of the pipeline and prints
recommended config values for this specific Mac Mini M4 16 GB.

What it measures
────────────────
1. DDG Search throughput        — 1 / 2 / 3 queries in parallel
2. HTTP Scraper throughput       — 1 / 2 / 3 / 4 concurrent workers
3. Embedding (bge-small)        — tokens/s for various text sizes
4. LLM throughput (mlx or llama_cpp) — tok/s at 128 / 256 / 512 / 1024 max_new_tokens
5. End-to-end evidence builder  — chunk + rank latency vs corpus size
6. Memory usage at each stage   — RSS + system free via psutil

Usage
─────
  # Full benchmark (takes ~10-15 minutes with model loaded):
  python scripts/benchmark.py

  # Skip LLM benchmark (fast — only I/O layers):
  python scripts/benchmark.py --no-llm

  # Skip web I/O (offline mode — only LLM + embed):
  python scripts/benchmark.py --offline

  # Custom output file:
  python scripts/benchmark.py --out results/bench_2026.json

Output
──────
  • Live table printed to stdout
  • JSON report written to logs/metrics/benchmark_<date>.json
  • Final "RECOMMENDED CONFIG" block printed to stdout

Design
──────
  All timings use time.monotonic() for monotonic accuracy.
  LLM calls are always sequential — protected by _LLM_SEMAPHORE in model_manager.
  I/O benchmarks use ThreadPoolExecutor mirroring production code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Bootstrap: add project root to sys.path ───────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import logging
logging.basicConfig(
    level=logging.WARNING,   # suppress info noise during benchmark
    format="%(levelname)s %(name)s — %(message)s",
)
# Keep our own benchmark logger at INFO
_log = logging.getLogger("benchmark")
_log.setLevel(logging.INFO)
_log.propagate = True

# Suppress BertModel LOAD REPORT spam from sentence-transformers / transformers
for _noisy_lib in (
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "sentence_transformers.models",
    "sentence_transformers.models.Transformer",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
):
    logging.getLogger(_noisy_lib).setLevel(logging.ERROR)

# ── Optional psutil for memory ─────────────────────────────────────────────────
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None           # type: ignore
    _HAS_PSUTIL = False

# ── Colour helpers (ANSI) ──────────────────────────────────────────────────────
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_WARN  = "\033[93m"
_RED   = "\033[91m"
_RESET = "\033[0m"


def _h(text: str) -> str:
    return f"{_BOLD}{_CYAN}{text}{_RESET}"

def _ok(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"

def _warn(text: str) -> str:
    return f"{_WARN}{text}{_RESET}"

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


# ── Memory helpers ─────────────────────────────────────────────────────────────

def _mem_mb() -> Dict[str, float]:
    """Return current process RSS and system free memory in MB."""
    result: Dict[str, float] = {"rss_mb": 0.0, "free_gb": 0.0}
    if not _HAS_PSUTIL:
        return result
    proc = _psutil.Process(os.getpid())
    result["rss_mb"] = proc.memory_info().rss / 1024 / 1024
    vm = _psutil.virtual_memory()
    result["free_gb"] = vm.available / 1024 / 1024 / 1024
    return result


# ── Timing context manager ─────────────────────────────────────────────────────

class _Timer:
    """Simple monotonic timer used as a context manager."""
    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.monotonic() - self._t0


# ── Results accumulator ────────────────────────────────────────────────────────

class BenchmarkResults:
    def __init__(self) -> None:
        self.search:    List[Dict] = []
        self.scrape:    List[Dict] = []
        self.embed:     List[Dict] = []
        self.llm:       List[Dict] = []
        self.evidence:  List[Dict] = []
        self.memory:    List[Dict] = []
        self.meta: Dict = {
            "date":        str(date.today()),
            "platform":    sys.platform,
            "python":      sys.version.split()[0],
        }
        try:
            import platform
            self.meta["cpu"] = platform.processor() or platform.machine()
        except Exception:
            pass
        if _HAS_PSUTIL:
            vm = _psutil.virtual_memory()
            self.meta["total_ram_gb"] = round(vm.total / 1024**3, 1)

    def to_dict(self) -> Dict:
        return {
            "meta":     self.meta,
            "search":   self.search,
            "scrape":   self.scrape,
            "embed":    self.embed,
            "llm":      self.llm,
            "evidence": self.evidence,
            "memory":   self.memory,
        }


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTION LOGGER — records every timed call as a JSONL line
# ══════════════════════════════════════════════════════════════════════════════

class InteractionLogger:
    """
    Appends one JSONL record per timed benchmark interaction to a sidecar file.

    File: logs/metrics/benchmark_interactions_<date>.jsonl

    Each record shape:
        {
          "ts":          "<ISO-8601 UTC>",
          "section":     "search" | "scrape" | "embed" | "llm" | "evidence" | "memory",
          "op":          "<short description of the operation>",
          "elapsed_s":   <float>,
          "ok":          <bool>,
          "detail":      { … section-specific fields … },
          "mem_rss_mb":  <float | null>,
          "mem_free_gb": <float | null>,
        }
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write a session-start record
        self._append({
            "type":    "session_start",
            "ts":      datetime.now(timezone.utc).isoformat(),
            "section": "meta",
            "op":      "benchmark_start",
            "elapsed_s": 0.0,
            "ok":      True,
            "detail":  {"date": str(date.today())},
        })

    # ── Internal ──────────────────────────────────────────────────────────────

    def _append(self, record: Dict) -> None:
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    @staticmethod
    def _mem_snapshot() -> Dict[str, Optional[float]]:
        snap: Dict[str, Optional[float]] = {"mem_rss_mb": None, "mem_free_gb": None}
        if _HAS_PSUTIL:
            import psutil as _ps
            snap["mem_rss_mb"]  = round(_ps.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
            snap["mem_free_gb"] = round(_ps.virtual_memory().available / 1024 / 1024 / 1024, 3)
        return snap

    # ── Public log helpers ────────────────────────────────────────────────────

    def log(
        self,
        section:   str,
        op:        str,
        elapsed_s: float,
        ok:        bool = True,
        detail:    Optional[Dict] = None,
        capture_mem: bool = True,
    ) -> None:
        """Log one interaction."""
        record: Dict[str, Any] = {
            "section":   section,
            "op":        op,
            "elapsed_s": round(elapsed_s, 4),
            "ok":        ok,
            "detail":    detail or {},
        }
        if capture_mem:
            record.update(self._mem_snapshot())
        self._append(record)

    def log_error(self, section: str, op: str, elapsed_s: float, error: str) -> None:
        self.log(section, op, elapsed_s, ok=False, detail={"error": error})

    def finish(self, summary_path: Path) -> None:
        self._append({
            "type":      "session_end",
            "section":   "meta",
            "op":        "benchmark_end",
            "elapsed_s": 0.0,
            "ok":        True,
            "detail":    {"summary_json": str(summary_path)},
        })
        print(f"\n  {_ok(f'📋 Interaction log → {self.path}')}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DDG Search throughput
# ══════════════════════════════════════════════════════════════════════════════

_BENCH_QUERIES = [
    "Apple M4 chip benchmark performance",
    "MLX framework Apple Silicon 2025",
    "large language model inference throughput",
    "DuckDuckGo search API rate limits",
]

def bench_search(results: BenchmarkResults, offline: bool, ilog: "InteractionLogger") -> None:
    if offline:
        print(f"\n{_warn('⚡ [Search] Skipped (--offline mode)')}")
        ilog.log("search", "skipped_offline", 0.0, ok=True, detail={"reason": "offline"})
        return

    from backend.tools.web_search import search_web
    from backend.tools.web_search_engine import _search_one, _url_key

    print(f"\n{_h('━━━ 1. DDG SEARCH THROUGHPUT ━━━')}")
    print(f"{'Workers':<10} {'Queries':<8} {'Time (s)':<12} {'URLs/s':<10} {'Total URLs'}")
    print(_sep())

    for n_workers in [1, 2, 3]:
        queries = _BENCH_QUERIES[:n_workers + 1]   # 2, 3, 4 queries
        n = len(queries)
        urls_found = 0

        with _Timer() as t:
            if n_workers == 1:
                for i, q in enumerate(queries):
                    t0 = time.monotonic()
                    try:
                        _, raw = _search_one(q, 8, i)
                        n_urls = len(raw)
                    except Exception as e:
                        ilog.log_error("search", f"query_{i}", time.monotonic() - t0, str(e))
                        n_urls = 0
                        raw = []
                    urls_found += n_urls
                    ilog.log("search", "single_query", time.monotonic() - t0, ok=True, detail={
                        "query":   q,
                        "n_urls":  n_urls,
                        "workers": 1,
                        "query_i": i,
                    })
                    if i < n - 1:
                        time.sleep(1.0)   # rate limit
            else:
                query_times: Dict[str, float] = {}
                start_times: Dict[str, float] = {}
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    future_to_q = {
                        pool.submit(_search_one, q, 8, i): (q, i, time.monotonic())
                        for i, q in enumerate(queries)
                    }
                    for fut in as_completed(future_to_q):
                        q, qi, t0_q = future_to_q[fut]
                        elapsed_q = time.monotonic() - t0_q
                        try:
                            _, raw = fut.result()
                            n_urls = len(raw)
                            urls_found += n_urls
                            ilog.log("search", "parallel_query", elapsed_q, ok=True, detail={
                                "query":   q,
                                "n_urls":  n_urls,
                                "workers": n_workers,
                                "query_i": qi,
                            })
                        except Exception as e:
                            ilog.log_error("search", f"parallel_query_{qi}", elapsed_q, str(e))

        urls_per_s = urls_found / t.elapsed if t.elapsed > 0 else 0
        row = {
            "workers":   n_workers,
            "queries":   n,
            "elapsed_s": round(t.elapsed, 2),
            "urls_per_s": round(urls_per_s, 2),
            "total_urls": urls_found,
        }
        results.search.append(row)
        ilog.log("search", "worker_level", t.elapsed, ok=True, detail=row)
        print(f"{n_workers:<10} {n:<8} {t.elapsed:<12.2f} {urls_per_s:<10.1f} {urls_found}")
        time.sleep(2.0)   # cooldown between worker levels

    print()
    _best = max(results.search, key=lambda r: r["urls_per_s"])
    _bw, _bs = _best["workers"], _best["urls_per_s"]
    print(f"  → Best: {_ok(str(_bw))} workers = {_ok(f'{_bs:.1f}')} URLs/s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HTTP Scraper throughput
# ══════════════════════════════════════════════════════════════════════════════

_BENCH_URLS = [
    "https://en.wikipedia.org/wiki/Apple_M4",
    "https://arstechnica.com/gadgets/2024/05/apple-announces-m4-chip/",
    "https://www.nasa.gov/missions/",
    "https://spacenews.com/",
    "https://www.nature.com/articles/d41586-024-00364-4",
    "https://simonwillison.net/2024/Mar/8/gpt-4-barrier/",
    "https://news.ycombinator.com/",
]

def bench_scrape(results: BenchmarkResults, offline: bool, ilog: "InteractionLogger") -> None:
    if offline:
        print(f"\n{_warn('⚡ [Scrape] Skipped (--offline mode)')}")
        ilog.log("scrape", "skipped_offline", 0.0, ok=True, detail={"reason": "offline"})
        return

    from backend.tools.page_scraper import scrape_page

    print(f"\n{_h('━━━ 2. HTTP SCRAPER THROUGHPUT ━━━')}")
    print(f"{'Workers':<10} {'URLs':<8} {'Time (s)':<12} {'Pages/s':<10} {'Chars/page (avg)'}")
    print(_sep())

    def _do_scrape(url: str) -> Tuple[str, int]:
        try:
            page = scrape_page(url, follow_links=False, follow_links_depth=0, max_follow_links=0)
            return url, len((page.get("text") or "").strip())
        except Exception:
            return url, 0

    for n_workers in [1, 2, 3, 4]:
        urls = _BENCH_URLS[:n_workers + 1]   # 2, 3, 4, 5 URLs
        chars_total = 0
        pages_done  = 0

        with _Timer() as t:
            if n_workers == 1:
                for url in urls:
                    t0 = time.monotonic()
                    _, chars = _do_scrape(url)
                    elapsed_url = time.monotonic() - t0
                    ok = chars > 0
                    ilog.log("scrape", "single_url", elapsed_url, ok=ok, detail={
                        "url": url, "chars": chars, "workers": 1,
                    })
                    if ok:
                        chars_total += chars
                        pages_done  += 1
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    future_to_url = {pool.submit(_do_scrape, u): (u, time.monotonic()) for u in urls}
                    for fut in as_completed(future_to_url):
                        url, t0_u = future_to_url[fut]
                        elapsed_url = time.monotonic() - t0_u
                        try:
                            _, chars = fut.result()
                            ok = chars > 0
                            ilog.log("scrape", "parallel_url", elapsed_url, ok=ok, detail={
                                "url": url, "chars": chars, "workers": n_workers,
                            })
                            if ok:
                                chars_total += chars
                                pages_done  += 1
                        except Exception as e:
                            ilog.log_error("scrape", f"url_{url[:40]}", elapsed_url, str(e))

        pages_per_s  = pages_done / t.elapsed if t.elapsed > 0 else 0
        avg_chars    = chars_total // pages_done if pages_done else 0
        row = {
            "workers":    n_workers,
            "urls":       len(urls),
            "elapsed_s":  round(t.elapsed, 2),
            "pages_per_s": round(pages_per_s, 2),
            "avg_chars":  avg_chars,
        }
        results.scrape.append(row)
        ilog.log("scrape", "worker_level", t.elapsed, ok=True, detail=row)
        print(f"{n_workers:<10} {len(urls):<8} {t.elapsed:<12.2f} {pages_per_s:<10.2f} {avg_chars:,}")

    print()
    _best = max(results.scrape, key=lambda r: r["pages_per_s"])
    _bw, _bps = _best["workers"], _best["pages_per_s"]
    print(f"  → Best: {_ok(str(_bw))} workers = {_ok(f'{_bps:.2f}')} pages/s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Embedding throughput (bge-small-en-v1.5)
# ══════════════════════════════════════════════════════════════════════════════

_EMBED_SIZES = [128, 512, 2048, 8192]   # characters (approx tokens / 4)

def bench_embed(results: BenchmarkResults, ilog: "InteractionLogger") -> None:
    print(f"\n{_h('━━━ 3. EMBEDDING THROUGHPUT (bge-small-en-v1.5) ━━━')}")
    print(f"{'Input size':<14} {'Texts':<8} {'Time (s)':<12} {'Texts/s':<10} {'Chars/s'}")
    print(_sep())

    try:
        from sentence_transformers import SentenceTransformer
        t0_load = time.monotonic()
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        load_elapsed = time.monotonic() - t0_load
        ilog.log("embed", "model_load", load_elapsed, ok=True, detail={"model": "bge-small-en-v1.5"})
    except Exception as e:
        print(f"  {_warn(f'Embedding model unavailable: {e}')}")
        ilog.log_error("embed", "model_load", 0.0, str(e))
        return

    base_text = "The Apple M4 chip delivers exceptional performance for " * 20

    for size in _EMBED_SIZES:
        texts = [base_text[:size]] * 8   # batch of 8
        total_chars = sum(len(t) for t in texts)

        with _Timer() as t:
            model.encode(texts, show_progress_bar=False, batch_size=8)

        texts_per_s = len(texts) / t.elapsed if t.elapsed > 0 else 0
        chars_per_s = total_chars / t.elapsed if t.elapsed > 0 else 0
        row = {
            "input_chars": size,
            "batch_size":  len(texts),
            "elapsed_s":   round(t.elapsed, 3),
            "texts_per_s": round(texts_per_s, 2),
            "chars_per_s": round(chars_per_s, 0),
        }
        results.embed.append(row)
        ilog.log("embed", "encode_batch", t.elapsed, ok=True, detail=row)
        mem = _mem_mb()
        print(
            f"{size:<14} {len(texts):<8} {t.elapsed:<12.3f} "
            f"{texts_per_s:<10.2f} {chars_per_s:,.0f}"
            + (f"  (RSS {mem['rss_mb']:.0f} MB)" if _HAS_PSUTIL else "")
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LLM throughput
# ══════════════════════════════════════════════════════════════════════════════

_LLM_TOKEN_SIZES = [128, 256, 512, 1024]
_LLM_PROMPT = (
    "You are a research assistant. "
    "Write a detailed analysis of Apple Silicon M4 chip performance "
    "for machine learning workloads, covering: "
    "(1) raw compute specs, (2) memory bandwidth, (3) MLX framework support, "
    "(4) real-world inference benchmarks, (5) comparison with NVIDIA RTX 4090. "
    "Include specific numbers and technical details."
)

def bench_llm(results: BenchmarkResults, ilog: "InteractionLogger") -> None:
    print(f"\n{_h('━━━ 4. LLM THROUGHPUT ━━━')}")

    try:
        from backend.model_loader import generate_text
        from backend.model_manager import get_active_runtime, get_active_name, _lock, _active
    except Exception as e:
        print(f"  {_warn(f'model_loader unavailable: {e}')}")
        ilog.log_error("llm", "import", 0.0, str(e))
        return

    print(f"  Loading writer model…")
    t0_load = time.monotonic()
    try:
        from backend.model_manager import _load
        handle = _load("writer")
        runtime = handle.runtime
        name    = handle.name
        ilog.log("llm", "model_load", time.monotonic() - t0_load, ok=True, detail={
            "model": name, "runtime": runtime,
        })
    except Exception as e:
        ilog.log_error("llm", "model_load", time.monotonic() - t0_load, str(e))
        print(f"  {_warn(f'Model load failed: {e}')}")
        return

    print(f"  Model: {_ok(name)}  Runtime: {_ok(runtime)}")
    print()
    print(f"{'max_new_tokens':<18} {'Time (s)':<12} {'Approx tok/s':<16} {'Output words':<14} {'Memory (RSS MB)'}")
    print(_sep())

    for max_tok in _LLM_TOKEN_SIZES:
        mem_before = _mem_mb()
        with _Timer() as t:
            try:
                out = generate_text(
                    _LLM_PROMPT,
                    max_new_tokens=max_tok,
                    role="writer",
                    temperature=0.3,
                )
            except Exception as e:
                print(f"  {max_tok:<18} ERROR: {e}")
                ilog.log_error("llm", f"generate_{max_tok}tok", t.elapsed, str(e))
                continue
        mem_after = _mem_mb()

        word_count  = len(out.split())
        # Approximate: generated text ≈ tokens × 0.75 words; tok = words / 0.75
        approx_toks = max(1, int(word_count / 0.75))
        tok_per_s   = approx_toks / t.elapsed if t.elapsed > 0 else 0

        row = {
            "max_new_tokens": max_tok,
            "elapsed_s":      round(t.elapsed, 2),
            "approx_tok_s":   round(tok_per_s, 1),
            "word_count":     word_count,
            "rss_mb":         round(mem_after.get("rss_mb", 0), 0),
        }
        results.llm.append(row)
        ilog.log("llm", "generate", t.elapsed, ok=True, detail={
            **row,
            "model":        name,
            "runtime":      runtime,
            "rss_mb_before": round(mem_before.get("rss_mb", 0), 1),
            "rss_mb_after":  round(mem_after.get("rss_mb", 0), 1),
            "free_gb_after": round(mem_after.get("free_gb", 0), 3),
            "prompt_len":    len(_LLM_PROMPT),
        })
        print(
            f"{max_tok:<18} {t.elapsed:<12.2f} {tok_per_s:<16.1f} "
            f"{word_count:<14} {mem_after.get('rss_mb', 0):.0f}"
        )

    if results.llm:
        # Best tok/s is usually at smaller token counts (cache warm-up amortised)
        best = max(results.llm, key=lambda r: r["approx_tok_s"])
        _btok, _bts = best["max_new_tokens"], best["approx_tok_s"]
        print(f"\n  → Best: {_ok(str(_btok))} max_new_tokens = {_ok(f'{_bts:.1f}')} tok/s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Evidence builder (chunk + rank) latency
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_pages(n: int, chars_each: int = 8000) -> List[Dict]:
    """Generate synthetic page dicts for evidence builder benchmarking."""
    lorem = (
        "The quick brown fox jumps over the lazy dog. "
        "Apple M4 chip delivers best-in-class performance per watt. "
        "MLX framework enables efficient inference on Apple Silicon. "
        "Researchers have demonstrated 15 tok/s on Qwen2.5-7B 4-bit models. "
        "Memory bandwidth is the primary bottleneck for LLM inference. "
    ) * 40
    return [
        {
            "url":   f"https://example{i}.com/article",
            "title": f"Benchmark Page {i+1}",
            "text":  lorem[:chars_each],
        }
        for i in range(n)
    ]

def bench_evidence(results: BenchmarkResults, ilog: "InteractionLogger") -> None:
    print(f"\n{_h('━━━ 5. EVIDENCE BUILDER (chunk + rank) LATENCY ━━━')}")
    print(f"{'Pages':<8} {'Chars/page':<14} {'Chunks':<10} {'Time (s)':<12} {'Chars/s'}")
    print(_sep())

    try:
        from backend.tools.evidence_builder import build_evidence
    except Exception as e:
        print(f"  {_warn(f'evidence_builder unavailable: {e}')}")
        ilog.log_error("evidence", "import", 0.0, str(e))
        return

    configs = [
        (5,  4_000),
        (10, 8_000),
        (15, 8_000),
        (20, 12_000),
    ]

    query = "Apple M4 MLX inference throughput performance"

    for n_pages, chars_each in configs:
        pages = _make_fake_pages(n_pages, chars_each)
        total_chars = n_pages * chars_each

        with _Timer() as t:
            try:
                ev = build_evidence(query, pages, top_k=20, max_chars=20_000)
                chunk_count = len(ev.sources)
                ok = True
                err_msg = ""
            except Exception as e:
                print(f"  {n_pages:<8} ERROR: {e}")
                ilog.log_error("evidence", f"build_{n_pages}pages", t.elapsed, str(e))
                continue

        chars_per_s = total_chars / t.elapsed if t.elapsed > 0 else 0
        row = {
            "pages":       n_pages,
            "chars_each":  chars_each,
            "total_chars": total_chars,
            "elapsed_s":   round(t.elapsed, 3),
            "chars_per_s": round(chars_per_s, 0),
            "sources":     chunk_count,
        }
        results.evidence.append(row)
        ilog.log("evidence", "build", t.elapsed, ok=True, detail=row)
        print(
            f"{n_pages:<8} {chars_each:<14,} {chunk_count:<10} "
            f"{t.elapsed:<12.3f} {chars_per_s:,.0f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Memory snapshot
# ══════════════════════════════════════════════════════════════════════════════

def bench_memory(results: BenchmarkResults, ilog: "InteractionLogger") -> None:
    print(f"\n{_h('━━━ 6. MEMORY SNAPSHOT ━━━')}")
    if not _HAS_PSUTIL:
        print(f"  {_warn('psutil not installed — install it for memory readings:')}")
        print("    pip install psutil")
        ilog.log("memory", "snapshot", 0.0, ok=False, detail={"reason": "psutil_missing"})
        return

    with _Timer() as t:
        proc = _psutil.Process(os.getpid())
        vm   = _psutil.virtual_memory()
        swap = _psutil.swap_memory()
        mem  = proc.memory_info()

    row = {
        "rss_mb":       round(mem.rss       / 1024**2, 1),
        "vms_mb":       round(mem.vms       / 1024**2, 1),
        "sys_used_gb":  round(vm.used       / 1024**3, 2),
        "sys_free_gb":  round(vm.available  / 1024**3, 2),
        "sys_total_gb": round(vm.total      / 1024**3, 2),
        "percent":      vm.percent,
        "swap_used_mb": round(swap.used     / 1024**2, 1),
        "swap_total_mb":round(swap.total    / 1024**2, 1),
        "swap_pct":     swap.percent,
    }
    results.memory.append(row)
    ilog.log("memory", "snapshot", t.elapsed, ok=True, detail=row, capture_mem=False)

    rss_s  = f'{row["rss_mb"]:.1f} MB'
    free_s = f'{row["sys_free_gb"]:.2f} GB'
    print(f"  Process RSS     : {_ok(rss_s)}")
    print(f"  Process VMS     : {row['vms_mb']:.1f} MB")
    print(f"  System used     : {row['sys_used_gb']:.2f} GB / {row['sys_total_gb']:.1f} GB  ({row['percent']}%)")
    print(f"  System free     : {_ok(free_s)}")
    print(f"  Swap used       : {row['swap_used_mb']:.1f} MB / {row['swap_total_mb']:.1f} MB  ({row['swap_pct']}%)")

    # Headroom estimate for an additional model load (~4–5 GB)
    headroom = row["sys_free_gb"]
    if headroom >= 5.0:
        print(f"  Headroom        : {_ok(f'{headroom:.2f} GB free')} — ✅ safe to load second model")
    elif headroom >= 3.0:
        print(f"  Headroom        : {_warn(f'{headroom:.2f} GB free')} — ⚠ tight margin for second model")
    else:
        print(f"  Headroom        : {_warn(f'{headroom:.2f} GB free')} — ❌ insufficient for model swap")


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _print_recommendations(results: BenchmarkResults) -> Dict[str, Any]:
    print(f"\n{_h('━━━ RECOMMENDED CONFIG VALUES ━━━')}")
    print("  (update config/config.yaml with these values)\n")

    rec: Dict[str, Any] = {}

    # ── Search workers ─────────────────────────────────────────────────────
    if results.search:
        best_s = max(results.search, key=lambda r: r["urls_per_s"])
        w = best_s["workers"]
        # Safety: DDG rate-limits at >3 parallel queries, cap recommendation
        w = min(w, 2)   # conservative — production traffic adds jitter
        rec["parallel_search_workers"] = w
        print(f"  parallel_search_workers: {_ok(str(w))}")
        print(f"    (measured {best_s['urls_per_s']:.1f} URLs/s at {best_s['workers']} workers)")
    else:
        rec["parallel_search_workers"] = 2
        print(f"  parallel_search_workers: {_warn('2')} (default — search not benchmarked)")

    # ── Scrape workers ─────────────────────────────────────────────────────
    if results.scrape:
        best_sc = max(results.scrape, key=lambda r: r["pages_per_s"])
        w = min(best_sc["workers"], 4)
        rec["parallel_scrape_workers"] = w
        print(f"  parallel_scrape_workers: {_ok(str(w))}")
        print(f"    (measured {best_sc['pages_per_s']:.2f} pages/s at {best_sc['workers']} workers)")
    else:
        rec["parallel_scrape_workers"] = 3
        print(f"  parallel_scrape_workers: {_warn('3')} (default — scraper not benchmarked)")

    # ── LLM config ────────────────────────────────────────────────────────
    if results.llm:
        rows = results.llm
        # tok/s should be ≥ 12 tok/s for any size — flag if it drops
        fast_rows = [r for r in rows if r["approx_tok_s"] >= 12]
        if fast_rows:
            max_safe_tok = max(r["max_new_tokens"] for r in fast_rows)
        else:
            max_safe_tok = rows[0]["max_new_tokens"]

        # Typical observed: writer model used for research sections (1000 tok budget)
        avg_tok_s = sum(r["approx_tok_s"] for r in rows) / len(rows)
        print(f"\n  models.writer.max_new_tokens: {_ok(str(max_safe_tok))}")
        print(f"    (avg {avg_tok_s:.1f} tok/s — "
              f"1000 tok ≈ {1000/avg_tok_s:.0f}s generation time)")

        rec["avg_tok_s"] = round(avg_tok_s, 1)
        rec["max_new_tokens_writer"] = max_safe_tok

    # ── Memory ────────────────────────────────────────────────────────────
    if results.memory:
        m = results.memory[-1]
        free = m.get("sys_free_gb", 0)
        print(f"\n  System free RAM : {free:.2f} GB")
        if free >= 5.0:
            print(f"  {_ok('→ Comfortable — current config is safe.')}")
        elif free >= 3.0:
            print(f"  {_warn('→ Moderate — avoid deep_crawl.max_articles > 20.')}")
        else:
            print(f"  {_warn('→ Low — consider reducing models.writer.context_length to 8192.')}")

        rec["sys_free_gb"] = round(free, 2)

    # ── Evidence builder ──────────────────────────────────────────────────
    if results.evidence:
        fastest = min(results.evidence, key=lambda r: r["elapsed_s"])
        print(f"\n  deep_crawl.max_articles: {_ok(str(fastest['pages'] + 5))}")
        print(f"    (evidence builder handles {fastest['pages']} pages in {fastest['elapsed_s']:.2f}s)")
        rec["max_articles"] = fastest["pages"] + 5

    print()
    print(_sep("═"))
    print(f"  Full results written to JSON — see --out flag or default path.")
    print(_sep("═"))

    return rec


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Research AI — Pipeline Benchmarker"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM benchmark (fast — only I/O layers)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip web I/O benchmarks (DDG search + HTTP scrape)",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Output JSON path (default: logs/metrics/benchmark_<date>.json)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (
        _ROOT / "logs" / "metrics" / f"benchmark_{date.today()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Interaction log lives next to the summary JSON, with _interactions suffix
    ilog_path = out_path.with_name(
        out_path.stem + "_interactions.jsonl"
    )
    ilog = InteractionLogger(ilog_path)

    print()
    print(_sep("═"))
    print(f"  {_bold_str('Deep Research AI — Pipeline Benchmarker')}")
    print(f"  Date        : {date.today()}")
    print(f"  Summary     : {out_path}")
    print(f"  Interactions: {ilog_path}")
    print(_sep("═"))

    results = BenchmarkResults()

    # ── Run all sections ───────────────────────────────────────────────────
    bench_search(results, offline=args.offline, ilog=ilog)
    bench_scrape(results, offline=args.offline, ilog=ilog)
    bench_embed(results, ilog=ilog)
    bench_evidence(results, ilog=ilog)
    bench_memory(results, ilog=ilog)

    if not args.no_llm:
        print(f"\n{_warn('⚡ LLM benchmark will load the writer model (~4 GB). Ctrl-C to skip.')}")
        try:
            bench_llm(results, ilog=ilog)
        except KeyboardInterrupt:
            print(f"\n  {_warn('LLM benchmark skipped by user.')}")
            ilog.log("llm", "skipped_keyboard_interrupt", 0.0, ok=True,
                     detail={"reason": "KeyboardInterrupt"})
    else:
        print(f"\n{_warn('⚡ [LLM] Skipped (--no-llm flag)')}")
        ilog.log("llm", "skipped_no_llm_flag", 0.0, ok=True, detail={"reason": "--no-llm"})

    # ── Recommendations ────────────────────────────────────────────────────
    rec = _print_recommendations(results)

    # ── Write summary JSON ─────────────────────────────────────────────────
    summary = results.to_dict()
    if rec:
        summary["recommendations"] = rec
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Close interaction log ──────────────────────────────────────────────
    ilog.finish(out_path)

    print(f"\n{_ok(f'✅ Benchmark complete. JSON → {out_path}')}\n")


def _bold_str(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


if __name__ == "__main__":
    main()
