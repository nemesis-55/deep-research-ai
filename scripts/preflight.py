#!/usr/bin/env python3
"""
preflight.py — Deep Research AI
================================
Pre-flight test suite that runs BEFORE the server starts.

Checks (in order):
  1.  Package imports        — every required pip package importable
  2.  Config loading         — config.yaml parses, required keys present
  3.  Constants              — backend/constants.py importable, key values sane
  4.  Local cache folder     — cache/hub exists and is readable
  5.  Model files            — GGUF / MLX cache exists on disk
  6.  Web search             — DuckDuckGo returns ≥ 1 result (live network)
  7.  Web scraper            — trafilatura / BS4 can scrape a known stable URL
  8.  Chat web-search flow   — _needs_web_search() correctly classifies prompts
  9.  Model load + inference — loads the chat model, runs one generation
  10. Vector store           — ChromaDB initialises and round-trips a document
  11. Planner agent          — loads planner model, generates a research plan
  12. Full pipeline smoke    — runs one pipeline step end-to-end (no server needed)

Usage:
    python scripts/preflight.py            # full suite
    python scripts/preflight.py --fast     # skip model load + pipeline (checks 1-8, 10)
    python scripts/preflight.py --web-only # checks 6 + 7 only

Exit code 0 = all checks passed (or only warnings).
Exit code 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

# ── Bootstrap path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ── Silence noisy third-party DEBUG loggers immediately ───────────────────────
import logging as _logging
for _noisy in (
    "trafilatura", "trafilatura.htmlprocessing", "trafilatura.main_extractor",
    "trafilatura.readability_lxml", "trafilatura.external",
    "htmldate", "htmldate.validators",
    "filelock", "filelock._unix", "filelock._windows",
    "chromadb", "chromadb.config",
    "sentence_transformers", "sentence_transformers.SentenceTransformer",
    "httpx", "httpcore", "urllib3", "transformers", "huggingface_hub",
    "rustls", "rustls.common_state",
):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

# ── ANSI colours ───────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

# ── Result tracking ────────────────────────────────────────────────────────────
_results: list[dict] = []


def _print_header(title: str) -> None:
    width = 60
    print(f"\n{_BOLD}{_CYAN}{'═' * width}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * width}{_RESET}")


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  {_CYAN}{detail}{_RESET}" if detail else ""
    print(f"  {_GREEN}✅ PASS{_RESET}  {label}{suffix}")
    _results.append({"label": label, "status": "pass"})


def _warn(label: str, detail: str = "") -> None:
    suffix = f"\n         {_YELLOW}{detail}{_RESET}" if detail else ""
    print(f"  {_YELLOW}⚠️  WARN{_RESET}  {label}{suffix}")
    _results.append({"label": label, "status": "warn", "detail": detail})


def _fail(label: str, detail: str = "") -> None:
    suffix = f"\n         {_RED}{detail}{_RESET}" if detail else ""
    print(f"  {_RED}❌ FAIL{_RESET}  {label}{suffix}")
    _results.append({"label": label, "status": "fail", "detail": detail})


def _run(label: str, fn: Callable, warn_only: bool = False) -> bool:
    """Run fn(); record pass/warn/fail; return True if not a hard fail."""
    t0 = time.monotonic()
    try:
        result = fn()
        elapsed = time.monotonic() - t0
        detail  = result if isinstance(result, str) else f"{elapsed:.2f}s"
        _ok(label, detail)
        return True
    except Warning as w:
        _warn(label, str(w))
        return True
    except Exception as e:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc().strip().splitlines()[-1]
        if warn_only:
            _warn(label, f"{e}  [{elapsed:.2f}s]")
            return True
        _fail(label, f"{e}  [{elapsed:.2f}s]\n         {tb}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 1 — Package imports
# ══════════════════════════════════════════════════════════════════════════════

def check_packages() -> None:
    _print_header("1 · Package imports")
    from backend.constants import REQUIRED_PACKAGES
    failed = []
    for pkg in REQUIRED_PACKAGES:
        try:
            spec = importlib.util.find_spec(pkg)
            if spec is None:
                raise ImportError(f"find_spec returned None")
            _ok(pkg)
        except Exception as e:
            _fail(pkg, str(e))
            failed.append(pkg)
    if failed:
        raise RuntimeError(f"Missing packages: {failed}")


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 2 — Config loading
# ══════════════════════════════════════════════════════════════════════════════

def check_config() -> None:
    _print_header("2 · Config loading")

    def _load():
        from backend.config_loader import get
        required_keys = [
            "models.planner.name",
            "models.writer.name",
            "models.chat.name",
            "models.embedding.name",
            "server.host",
            "server.port",
            "storage.models",
            "deep_crawl.enabled",
        ]
        missing = [k for k in required_keys if get(k) is None]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        return f"{len(required_keys)} keys OK"

    _run("config.yaml loads + required keys present", _load)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 3 — Constants
# ══════════════════════════════════════════════════════════════════════════════

def check_constants() -> None:
    _print_header("3 · Constants (backend/constants.py)")

    def _import():
        from backend import constants as C
        assert C.CHAT_MAX_TOKENS["trivial"]        == 256,  "trivial budget wrong"
        assert C.CHAT_MAX_TOKENS["conversational"] == 768,  "conversational budget wrong"
        assert C.CHAT_MAX_TOKENS["technical"]      == 1536, "technical budget wrong"  # 1536 @ ~10 tok/s (14B) ≈ 154 s
        assert len(C.SEARCH_PHRASES) >= 10,    "SEARCH_PHRASES too short"
        assert len(C.TECHNICAL_KEYWORDS) >= 10, "TECHNICAL_KEYWORDS too short"
        assert C.DDG_BACKOFF_DELAYS == (2, 4, 8), "DDG back-off wrong"
        assert C.SCRAPER_TIMEOUT_S  == 15,        "scraper timeout wrong"
        return f"{len(vars(C))} constants exported"

    _run("constants.py imports and values sane", _import)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 4 — External drive
# ══════════════════════════════════════════════════════════════════════════════

def check_drive() -> None:
    _print_header("4 · Local model cache (cache/hub)")
    from pathlib import Path
    _root = Path(__file__).parent.parent
    cache_hub = _root / "cache" / "hub"

    def _check():
        if not cache_hub.exists():
            raise Warning(
                f"cache/hub not found at {cache_hub} — "
                "run Step 2 (prefetch) to download models"
            )
        return str(cache_hub)

    _run("cache/hub exists", _check, warn_only=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 5 — Model files on disk
# ══════════════════════════════════════════════════════════════════════════════

def check_model_files() -> None:
    _print_header("5 · Model files on disk")
    from backend.config_loader import get

    # Check GGUF files — only relevant for roles using llama_cpp runtime
    for role in ("planner", "writer", "chat"):
        runtime = get(f"models.{role}.runtime", "mlx")
        if runtime != "llama_cpp":
            continue   # MLX roles don't need a GGUF on disk
        def _check_gguf(r=role):
            p = Path(get(f"models.{r}.gguf_path", ""))
            if not p.exists():
                raise Warning(f"GGUF not found: {p} (will download on first use)")
            return f"{p.stat().st_size / 1e9:.1f} GB"
        _run(f"{role} GGUF on disk", _check_gguf, warn_only=True)

    # Check MLX cache — read cache dir from config.yaml (same as model_manager)
    _raw_hf = get("storage.hf_cache", str(Path.home() / ".cache" / "huggingface" / "hub"))
    hf_cache = Path(_raw_hf).expanduser()
    for role in ("planner", "writer", "chat"):
        def _check_mlx(r=role):
            repo = get(f"models.{r}.mlx_repo", "")
            repo_dir_name = "models--" + repo.replace("/", "--")
            mlx_path = hf_cache / repo_dir_name
            if not mlx_path.exists():
                raise Warning(f"MLX cache missing: {mlx_path}")
            return str(mlx_path)
        _run(f"{role} MLX cache exists", _check_mlx, warn_only=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 6 — Live web search (DuckDuckGo)
# ══════════════════════════════════════════════════════════════════════════════

def check_web_search() -> None:
    _print_header("6 · Live web search (DuckDuckGo)")

    def _search():
        from backend.tools.web_search import search_web
        results = search_web("Apple Silicon M4 Mac Mini", max_results=3)
        if not results:
            raise RuntimeError("DuckDuckGo returned 0 results — network issue or rate-limited")
        urls = [r["url"] for r in results]
        return f"{len(results)} results: {urls[0][:60]}…"

    _run("DuckDuckGo search returns results", _search)

    def _classify():
        from backend.main import _needs_web_search, _classify_prompt
        # Generic phrases that must trigger a web search (no brand / company bias)
        triggers = [
            "latest tech news today",
            "current weather forecast this week",
            "what's happening in AI right now",
            "recent breakthroughs in quantum computing",
            "breaking news this morning",
        ]
        # Phrases that must NOT trigger a web search
        non_triggers = [
            "hi",
            "explain how transformers work",
            "what is RAG?",
            "how does attention mechanism work",
        ]
        for msg in triggers:
            assert _needs_web_search(msg), f"Should trigger search: {msg!r}"
            assert _classify_prompt(msg) == "technical", f"Should be technical: {msg!r}"
        for msg in non_triggers:
            assert not _needs_web_search(msg), f"Should NOT trigger search: {msg!r}"
        return f"{len(triggers)} triggers, {len(non_triggers)} non-triggers classified correctly"

    _run("_needs_web_search() + _classify_prompt() logic", _classify)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 7 — Web scraper
# ══════════════════════════════════════════════════════════════════════════════

def check_scraper() -> None:
    _print_header("7 · Web scraper (trafilatura + BS4)")

    def _scrape():
        from backend.tools.page_scraper import scrape_page
        # Use a stable, lightweight URL unlikely to block scrapers
        result = scrape_page("https://en.wikipedia.org/wiki/Apple_silicon", timeout=20)
        text = (result.get("text") or "").strip()
        if len(text) < 200:
            raise RuntimeError(f"Scrape returned too little text ({len(text)} chars) — possible block")
        return f"{len(text):,} chars scraped from Wikipedia"

    _run("scrape_page() extracts article text", _scrape)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 8 — Chat web-search end-to-end (search → context block)
# ══════════════════════════════════════════════════════════════════════════════

def check_chat_web_search() -> None:
    _print_header("8 · Chat web-search context builder (adaptive scraping)")

    def _build():
        from backend.main import _chat_web_search, _build_search_query, _score_confidence

        # ── Basic context block ───────────────────────────────────────────────
        ctx = _chat_web_search("latest Apple Silicon M4 news")
        if not ctx or len(ctx) < 100:
            raise RuntimeError(f"Web context too short ({len(ctx)} chars)")
        assert "Live web search" in ctx, "Missing header in context block"

        # ── Context-enriched query: pronouns resolved from history ─────────────
        # Simulate a generic prior turn about any topic; verify the enriched
        # query carries key words forward without hard-coding a company/brand.
        history = [{"role": "user", "content": "latest news on quantum computing chips"}]
        enriched = _build_search_query("any update on it today", history)
        assert "quantum" in enriched.lower(), f"Context enrichment failed: {enriched!r}"

        # ── Confidence scorer unit tests ──────────────────────────────────────
        # Low volume + identical snippets → low confidence
        low_conf = _score_confidence(
            scraped_texts=["tiny"],
            result_snippets=["apple apple apple", "apple apple apple"],
        )
        assert low_conf < 0.65, f"Expected low confidence, got {low_conf:.2f}"

        # High volume + diverse snippets → high confidence
        high_conf = _score_confidence(
            scraped_texts=["x" * 3000, "y" * 3000, "z" * 3000],
            result_snippets=[
                "quantum computing breakthrough research IBM",
                "apple silicon M4 chip performance benchmark",
                "reliance industries quarterly earnings revenue profit",
                "climate change renewable energy solar wind",
            ],
        )
        assert high_conf >= 0.65, f"Expected high confidence, got {high_conf:.2f}"

        return (
            f"Context: {len(ctx):,} chars · enrichment OK · "
            f"confidence scorer: low={low_conf:.2f} high={high_conf:.2f}"
        )

    _run("_chat_web_search() adaptive scraping + confidence scorer", _build)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 9 — Model load + inference (chat model)
# ══════════════════════════════════════════════════════════════════════════════

def check_model_inference() -> None:
    _print_header("9 · Model load + inference (chat model)")

    def _load_and_generate():
        from backend.model_manager import load_chat_model, generate, unload_model
        print(f"    Loading chat model…", flush=True)
        handle = load_chat_model()
        assert handle is not None, "load_chat_model() returned None"
        print(f"    ✓ Loaded: {handle.name} ({handle.runtime})", flush=True)

        print(f"    Running warmup inference…", flush=True)
        response = generate(
            handle,
            prompt="Reply with exactly three words: PREFLIGHT TEST PASSED",
            max_new_tokens=16,
            temperature=0.0,
        )
        assert response.strip(), "Empty response from model"
        print(f"    ✓ Response: {response.strip()!r}", flush=True)

        # Test token throughput
        t0 = time.monotonic()
        long_resp = generate(
            handle,
            prompt="List 5 benefits of local AI inference in one sentence each.",
            max_new_tokens=256,
            temperature=0.7,
        )
        elapsed = time.monotonic() - t0
        tokens  = len(long_resp.split())
        tps     = tokens / elapsed if elapsed > 0 else 0
        print(f"    ✓ Throughput: ~{tps:.1f} tok/s  ({tokens} words / {elapsed:.1f}s)", flush=True)

        unload_model()
        return f"{handle.runtime} · ~{tps:.1f} tok/s"

    _run("chat model loads + generates coherent output", _load_and_generate)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 10 — Vector store round-trip
# ══════════════════════════════════════════════════════════════════════════════

def check_vector_store() -> None:
    _print_header("10 · Vector store (ChromaDB + embeddings)")

    def _roundtrip():
        from backend.tools.vector_store import VectorStore
        vs = VectorStore()
        vs.clear()

        text = "Apple Silicon M4 chip delivers exceptional AI performance on-device. " * 5
        chunk_ids = vs.store_document(
            text,
            metadata={"title": "Preflight test document", "url": "https://preflight.test/doc"},
        )
        assert chunk_ids, "store_document() returned empty list"

        results = vs.get_context_for_query("Apple Silicon performance", n_results=1)
        assert results and len(results) > 10, f"Retrieved context too short: {results!r}"

        count = vs.count()
        vs.clear()
        return f"stored + retrieved OK  ({count} chunks, {len(chunk_ids)} IDs)"

    _run("ChromaDB stores and retrieves a document", _roundtrip)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 11 — Planner agent (DeepSeek-R1)
# ══════════════════════════════════════════════════════════════════════════════

def check_planner() -> None:
    _print_header("11 · Planner agent (DeepSeek-R1)")

    def _plan():
        from backend.model_manager import load_planner_model, unload_model
        from backend.agent.planner_agent import PlannerAgent
        print(f"    Loading planner model…", flush=True)
        load_planner_model()
        agent = PlannerAgent()
        plan  = agent.generate_plan("Latest developments in quantum computing 2026")
        assert isinstance(plan, list) and len(plan) >= 1, \
            f"Plan must be a non-empty list, got: {plan!r}"
        print(f"    ✓ Plan: {len(plan)} tasks", flush=True)
        for i, task in enumerate(plan[:3], 1):
            print(f"      {i}. {task[:80]}", flush=True)
        unload_model()
        return f"{len(plan)} research tasks generated"

    _run("PlannerAgent generates a valid research plan", _plan)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 12 — Full pipeline smoke test (one task, no server)
# ══════════════════════════════════════════════════════════════════════════════

def check_pipeline_smoke() -> None:
    _print_header("12 · Full pipeline smoke (one task, no server)")

    def _smoke():
        import asyncio
        from backend.pipeline.research_pipeline import ResearchPipeline

        pipeline = ResearchPipeline()

        events: list[dict] = []
        event_types: set[str] = set()

        async def _collect():
            async for ev in pipeline.run(
                query="What is Apple Silicon M4?",
                uploaded_files=[],
            ):
                events.append(ev)
                event_types.add(ev.get("type", ""))
                t = ev.get("type", "")
                msg = ev.get("message", ev.get("content", "")[:60])
                print(f"    [{t:10s}] {msg}", flush=True)
                # Stop early after report is generated — don't need full run
                if t in ("report", "error"):
                    break

        asyncio.run(_collect())

        if any(e.get("type") == "error" for e in events):
            err = next(e for e in events if e.get("type") == "error")
            raise RuntimeError(f"Pipeline emitted error: {err.get('message')}")

        assert "report" in event_types, \
            f"Pipeline never emitted 'report' event. Got: {event_types}"

        report_ev = next(e for e in events if e.get("type") == "report")
        report_text = report_ev.get("content", "")
        assert len(report_text) > 200, \
            f"Report too short ({len(report_text)} chars)"

        return f"Pipeline OK · {len(events)} events · report {len(report_text):,} chars"

    _run("ResearchPipeline runs end-to-end and emits a report", _smoke)


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary() -> int:
    passed = [r for r in _results if r["status"] == "pass"]
    warned = [r for r in _results if r["status"] == "warn"]
    failed = [r for r in _results if r["status"] == "fail"]

    width = 60
    print(f"\n{_BOLD}{'═' * width}{_RESET}")
    print(f"{_BOLD}  Pre-flight Summary{_RESET}")
    print(f"{'═' * width}")
    print(f"  {_GREEN}✅ Passed : {len(passed)}{_RESET}")
    if warned:
        print(f"  {_YELLOW}⚠️  Warned : {len(warned)}{_RESET}")
        for r in warned:
            print(f"      • {r['label']}")
    if failed:
        print(f"  {_RED}❌ Failed : {len(failed)}{_RESET}")
        for r in failed:
            print(f"      • {r['label']}")
            if r.get("detail"):
                print(f"        {_RED}{r['detail'][:120]}{_RESET}")
    print(f"{'═' * width}\n")

    if failed:
        print(f"{_RED}{_BOLD}  ✗  Pre-flight FAILED — fix the above errors before starting the server.{_RESET}\n")
        return 1
    if warned:
        print(f"{_YELLOW}{_BOLD}  ⚠  Pre-flight passed with warnings — server should still start.{_RESET}\n")
    else:
        print(f"{_GREEN}{_BOLD}  ✓  All checks passed — safe to start the server.{_RESET}\n")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deep Research AI — pre-flight check suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip model load, planner, and pipeline checks (checks 1-8, 10)",
    )
    parser.add_argument(
        "--web-only", action="store_true",
        help="Run only web search + scraper checks (6 + 7)",
    )
    parser.add_argument(
        "--no-pipeline", action="store_true",
        help="Skip full pipeline smoke test (check 12) but run all others",
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}{_CYAN}  🔬  Deep Research AI — Pre-flight Check{_RESET}")
    print(f"{_CYAN}  {time.strftime('%Y-%m-%d %H:%M:%S')}{_RESET}")

    if args.web_only:
        check_web_search()
        check_scraper()
        return _print_summary()

    # ── Always run ─────────────────────────────────────────────────────────────
    check_packages()
    check_config()
    check_constants()
    check_drive()
    check_model_files()
    check_web_search()
    check_scraper()
    check_chat_web_search()
    check_vector_store()

    if args.fast:
        print(f"\n  {_YELLOW}--fast mode: skipping model inference, planner, pipeline.{_RESET}")
        return _print_summary()

    # ── Heavy checks (need model load) ─────────────────────────────────────────
    check_model_inference()

    if not args.no_pipeline:
        check_planner()
        check_pipeline_smoke()
    else:
        print(f"\n  {_YELLOW}--no-pipeline: skipping planner + pipeline smoke.{_RESET}")

    return _print_summary()


if __name__ == "__main__":
    sys.exit(main())
