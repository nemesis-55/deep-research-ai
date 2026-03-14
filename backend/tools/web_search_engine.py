"""
Multi-Query Web Search Engine — Deep Research AI

Design pattern: Facade — wraps the existing search_web() primitive with a
fan-out strategy: N queries × M results each, then deduplicates by URL.

Key properties
──────────────
- Reads config from deep_crawl.* keys (num_queries, results_per_query)
- Parallel query execution via ThreadPoolExecutor (I/O-bound — safe to parallelise)
- Max workers capped at parallel_search_workers (default 3) to respect DDG rate limits
- Inter-query delay applied per-worker when sequential fallback is used
- URL-level deduplication preserving the highest-scoring occurrence
- Returns a flat, deduplicated list of {title, url, snippet, query} dicts
- Zero impact on the existing search_web() function

Parallelism model
─────────────────
  Web search (DDG HTTP)  → I/O-bound → ThreadPoolExecutor (safe, 2–3 workers)
  Web scraping (HTTP)    → I/O-bound → ThreadPoolExecutor (safe, 2–4 workers)
  LLM generation (MLX)   → GPU-bound → sequential, protected by _LLM_SEMAPHORE
  Embedding (CPU/Metal)  → CPU-bound → sequential (model_manager holds _lock)
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from backend.config_loader import get
from backend.constants import DDG_INTER_QUERY_DELAY_S
from backend.tools.web_search import search_web

logger = logging.getLogger(__name__)

_INTER_QUERY_DELAY = DDG_INTER_QUERY_DELAY_S

# Thread-safe merge lock for parallel result accumulation
_merge_lock = threading.Lock()


def _url_key(url: str) -> str:
    """Canonical dedup key — strip trailing slash and lowercase scheme+host."""
    url = url.strip().rstrip("/").lower()
    return hashlib.md5(url.encode()).hexdigest()


def _search_one(query: str, rpq: int, worker_id: int) -> Tuple[str, List[Dict]]:
    """Execute a single DDG query and return (query, results). Thread-safe."""
    logger.info(f"[MultiSearch] Worker-{worker_id} ▶ '{query[:70]}'")
    t0 = time.monotonic()
    raw = search_web(query, max_results=rpq)
    elapsed = time.monotonic() - t0
    logger.info(
        f"[MultiSearch] Worker-{worker_id} ◀ {len(raw)} results "
        f"in {elapsed:.2f}s for '{query[:50]}'"
    )
    return query, raw


def multi_query_search(
    queries: List[str],
    results_per_query: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[Dict]:
    """
    Run each query through DuckDuckGo **in parallel** and return a flat,
    deduplicated list.

    Each result dict has keys: title, url, snippet, query (originating query).
    Results are ordered: first-query results first (preserves rank within query),
    with cross-query duplicates removed.

    Args:
        queries:           List of search query strings.
        results_per_query: Max DDG results per query (default: deep_crawl.results_per_query).
        max_workers:       Thread pool size (default: parallel_search_workers config key,
                           capped at min(3, len(queries)) to avoid DDG rate limits).

    Returns:
        Deduplicated list of result dicts.
    """
    rpq = results_per_query or get("deep_crawl.results_per_query", 10)

    # Worker count: read config, cap to min(3, n_queries) to respect DDG limits
    cfg_workers = max_workers or get("parallel_search_workers", 2)
    n_workers   = min(cfg_workers, len(queries), 3)

    seen_keys: Dict[str, bool] = {}
    all_results: List[Dict]    = []

    logger.info(
        f"[MultiSearch] ══ Fan-out search  "
        f"queries={len(queries)}  results_per_query={rpq}  "
        f"workers={n_workers}  max_possible={len(queries) * rpq} ══"
    )

    if n_workers <= 1 or len(queries) == 1:
        # ── Sequential path (fallback / single query) ─────────────────────
        for i, query in enumerate(queries):
            logger.info(f"[MultiSearch] ── Query {i+1}/{len(queries)} (sequential) ──")
            _, raw = _search_one(query, rpq, worker_id=0)
            _merge_raw(raw, query, seen_keys, all_results)
            if i < len(queries) - 1:
                logger.debug(f"[MultiSearch]   💤 sleeping {_INTER_QUERY_DELAY}s …")
                time.sleep(_INTER_QUERY_DELAY)
    else:
        # ── Parallel path — fire all queries concurrently ─────────────────
        # Order results by query index so ranking is deterministic
        ordered: List[Optional[Tuple[str, List[Dict]]]] = [None] * len(queries)

        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ddg") as pool:
            futures = {
                pool.submit(_search_one, q, rpq, i): i
                for i, q in enumerate(queries)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    ordered[idx] = fut.result()
                except Exception as e:
                    logger.warning(f"[MultiSearch] Query {idx} failed: {e}")
                    ordered[idx] = (queries[idx], [])

        # Merge in original query order (so first query results rank first)
        for i, item in enumerate(ordered):
            if item is None:
                continue
            query, raw = item
            logger.info(f"[MultiSearch] ── Merging query {i+1}/{len(queries)} ({len(raw)} raw) ──")
            _merge_raw(raw, query, seen_keys, all_results)

    logger.info(
        f"[MultiSearch] ══ Done  queries={len(queries)}  "
        f"unique_urls={len(all_results)} ══"
    )
    return all_results


def _merge_raw(
    raw: List[Dict],
    query: str,
    seen_keys: Dict[str, bool],
    all_results: List[Dict],
) -> None:
    """Merge raw DDG results into all_results, deduplicating by URL. Thread-safe."""
    new_count = dup_count = 0
    with _merge_lock:
        for r in raw:
            url = r.get("url", "").strip()
            if not url:
                continue
            key = _url_key(url)
            if key in seen_keys:
                dup_count += 1
                logger.debug(f"[MultiSearch]   ⏭ dup: {url}")
                continue
            seen_keys[key] = True
            all_results.append({
                "title":   r.get("title", ""),
                "url":     url,
                "snippet": r.get("snippet", ""),
                "query":   query,
            })
            new_count += 1
    logger.info(
        f"[MultiSearch]   Merged: {len(raw)} raw → {new_count} new, "
        f"{dup_count} dups  (total: {len(all_results)})"
    )


def parallel_scrape_pages(
    candidates: List[Dict],
    max_workers: Optional[int] = None,
    min_chars: int = 1000,
    follow_links: bool = False,
) -> List[Dict]:
    """
    Scrape a list of candidate URL dicts **in parallel** using a thread pool.

    HTTP scraping is I/O-bound — threads block on network, not CPU/GPU — so
    running 2–4 concurrent scrapers is safe alongside the sequential LLM.

    Args:
        candidates:  List of {url, title, snippet, ...} dicts from multi_query_search.
        max_workers: Thread pool size (default: parallel_scrape_workers config key, max 4).
        min_chars:   Minimum text length to accept a page (default 1000).
        follow_links: Whether to follow internal links during scraping (default False).

    Returns:
        List of scraped page dicts with keys: url, title, text, images, youtube_embeds.
        Ordered by original candidate rank. Failures / short pages are silently skipped.
    """
    from backend.tools.page_scraper import scrape_page

    if not candidates:
        return []

    cfg_workers = max_workers or get("parallel_scrape_workers", 3)
    n_workers   = min(cfg_workers, len(candidates), 4)

    logger.info(
        f"[ParallelScrape] ══ Scraping {len(candidates)} URLs  "
        f"workers={n_workers}  min_chars={min_chars} ══"
    )

    def _scrape_one(idx_cand: Tuple[int, Dict]) -> Tuple[int, Optional[Dict]]:
        idx, cand = idx_cand
        url   = cand.get("url", "")
        title = cand.get("title", url)
        if not url:
            return idx, None
        try:
            t0   = time.monotonic()
            page = scrape_page(url, follow_links=follow_links,
                               follow_links_depth=1, max_follow_links=2)
            text = (page.get("text") or "").strip()
            elapsed = time.monotonic() - t0
            logger.info(
                f"[ParallelScrape] [{idx+1}] {len(text):,} chars "
                f"in {elapsed:.2f}s — {url[:70]}"
            )
            if len(text) < min_chars:
                logger.debug(
                    f"[ParallelScrape]   ⏭ SKIP too short "
                    f"({len(text)} < {min_chars}) — {url[:60]}"
                )
                return idx, None
            return idx, {
                "url":            url,
                "title":          title,
                "text":           text,
                "domain":         cand.get("domain", ""),
                "images":         page.get("images", []),
                "youtube_embeds": page.get("youtube_embeds", []),
            }
        except Exception as e:
            logger.debug(f"[ParallelScrape]   ❌ {url[:60]}: {e}")
            return idx, None

    results: List[Optional[Dict]] = [None] * len(candidates)

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="scrape") as pool:
        futures = {
            pool.submit(_scrape_one, (i, c)): i
            for i, c in enumerate(candidates)
        }
        for fut in as_completed(futures):
            try:
                idx, page = fut.result()
                results[idx] = page
            except Exception as e:
                logger.warning(f"[ParallelScrape] Unexpected future error: {e}")

    pages = [p for p in results if p is not None]
    logger.info(
        f"[ParallelScrape] ══ Done  scraped={len(pages)}/{len(candidates)} ══"
    )
    return pages
