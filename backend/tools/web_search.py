"""
Web Search Tool — Deep Research AI

Design pattern: Simple Function Module — a stateless utility with two public
functions:
  - search_web()        single DDG query, exponential back-off
  - fan_out_search()    multi-query fan-out, dedup, cap at max_urls

Fan-out model
─────────────
  One DDG call per query (max_results=10 each), results deduplicated by URL,
  capped at max_urls (default 15) unique URLs.  Preserves first-seen rank so
  higher-ranking queries dominate when there is overlap.
"""
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional

from backend.config_loader import get
from backend.constants import DDG_BACKOFF_DELAYS

# Hard wall-clock budget for one full search_web() call (all retries included).
# primp's per-request socket timeout is set via DDGS(timeout=…).
_SEARCH_HARD_TIMEOUT_S = 30

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = None) -> List[Dict]:
    """
    Search the web using DuckDuckGo.
    Returns list of {title, url, snippet}.
    Retries up to 3 times with exponential back-off on failure.
    """
    n = max_results or get("search.max_results", 8)

    def _run() -> List[Dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # fallback to old package name
        # timeout=15 sets the per-request socket timeout in primp's HTTP client
        with DDGS(timeout=15) as ddgs:
            raw = list(ddgs.text(query, max_results=n))
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
            if r.get("href")
        ]

    last_error: Exception = Exception("No attempts made")
    deadline = time.monotonic() + _SEARCH_HARD_TIMEOUT_S

    for attempt, wait in enumerate(DDG_BACKOFF_DELAYS, start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.error("[Search] Hard timeout reached before attempt %d", attempt)
            break
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run)
                results = future.result(timeout=remaining)
            if attempt > 1:
                logger.info(f"[Search] Succeeded on attempt {attempt}.")
            logger.info(f"[Search] '{query[:70]}' → {len(results)} results")
            return results
        except FuturesTimeoutError:
            logger.error(
                "[Search] DDG call timed out after %.0fs (attempt %d/%d)",
                remaining, attempt, len(DDG_BACKOFF_DELAYS),
            )
            break  # no point retrying — network is stuck
        except Exception as e:
            last_error = e
            if attempt < len(DDG_BACKOFF_DELAYS):
                sleep_s = min(wait, deadline - time.monotonic())
                if sleep_s > 0:
                    logger.warning(
                        f"[Search] DuckDuckGo error (attempt {attempt}/{len(DDG_BACKOFF_DELAYS)}): "
                        f"{e} — retrying in {sleep_s:.0f}s"
                    )
                    time.sleep(sleep_s)
            else:
                logger.error(f"[Search] All {len(DDG_BACKOFF_DELAYS)} attempts failed: {e}")

    return []


def _url_key(url: str) -> str:
    """Canonical dedup key — strip trailing slash and lowercase."""
    return hashlib.md5(url.strip().rstrip("/").lower().encode()).hexdigest()


def fan_out_search(
    queries: List[str],
    results_per_query: int = 10,
    max_urls: int = 15,
) -> List[Dict]:
    """
    Run each query through DDG (max_results=10 each), deduplicate by URL,
    and cap at `max_urls` unique results.

    Each result dict has keys: title, url, snippet, query.
    Results are ordered: first-query results first; cross-query duplicates
    are dropped (first-seen wins, preserving rank).

    Args:
        queries:           List of search query strings.
        results_per_query: DDG max_results per query (default 10).
        max_urls:          Hard cap on total unique URLs returned (default 15).

    Returns:
        Deduplicated, capped list of result dicts.
    """
    seen: Dict[str, bool] = {}
    all_results: List[Dict] = []

    logger.info(
        f"[Search] Fan-out: {len(queries)} queries × {results_per_query} results "
        f"→ cap {max_urls} unique URLs"
    )

    for i, query in enumerate(queries):
        if len(all_results) >= max_urls:
            logger.info(f"[Search] Fan-out: URL cap ({max_urls}) reached at query {i+1}")
            break

        raw = search_web(query, max_results=results_per_query)
        new_count = 0
        for r in raw:
            if len(all_results) >= max_urls:
                break
            url = r.get("url", "").strip()
            if not url:
                continue
            key = _url_key(url)
            if key in seen:
                continue
            seen[key] = True
            all_results.append({
                "title":   r.get("title", ""),
                "url":     url,
                "snippet": r.get("snippet", ""),
                "query":   query,
            })
            new_count += 1

        logger.info(
            f"[Search] Fan-out query {i+1}/{len(queries)}: "
            f"{new_count} new URLs (total {len(all_results)})"
        )

    logger.info(f"[Search] Fan-out complete → {len(all_results)} unique URLs")
    return all_results
