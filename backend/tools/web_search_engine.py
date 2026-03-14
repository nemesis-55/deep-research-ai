"""
Multi-Query Web Search Engine — Deep Research AI

Design pattern: Facade — wraps the existing search_web() primitive with a
fan-out strategy: N queries × M results each, then deduplicates by URL.

Key properties
──────────────
- Reads config from deep_crawl.* keys (num_queries, results_per_query)
- Inter-query delay of 1–2 s to stay within DuckDuckGo rate limits
- URL-level deduplication preserving the highest-scoring occurrence
- Returns a flat, deduplicated list of {title, url, snippet, query} dicts
- Zero impact on the existing search_web() function
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict, List

from backend.config_loader import get
from backend.constants import DDG_INTER_QUERY_DELAY_S
from backend.tools.web_search import search_web

logger = logging.getLogger(__name__)

_INTER_QUERY_DELAY = DDG_INTER_QUERY_DELAY_S


def _url_key(url: str) -> str:
    """Canonical dedup key — strip trailing slash and lowercase scheme+host."""
    url = url.strip().rstrip("/").lower()
    return hashlib.md5(url.encode()).hexdigest()


def multi_query_search(
    queries: List[str],
    results_per_query: int = None,
) -> List[Dict]:
    """
    Run each query through DuckDuckGo and return a flat, deduplicated list.

    Each result dict has keys: title, url, snippet, query (originating query).
    Results are ordered: first occurrence of a URL wins (preserves rank).

    Args:
        queries:           List of search query strings.
        results_per_query: Max DDG results per query (default: deep_crawl.results_per_query).

    Returns:
        Deduplicated list of result dicts.
    """
    rpq = results_per_query or get("deep_crawl.results_per_query", 10)

    seen_keys: Dict[str, bool] = {}
    all_results: List[Dict]    = []

    logger.info(
        f"[MultiSearch] ══ Fan-out search  "
        f"queries={len(queries)}  results_per_query={rpq}  "
        f"max_possible={len(queries) * rpq} ══"
    )

    for i, query in enumerate(queries):
        logger.info(f"[MultiSearch] ── Query {i+1}/{len(queries)} ──────────────────────────────")
        logger.info(f"[MultiSearch]   Input : {query}")

        raw = search_web(query, max_results=rpq)

        logger.debug(f"[MultiSearch]   DDG returned {len(raw)} raw results")
        for j, r in enumerate(raw, 1):
            logger.debug(f"[MultiSearch]     #{j:>2}  {r.get('url', '(no url)')}")

        new_count = 0
        dup_count = 0
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
            f"[MultiSearch]   Output: {len(raw)} raw → {new_count} new unique, "
            f"{dup_count} cross-query dups  (running total: {len(all_results)})"
        )

        # Rate-limit guard — sleep between queries (skip after the last one)
        if i < len(queries) - 1:
            logger.debug(f"[MultiSearch]   💤 sleeping {_INTER_QUERY_DELAY}s …")
            time.sleep(_INTER_QUERY_DELAY)

    logger.info(
        f"[MultiSearch] ══ Done  queries={len(queries)}  "
        f"unique_urls={len(all_results)} ══"
    )
    return all_results
