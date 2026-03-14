"""
Web Search Tool — Deep Research AI

Design pattern: Simple Function Module — a stateless utility with a single
public function search_web().  Uses DuckDuckGo (no API key required) with
exponential back-off (3 attempts: 2 s → 4 s → 8 s) to handle rate-limiting.
"""
import logging
import time
from typing import Dict, List

from backend.config_loader import get
from backend.constants import DDG_BACKOFF_DELAYS

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
        with DDGS() as ddgs:
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
    for attempt, wait in enumerate(DDG_BACKOFF_DELAYS, start=1):
        try:
            results = _run()
            if attempt > 1:
                logger.info(f"[Search] Succeeded on attempt {attempt}.")
            logger.info(f"[Search] '{query[:70]}' → {len(results)} results")
            return results
        except Exception as e:
            last_error = e
            if attempt < len(DDG_BACKOFF_DELAYS):
                logger.warning(
                    f"[Search] DuckDuckGo error (attempt {attempt}/{len(DDG_BACKOFF_DELAYS)}): "
                    f"{e} — retrying in {wait} s"
                )
                time.sleep(wait)
            else:
                logger.error(f"[Search] All {len(DDG_BACKOFF_DELAYS)} attempts failed: {e}")

    return []
