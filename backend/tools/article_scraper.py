"""
Article Scraper — Deep Research AI

Design pattern: Pipeline Stage — takes a list of candidate URLs (from
multi_query_search), selects the top `max_articles` to actually scrape,
deduplicates by near-duplicate content fingerprint, filters by minimum
character length, and stores every passing article in the vector store.

Selection priority
──────────────────
1. Credibility score from source_scorer (domain authority + recency)
2. Snippet length as a proxy for content richness
3. URL stability (prefer stable paths over query-heavy URLs)

Deduplication
─────────────
- URL-level: already handled upstream by multi_query_search
- Content-level: 64-char prefix hash of stripped text (catches mirrors/reposts)

Config keys consumed
────────────────────
  deep_crawl.max_articles    (default 30)  — how many articles to scrape
  deep_crawl.min_doc_chars   (default 200) — discard shorter articles
  deep_crawl.refine_threshold (default 8)  — minimum docs needed before
                                              the pipeline skips a refine loop
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from backend.config_loader import get
from backend.tools.page_scraper import scrape_page
from backend.tools.source_scorer import score_source
from backend.tools.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Patterns that reliably indicate low-value / un-scrapable pages
_SKIP_URL_RE = re.compile(
    r"(youtube\.com|youtu\.be|twitter\.com|x\.com|instagram\.com"
    r"|facebook\.com|tiktok\.com|reddit\.com/r/|pinterest\.com"
    r"|linkedin\.com/in/|\.pdf$|\.zip$|\.exe$)",
    re.IGNORECASE,
)


def _content_key(text: str) -> str:
    """64-char prefix hash for near-duplicate content detection."""
    fingerprint = re.sub(r'\s+', ' ', text.strip())[:512]
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _url_score(url: str, snippet: str) -> int:
    """Rough pre-scrape priority score so we scrape the best URLs first."""
    return score_source(url, snippet, text="")


def _is_skippable(url: str) -> bool:
    return bool(_SKIP_URL_RE.search(url))


class ArticleScraper:
    """
    Scrapes, deduplicates, and stores articles from a candidate URL list.

    Usage::

        scraper = ArticleScraper(vector_store)
        docs = scraper.scrape_and_store(candidates, topic, status_callback=cb)
        # docs is a list of stored article dicts
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store   = vector_store
        self.max_articles   = get("deep_crawl.max_articles",    30)
        self.min_doc_chars  = get("deep_crawl.min_doc_chars",   200)

    def scrape_and_store(
        self,
        candidates: List[Dict],
        topic: str,
        status_callback: Optional[callable] = None,
    ) -> List[Dict]:
        """
        Scrape the best articles from `candidates` and store them in the
        vector store.  Returns a list of successfully scraped article dicts,
        each with keys: url, title, text, snippet, query, credibility_score.

        Args:
            candidates:       Output of multi_query_search() — list of
                              {title, url, snippet, query} dicts.
            topic:            Original research topic (used for metadata).
            status_callback:  Optional callable(str) for progress messages.
        """
        def _status(msg: str) -> None:
            if status_callback:
                status_callback(msg)
            logger.info(f"[ArticleScraper] {msg}")

        # ── 1. Pre-filter obvious skips ───────────────────────────────────
        skipped_pre = [c["url"] for c in candidates if c.get("url") and _is_skippable(c["url"])]
        viable = [c for c in candidates if c.get("url") and not _is_skippable(c["url"])]

        logger.debug(f"[ArticleScraper] ── Pre-filter ──────────────────────────────────────")
        logger.debug(f"[ArticleScraper]   Input candidates : {len(candidates)}")
        logger.debug(f"[ArticleScraper]   Viable           : {len(viable)}")
        logger.debug(f"[ArticleScraper]   Skipped (social/binary): {len(skipped_pre)}")
        for u in skipped_pre:
            logger.debug(f"[ArticleScraper]     ⛔ pre-filtered: {u}")
        _status(f"Candidate URLs: {len(candidates)} total → {len(viable)} viable after pre-filter")

        # ── 2. Priority-rank viable candidates (pre-scrape) ───────────────
        ranked = sorted(
            viable,
            key=lambda c: _url_score(c["url"], c.get("snippet", "")),
            reverse=True,
        )

        logger.debug(f"[ArticleScraper] ── Ranked candidates (top 10 by pre-scrape score) ──")
        for idx, c in enumerate(ranked[:10], 1):
            pre_score = _url_score(c["url"], c.get("snippet", ""))
            logger.debug(f"[ArticleScraper]   #{idx:>2}  score={pre_score:<4}  {c['url']}")

        # ── 3. Scrape up to max_articles ──────────────────────────────────
        seen_content: Dict[str, bool] = {}
        scraped: List[Dict]           = []
        attempted                     = 0

        logger.debug(
            f"[ArticleScraper] ── Scrape loop  "
            f"(max={self.max_articles}, min_chars={self.min_doc_chars}) ──"
        )

        for candidate in ranked:
            if len(scraped) >= self.max_articles:
                logger.debug(
                    f"[ArticleScraper]   ✋ max_articles={self.max_articles} reached — stopping loop"
                )
                break

            url     = candidate["url"]
            title   = candidate.get("title", url)
            snippet = candidate.get("snippet", "")
            query   = candidate.get("query", topic)
            attempted += 1

            logger.debug(
                f"[ArticleScraper] ── Attempt {attempted}  "
                f"(stored so far: {len(scraped)}/{self.max_articles}) ──"
            )
            logger.debug(f"[ArticleScraper]   URL   : {url}")
            logger.debug(f"[ArticleScraper]   Title : {title[:80]}")
            logger.debug(f"[ArticleScraper]   Query : {query[:60]}")

            _status(f"  [{len(scraped)+1}/{self.max_articles}] Scraping: {title[:70]}")

            try:
                page = scrape_page(
                    url,
                    follow_links=False,   # Phase 0: breadth over depth
                    follow_links_depth=0,
                    max_follow_links=0,
                )
            except Exception as e:
                logger.debug(f"[ArticleScraper]   ❌ SCRAPE ERROR: {e}")
                _status(f"    ⚠️ Scrape error {url[:55]}: {e}")
                continue

            text = (page.get("text") or "").strip()
            logger.debug(f"[ArticleScraper]   Raw text length: {len(text):,} chars")

            # ── 4. Content quality gate ───────────────────────────────────
            if len(text) < self.min_doc_chars:
                logger.debug(
                    f"[ArticleScraper]   ⏭ SKIP (too short: {len(text)} < {self.min_doc_chars})"
                )
                _status(f"    ⏭ Too short ({len(text)} chars): {url[:55]}")
                continue

            # ── 5. Content-level deduplication ────────────────────────────
            ckey = _content_key(text)
            if ckey in seen_content:
                logger.debug(f"[ArticleScraper]   ⏭ SKIP (near-duplicate, key={ckey})")
                _status(f"    ⏭ Near-duplicate content: {url[:55]}")
                continue
            seen_content[ckey] = True
            logger.debug(f"[ArticleScraper]   Content key (dedup): {ckey}")

            # ── 6. Credibility score ──────────────────────────────────────
            cred = score_source(url, snippet, text)
            logger.debug(f"[ArticleScraper]   Credibility score: {cred}")

            # ── 7. Store in vector store ──────────────────────────────────
            store_text = (
                f"Topic: {topic}\n"
                f"Query: {query}\n"
                f"Source: {title}\n"
                f"URL: {url}\n\n"
                f"{text[:8000]}"
            )
            try:
                self.vector_store.store_document(
                    text     = store_text,
                    metadata = {
                        "url":              url,
                        "title":            title,
                        "task":             f"deep_crawl:{query[:60]}",
                        "snippet":          snippet[:300],
                        "credibility":      cred,
                        "phase":            "deep_crawl",
                    },
                )
            except Exception as e:
                logger.debug(f"[ArticleScraper]   ❌ STORE ERROR: {e}")
                _status(f"    ⚠️ Store failed {url[:55]}: {e}")
                continue

            scraped.append({
                "url":              url,
                "title":            title,
                "text":             text[:3000],
                "snippet":          snippet,
                "query":            query,
                "credibility_score": cred,
                "images":           page.get("images", [])[:4],
            })

            logger.debug(
                f"[ArticleScraper]   ✅ STORED  #{len(scraped)}  "
                f"chars={len(text):,}  cred={cred}  key={ckey}"
            )
            _status(
                f"    ✅ Stored ({len(text):,} chars, cred={cred}) — {title[:55]}"
            )

        logger.debug(
            f"[ArticleScraper] ── Scrape loop complete ──────────────────────"
        )
        logger.debug(f"[ArticleScraper]   Attempted : {attempted}")
        logger.debug(f"[ArticleScraper]   Stored    : {len(scraped)}")
        logger.debug(f"[ArticleScraper]   Skipped   : {attempted - len(scraped)}")
        logger.debug(f"[ArticleScraper]   Duplicates in content-hash map: {len(seen_content)}")

        _status(
            f"Deep crawl complete: {len(scraped)} articles stored "
            f"(attempted {attempted}/{len(viable)} viable URLs)"
        )
        return scraped
