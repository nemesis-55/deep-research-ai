"""
Source Filter — Deep Research AI
==================================
Domain-based blocklist and preferlist for filtering/boosting web search results
before scraping.

Design
──────
- BLOCKLIST  — domains that produce low-signal content (social media, forums,
               e-commerce, opinion sites). Results from these are removed.
- PREFERLIST — authoritative domains given a score bonus so they bubble up
               in sort order before the URL cap is applied.
- Pure functions, no I/O, no LLM calls.

Public API
──────────
  filter_results(results)            → list[dict]  (blocked URLs removed)
  sort_by_preference(results)        → list[dict]  (preferred sources first)
  filter_and_sort(results, cap=15)   → list[dict]  (combined, capped)
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Blocklist ─────────────────────────────────────────────────────────────────
# Domains that reliably produce low-quality content for research purposes.
# Matched against the full netloc (partial match — "reddit" matches "old.reddit.com").

_BLOCKLIST_PATTERNS = re.compile(
    r"(reddit\.com|pinterest\.|linkedin\.com/in/|linkedin\.com/posts/"
    r"|twitter\.com|x\.com|instagram\.com|facebook\.com|tiktok\.com"
    r"|snapchat\.com|tumblr\.com|telegram\.org|t\.me|discord\."
    r"|quora\.com|yahoo\.answers|answers\.com"
    r"|yelp\.com|tripadvisor\.|booking\.com|airbnb\."
    r"|ebay\.com/itm|etsy\.com|shopify\."
    r"|patreon\.com|onlyfans\."
    r"|casino|poker|betting|slots|sportsbook"
    r"|spam|clickbait|buzzfeed\.com"
    r"|dailymail\.co\.uk|thesun\.co\.uk|nypost\.com"
    r"|breitbart\.|infowars\.|naturalnews\.)",
    re.IGNORECASE,
)

# URL path patterns that indicate low-value pages even on good domains
_BLOCKLIST_PATH_PATTERNS = re.compile(
    r"(/login|/signup|/register|/subscribe|/checkout|/cart"
    r"|/account|/password|/profile|/settings"
    r"|/ads?/|/advertisement|/sponsored"
    r"|/tag/|/category/$|/author/[^/]+/?$"
    r"|/search\?|/results\?"
    r"|\.zip$|\.exe$|\.dmg$|\.apk$)",
    re.IGNORECASE,
)

# ── Preferlist ────────────────────────────────────────────────────────────────
# Higher score → sorted closer to the top before the URL cap is applied.
# Score values are additive bonuses (not absolute scores).

_PREFER_TIERS: List[tuple[frozenset[str], int]] = [
    # Tier 1 — authoritative primary sources (+30)
    (frozenset({
        "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
        "apnews.com", "afp.com",
        "nature.com", "science.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com", "jstor.org", "sciencedirect.com",
        "nejm.org", "bmj.com", "thelancet.com",
        "nasa.gov", "nih.gov", "cdc.gov", "who.int", "europa.eu",
        "sec.gov", "fda.gov", "epa.gov",
        "marketwatch.com",
    }), 30),
    # Tier 2 — reputable tech / business / science media (+20)
    (frozenset({
        "techcrunch.com", "arstechnica.com", "theverge.com", "wired.com",
        "zdnet.com", "cnet.com", "engadget.com", "venturebeat.com",
        "infoq.com", "newscientist.com", "scientificamerican.com",
        "spacenews.com", "nasaspaceflight.com",
        "economist.com", "fortune.com", "businessinsider.com",
        "cnbc.com", "investopedia.com", "hbr.org", "mckinsey.com",
        "statista.com",
        "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com",
        "washingtonpost.com", "theatlantic.com", "axios.com", "npr.org",
        "wikipedia.org", "britannica.com",
        "github.com", "stackoverflow.com", "pytorch.org",
    }), 20),
    # Tier 3 — general quality media (+10)
    (frozenset({
        "medium.com", "substack.com", "forbes.com", "time.com",
        "newsweek.com", "theregister.com", "digitaltrends.com",
        "towardsdatascience.com", "hackernoon.com", "dev.to",
    }), 10),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_blocked(url: str) -> bool:
    if _BLOCKLIST_PATTERNS.search(url):
        return True
    path = urlparse(url).path
    if _BLOCKLIST_PATH_PATTERNS.search(path):
        return True
    return False


def _prefer_score(url: str) -> int:
    nl = _netloc(url)
    for domains, bonus in _PREFER_TIERS:
        if any(d in nl for d in domains):
            return bonus
    return 0


# ── Public API ────────────────────────────────────────────────────────────────

def filter_results(results: List[Dict]) -> List[Dict]:
    """
    Remove results whose URL matches the blocklist.

    Args:
        results: List of dicts with at least a 'url' key.

    Returns:
        Filtered list — blocked URLs removed.
    """
    kept, blocked = [], []
    for r in results:
        url = r.get("url", "")
        if _is_blocked(url):
            blocked.append(url)
            logger.debug(f"[SourceFilter] ⛔ blocked: {url}")
        else:
            kept.append(r)

    if blocked:
        logger.info(
            f"[SourceFilter] Blocklist: {len(kept)} kept / {len(blocked)} removed "
            f"({len(results)} total)"
        )
    return kept


def sort_by_preference(results: List[Dict]) -> List[Dict]:
    """
    Sort results so preferred (authoritative) domains appear first.
    Stable sort — preserves original order within the same tier.

    Args:
        results: List of dicts with at least a 'url' key.

    Returns:
        Sorted list — preferred sources first.
    """
    return sorted(results, key=lambda r: _prefer_score(r.get("url", "")), reverse=True)


def filter_and_sort(results: List[Dict], cap: int = 15) -> List[Dict]:
    """
    Apply blocklist filter → prefer-list sort → URL cap.

    This is the main entry point for the pipeline.

    Args:
        results: Raw search result dicts (title, url, snippet, ...).
        cap:     Maximum number of URLs to return after filtering (default 15).

    Returns:
        Filtered, sorted, capped list of result dicts.
    """
    filtered = filter_results(results)
    sorted_  = sort_by_preference(filtered)
    capped   = sorted_[:cap]
    logger.info(
        f"[SourceFilter] filter_and_sort: "
        f"{len(results)} in → {len(filtered)} after block → "
        f"{len(capped)} after cap({cap})"
    )
    return capped
