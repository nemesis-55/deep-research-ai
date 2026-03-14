"""
URL Domain Filter — Deep Research AI
======================================
Classifies URLs as ALLOWED or REJECTED based on domain credibility tiers.

Design: pure functions, no I/O, no imports from other backend modules.
Called by ArticleScraper (deep research) and _chat_web_search (chat mode).

Tier system
───────────
  TIER_1  — authoritative primary sources (journals, government, wire services)
  TIER_2  — reputable tech / business / science media
  TIER_3  — general quality media (allowed but scored lower)
  BLOCKED — social platforms, forums, paywalled noise, spam sinks

Any domain not in a tier is NEUTRAL — allowed but gets no bonus.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Allowed domain tiers (partial-match on netloc) ────────────────────────────

TIER_1: frozenset[str] = frozenset({
    # Science / Academic
    "nature.com", "science.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com", "jstor.org", "sciencedirect.com", "springer.com",
    "cell.com", "nejm.org", "bmj.com", "thelancet.com",
    # Government / official
    "nasa.gov", "nih.gov", "cdc.gov", "who.int", "europa.eu",
    "congress.gov", "sec.gov", "fda.gov", "energy.gov",
    "esa.int", "faa.gov", "epa.gov",
    # Financial wire / primary data
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "apnews.com", "afp.com", "marketwatch.com",
})

TIER_2: frozenset[str] = frozenset({
    # Tech media
    "techcrunch.com", "arstechnica.com", "theverge.com", "wired.com",
    "9to5mac.com", "macrumors.com", "zdnet.com", "cnet.com",
    "engadget.com", "gizmodo.com", "venturebeat.com", "thenextweb.com",
    "infoq.com", "slashdot.org", "hackaday.com",
    # Space / science specialty
    "spacenews.com", "nasaspaceflight.com", "spaceflightnow.com",
    "planetary.org", "space.com", "skyandtelescope.org",
    "newscientist.com", "scientificamerican.com", "popularmechanics.com",
    # Business / economy
    "economist.com", "fortune.com", "businessinsider.com",
    "cnbc.com", "thestreet.com", "investopedia.com",
    "hbr.org", "mckinsey.com", "statista.com",
    # General news (quality)
    "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "theatlantic.com", "politico.com",
    "axios.com", "vox.com", "npr.org", "pbs.org",
    # Reference
    "wikipedia.org", "britannica.com",
    # Developer
    "github.com", "stackoverflow.com", "docs.python.org",
    "developer.mozilla.org", "learn.microsoft.com",
    "cloud.google.com", "aws.amazon.com", "pytorch.org",
})

TIER_3: frozenset[str] = frozenset({
    "medium.com", "substack.com", "forbes.com", "time.com",
    "newsweek.com", "usatoday.com", "latimes.com", "sfgate.com",
    "theregister.com", "slashgear.com", "digitaltrends.com",
    "towardsdatascience.com", "hackernoon.com", "dev.to",
})

# ── Hard-blocked domain fragments ─────────────────────────────────────────────
_BLOCKED_RE = re.compile(
    r"(reddit\.com|x\.com|twitter\.com|instagram\.com|facebook\.com"
    r"|tiktok\.com|snapchat\.com|pinterest\.com|tumblr\.com"
    r"|telegram\.org|t\.me|discord\.com|discord\.gg"
    r"|linkedin\.com/in/|linkedin\.com/posts/"
    r"|quora\.com|yahoo\.answers|answers\.com"
    r"|yelp\.com|tripadvisor\.com|booking\.com|airbnb\.com"
    r"|ebay\.com/itm|etsy\.com|shopify\.com"
    r"|patreon\.com|onlyfans\.com"
    r"|casino|poker|betting|slots|sportsbook)",
    re.IGNORECASE,
)

# Paths that reliably indicate low-value pages even on good domains
_BLOCKED_PATH_RE = re.compile(
    r"(/login|/signup|/register|/subscribe|/checkout|/cart"
    r"|/account|/password|/profile|/settings"
    r"|/ads?/|/advertisement|/sponsored"
    r"|\.zip$|\.exe$|\.dmg$|\.apk$"
    r"|/tag/|/category/|/author/[^/]+/?$"
    r"|/search\?|/results\?)",
    re.IGNORECASE,
)


def _netloc(url: str) -> str:
    """Return cleaned netloc (strip www.)."""
    try:
        nl = urlparse(url).netloc.lower()
        return nl.removeprefix("www.")
    except Exception:
        return ""


def get_domain_tier(url: str) -> int:
    """
    Return domain tier: 1 (best) / 2 / 3 / 0 (neutral) / -1 (blocked).
    """
    nl = _netloc(url)
    if not nl:
        return -1

    if _BLOCKED_RE.search(url):
        return -1

    path = urlparse(url).path
    if _BLOCKED_PATH_RE.search(path):
        return -1

    for domain in TIER_1:
        if domain in nl:
            return 1
    for domain in TIER_2:
        if domain in nl:
            return 2
    for domain in TIER_3:
        if domain in nl:
            return 3

    return 0   # neutral — not blocked, not known-good


def is_allowed(url: str) -> bool:
    """Return True if the URL passes the domain filter."""
    return get_domain_tier(url) >= 0


def domain_score_bonus(url: str) -> int:
    """
    Return a credibility bonus (0–30) to add on top of source_scorer.
    Tier 1 → +30, Tier 2 → +20, Tier 3 → +10, neutral → 0.
    """
    tier = get_domain_tier(url)
    return {1: 30, 2: 20, 3: 10, 0: 0, -1: 0}.get(tier, 0)


def filter_urls(urls: list[str], log_prefix: str = "[URLFilter]") -> list[str]:
    """Filter a list of URL strings. Returns only allowed URLs, preserving order."""
    kept:    list[str] = []
    dropped: list[str] = []

    for url in urls:
        if is_allowed(url):
            kept.append(url)
        else:
            dropped.append(url)
            logger.debug(f"{log_prefix} ⛔ blocked: {url}")

    if dropped:
        logger.info(
            f"{log_prefix} domain filter: "
            f"{len(kept)} kept / {len(dropped)} blocked "
            f"({len(urls)} total)"
        )

    return kept


def filter_results(
    results: list[dict],
    log_prefix: str = "[URLFilter]",
) -> list[dict]:
    """
    Filter a list of search-result dicts (must have a 'url' key).
    Returns only dicts whose URL is allowed.
    """
    return [r for r in results if is_allowed(r.get("url", ""))]
