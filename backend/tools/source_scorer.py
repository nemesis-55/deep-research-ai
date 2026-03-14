"""
Source Credibility Scorer — Deep Research AI

Scores each source 0–100 based on:
  - Domain authority (known high-quality domains → bonus)
  - Recency signal  (URL/snippet contains recent year → bonus)
  - Content length  (longer = more substantive)
  - Cross-source repetition (same URL cited multiple times → bonus)

Only sources with score >= config research.source_min_score (default 50)
are used in the final report.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse

from backend.config_loader import get

logger = logging.getLogger(__name__)

# High-authority domain bonuses — now delegated to url_filter tiers.
# Kept as a thin shim so any direct callers of score_source() still work.
_HIGH_AUTHORITY: dict = {}   # replaced by domain_score_bonus() from url_filter

def _recent_years() -> frozenset:
    """Compute current + previous year at call time (not import time)."""
    y = datetime.now().year
    return frozenset({str(y), str(y - 1)})


def score_source(
    url:     str,
    snippet: str = "",
    text:    str = "",
) -> int:
    """Return a credibility score 0–100 for a single source."""
    score = 30   # base

    # ── Domain authority (via url_filter tier system) ─────────────────────
    try:
        from backend.tools.url_filter import domain_score_bonus
        score += domain_score_bonus(url)
    except Exception:
        pass

    # ── Recency ───────────────────────────────────────────────────────────
    combined = f"{url} {snippet}"
    if any(yr in combined for yr in _recent_years()):
        score += 15

    # ── Content length ────────────────────────────────────────────────────
    length = len(text) + len(snippet)
    if length > 5000:
        score += 15
    elif length > 1000:
        score += 8
    elif length > 200:
        score += 3

    return min(score, 100)


def score_and_filter(sources: List[Dict], min_score: int = None) -> List[Dict]:
    """
    Score all sources, attach 'credibility_score', sort descending,
    drop sources below min_score.
    """
    threshold = min_score if min_score is not None else get("research.source_min_score", 50)
    url_counts: Dict[str, int] = {}
    for s in sources:
        url_counts[s.get("url", "")] = url_counts.get(s.get("url", ""), 0) + 1

    scored = []
    for s in sources:
        url  = s.get("url", "")
        base = score_source(url, s.get("snippet", ""), s.get("text", "") or s.get("analysis", ""))
        # Repetition bonus — same URL appearing in multiple tasks
        repetition = min((url_counts[url] - 1) * 5, 15)
        final      = min(base + repetition, 100)
        s["credibility_score"] = final
        if final >= threshold:
            scored.append(s)

    scored.sort(key=lambda x: x["credibility_score"], reverse=True)
    dropped = len(sources) - len(scored)
    if dropped:
        logger.info(f"[Credibility] Dropped {dropped} sources below score {threshold}.")
    return scored
