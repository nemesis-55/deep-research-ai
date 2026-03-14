"""
Query Planner — Deep Research AI

Design pattern: Strategy — generates a diverse set of focused search queries
from a high-level research topic using the planner LLM (DeepSeek-R1).
Falls back to a deterministic template expansion if LLM output cannot be parsed.

This module is called during Phase 0 of the pipeline (deep crawl pre-population)
BEFORE the main planner generates the per-task research plan.  Its queries are
broader and more varied than the per-task queries, ensuring maximum URL coverage.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from backend.config_loader import get
from backend.model_loader import generate_text

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_QUERY_PROMPT = """\
You are an expert research librarian generating a diverse set of web search queries.

Topic: {topic}
Number of queries needed: {n}

Generate exactly {n} highly specific, non-overlapping search queries that together
cover the full breadth of this topic.  Vary the angle of each query:
  - factual overview / definition
  - recent news and events (2024-2025)
  - statistical data and metrics
  - expert opinions and analysis
  - comparisons and alternatives
  - historical context and timeline
  - technical details / how it works
  - business / financial angle
  - controversies or criticisms
  - future outlook and predictions

Return ONLY valid JSON — no text before or after:
{{
  "queries": [
    "specific search query 1",
    "specific search query 2"
  ]
}}

Each query must be 5-12 words and immediately usable as a DuckDuckGo search string."""

# ── Fallback template expansion ───────────────────────────────────────────────

_TEMPLATES = [
    "{topic}",
    "{topic} overview explained",
    "{topic} latest news 2025",
    "{topic} statistics data metrics",
    "{topic} expert analysis opinion",
    "{topic} history background origin",
    "{topic} how it works technical",
    "{topic} pros cons criticism",
    "{topic} future trends predictions",
    "{topic} comparison alternatives",
    "{topic} financial business impact",
    "{topic} key people leaders founders",
    "{topic} research findings study",
    "{topic} real world examples cases",
    "{topic} regulations policy legal",
]


class QueryPlanner:
    """
    Generates N focused search queries for a research topic.

    Call generate_queries(topic) to get a list of strings.
    The planner LLM must already be loaded by the caller.
    """

    def generate_queries(self, topic: str, n: int = None) -> List[str]:
        """Return a list of `n` search query strings for `topic`."""
        n = n or get("deep_crawl.num_queries", 10)
        logger.info(f"[QueryPlanner] Generating {n} queries for: {topic[:80]}")

        try:
            raw  = generate_text(
                _QUERY_PROMPT.format(topic=topic, n=n),
                max_new_tokens=512,
                role="planner",
            )
            queries = self._parse(raw, n)
            if queries:
                logger.info(f"[QueryPlanner] LLM produced {len(queries)} queries.")
                return queries
        except Exception as e:
            logger.warning(f"[QueryPlanner] LLM failed ({e}), using fallback.")

        return self._fallback(topic, n)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, raw: str, n: int) -> List[str]:
        """Try to extract a list of query strings from raw LLM output."""
        # 1. JSON extraction
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data    = json.loads(raw[start:end])
                queries = data.get("queries", [])
                if isinstance(queries, list) and len(queries) >= 3:
                    cleaned = [str(q).strip().strip('"') for q in queries if str(q).strip()]
                    return cleaned[:n]
        except (json.JSONDecodeError, Exception):
            pass

        # 2. Line-by-line fallback: numbered / bulleted lists
        lines: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            line = re.sub(r'^[\d]+[.)]\s*', '', line)   # "1. " or "1) "
            line = re.sub(r'^[-*•]\s*',     '', line)   # bullet
            line = line.strip('"').strip("'").strip()
            if 10 < len(line) < 200:
                lines.append(line)
        if len(lines) >= 3:
            return lines[:n]

        return []

    def _fallback(self, topic: str, n: int) -> List[str]:
        """Deterministic template expansion — always succeeds."""
        templates = _TEMPLATES[:n]
        # If we need more than the templates list, cycle through with index suffix
        while len(templates) < n:
            templates.append(f"{topic} part {len(templates) + 1}")
        queries = [t.format(topic=topic) for t in templates[:n]]
        logger.info(f"[QueryPlanner] Fallback produced {len(queries)} queries.")
        return queries
