"""
Query Generator — Deep Research AI
====================================
Converts a user research task/prompt into 5 concise, specific search queries
using the DeepSeek-R1 planner LLM.  Falls back to deterministic expansion if
LLM output cannot be parsed.

Design
──────
- Uses the *already-loaded* planner model (caller's responsibility).
- Returns exactly 5 queries: 3–8 words each, no instruction language.
- Output is pure search strings — no "Find out about …" or "Explain …" phrasing.
- Falls back to rule-based templates without crashing if the LLM fails.

Public API
──────────
  generate_queries(prompt, n=5) → List[str]
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_QUERY_PROMPT = """\
Convert the research topic below into {n} focused web search queries.

Research topic: {prompt}

Rules:
- Each query must be 3–8 words
- No instruction verbs (no "find", "explain", "describe", "what is", "how to")
- Vary the angle: facts, news, data, analysis, comparison
- Return ONLY valid JSON, no other text

Output format:
{{"queries": ["query one", "query two", "query three", "query four", "query five"]}}"""

# ── Fallback templates ────────────────────────────────────────────────────────

_TEMPLATES = [
    "{base}",
    "{base} {year}",
    "{base} analysis report",
    "{base} latest developments",
    "{base} statistics data",
    "{base} expert review",
    "{base} trends outlook",
    "{base} key findings",
]

_MIN_WORDS = 3
_MAX_WORDS = 8


def _clean_query(q: str) -> str:
    """Strip instruction language and normalise whitespace."""
    q = q.strip().strip('"').strip("'").rstrip("?").strip()
    # Remove leading instruction verbs
    q = re.sub(
        r"^(find|search|look up|explain|describe|list|what is|what are"
        r"|how to|how does|give me|tell me|show me|provide)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    return q


def _valid(q: str) -> bool:
    words = q.split()
    return _MIN_WORDS <= len(words) <= _MAX_WORDS and len(q) > 5


def _parse_llm_output(raw: str, n: int) -> List[str]:
    """Try JSON extraction then line-by-line fallback."""
    # 1. JSON extraction
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            qs   = data.get("queries", [])
            if isinstance(qs, list):
                cleaned = [_clean_query(str(q)) for q in qs if str(q).strip()]
                valid   = [q for q in cleaned if _valid(q)]
                if len(valid) >= 3:
                    return valid[:n]
    except Exception:
        pass

    # 2. Line-by-line: numbered / bulleted lists
    lines: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*•]\s*', '', line)
        q = _clean_query(line)
        if _valid(q):
            lines.append(q)
    if len(lines) >= 3:
        return lines[:n]

    return []


def _fallback(prompt: str, n: int) -> List[str]:
    """Deterministic template expansion — always succeeds."""
    import datetime
    base = prompt.strip().rstrip("?").strip()
    # Truncate to ≤6 words so templates stay within the 8-word cap
    base_words = base.split()[:6]
    base = " ".join(base_words)
    year = str(datetime.date.today().year)

    queries: List[str] = []
    for tmpl in _TEMPLATES:
        q = tmpl.format(base=base, year=year).strip()
        if _valid(q) and q not in queries:
            queries.append(q)
        if len(queries) >= n:
            break

    # Pad if needed (shouldn't happen with 8 templates and n≤5)
    while len(queries) < n:
        queries.append(f"{base} overview")

    logger.info(f"[QueryGen] Fallback produced {len(queries)} queries")
    return queries[:n]


def generate_queries(prompt: str, n: int = 5) -> List[str]:
    """
    Convert `prompt` into `n` focused search queries.

    The planner model must already be loaded before calling this function.
    Falls back to deterministic templates on any LLM/parse failure.

    Args:
        prompt: User research topic or task description.
        n:      Number of queries to generate (default 5).

    Returns:
        List of `n` search query strings, 3–8 words each.
    """
    logger.info(f"[QueryGen] Generating {n} queries for: {prompt[:80]}")

    try:
        from backend.model_loader import generate_text

        raw = generate_text(
            _QUERY_PROMPT.format(prompt=prompt, n=n),
            max_new_tokens=256,
            role="planner",
        )
        queries = _parse_llm_output(raw, n)

        if queries:
            logger.info(
                f"[QueryGen] LLM produced {len(queries)} queries: "
                + " | ".join(f'"{q}"' for q in queries)
            )
            return queries

        logger.warning("[QueryGen] LLM parse failed — using fallback")

    except Exception as e:
        logger.warning(f"[QueryGen] LLM call failed ({e}) — using fallback")

    return _fallback(prompt, n)
