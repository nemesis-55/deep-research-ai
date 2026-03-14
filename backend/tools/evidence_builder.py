"""
Evidence Builder & Citation Validator — Deep Research AI
=========================================================
Takes a list of scraped pages, runs the full chunking → semantic ranking
pipeline, then produces a citation-validated evidence block ready to inject
into the LLM report prompt.

Public API
──────────
  build_evidence(query, pages, top_k, max_chars) → EvidenceResult

EvidenceResult
──────────────
  .context      str          — formatted text block for the LLM prompt
  .sources      list[dict]   — validated source metadata (url, title, domain)
  .allowed_urls frozenset    — URLs that appear in crawled sources
                              (used downstream for citation validation)

Citation validation
───────────────────
  validate_citations(text, allowed_urls) rewrites the LLM output, removing
  any markdown links whose URL is NOT in the crawled source set, replacing
  them with [UNVERIFIED SOURCE] so hallucinated citations are visible rather
  than silently included.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvidenceResult:
    context:      str
    sources:      list[dict]           = field(default_factory=list)
    allowed_urls: frozenset[str]       = field(default_factory=frozenset)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_url(url: str) -> str:
    """Strip scheme variations and trailing slash for loose matching."""
    url = url.strip().rstrip("/").lower()
    url = re.sub(r"^https?://", "", url)
    url = url.removeprefix("www.")
    return url


def _domain(url: str) -> str:
    try:
        nl = urlparse(url).netloc.lower()
        return nl.removeprefix("www.")
    except Exception:
        return url


# ── Main builder ──────────────────────────────────────────────────────────────

def build_evidence(
    query:     str,
    pages:     list[dict],
    top_k:     int = 20,
    max_chars: int = 20_000,
) -> EvidenceResult:
    """
    Full pipeline: pages → chunks → semantic rank → evidence context.

    Args:
        query:     The user's research query (used for semantic similarity).
        pages:     List of scraped page dicts with keys: url, title, text.
        top_k:     Number of top-ranked chunks to include.
        max_chars: Hard char limit on combined context.

    Returns:
        EvidenceResult with .context, .sources, and .allowed_urls.
    """
    from backend.tools.document_chunks import (
        chunk_documents,
        rank_chunks,
        build_evidence_context,
    )

    if not pages:
        logger.warning("[Evidence] No pages provided — returning empty evidence")
        return EvidenceResult(context="", sources=[], allowed_urls=frozenset())

    # ── 1. Chunk all pages ────────────────────────────────────────────────
    chunks = chunk_documents(pages)

    if not chunks:
        logger.warning("[Evidence] No chunks produced — pages may be too short")
        return EvidenceResult(context="", sources=[], allowed_urls=frozenset())

    # ── 2. Semantic ranking ───────────────────────────────────────────────
    top_chunks = rank_chunks(query, chunks, top_k=top_k)

    # ── 3. Build evidence context string ─────────────────────────────────
    context = build_evidence_context(top_chunks, max_chars=max_chars)

    # ── 4. Collect validated source list ─────────────────────────────────
    seen:    set[str] = set()
    sources: list[dict] = []

    for chunk in top_chunks:
        url = chunk.url
        if url and url not in seen:
            seen.add(url)
            sources.append({
                "url":    url,
                "title":  chunk.title or url,
                "domain": chunk.domain or _domain(url),
                "score":  round(chunk.score, 4),
            })

    allowed_urls = frozenset(_normalise_url(s["url"]) for s in sources)

    logger.info(
        f"[Evidence] Built evidence: {len(top_chunks)} chunks, "
        f"{len(sources)} unique sources, "
        f"{len(context):,} chars"
    )

    return EvidenceResult(
        context      = context,
        sources      = sources,
        allowed_urls = allowed_urls,
    )


# ── Citation validator ────────────────────────────────────────────────────────

_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')


def validate_citations(
    text:         str,
    allowed_urls: frozenset[str],
    strict:       bool = False,
) -> tuple[str, list[str]]:
    """
    Scan `text` for markdown links [Title](URL) and validate each URL
    against `allowed_urls` (the set of actually-crawled source URLs).

    Args:
        text:         LLM-generated report text.
        allowed_urls: frozenset of normalised URLs from EvidenceResult.
        strict:       If True, replace unverified links with [UNVERIFIED].
                      If False (default), keep them but append a ⚠ marker.

    Returns:
        (cleaned_text, list_of_hallucinated_urls)
    """
    hallucinated: list[str] = []

    def _replace(m: re.Match) -> str:
        label = m.group(1)
        url   = m.group(2)
        norm  = _normalise_url(url)

        # Partial match: any allowed URL that contains the normalised form
        # handles sub-page links (e.g. reuters.com/article/xyz vs reuters.com)
        verified = any(
            norm in au or au in norm
            for au in allowed_urls
        )

        if verified:
            return m.group(0)   # keep as-is

        hallucinated.append(url)
        logger.debug(f"[Citation] ⚠ Unverified URL: {url}")

        if strict:
            return f"[{label}][UNVERIFIED SOURCE]"
        else:
            return f"[{label}]({url})⚠"

    cleaned = _MD_LINK_RE.sub(_replace, text)

    if hallucinated:
        logger.warning(
            f"[Citation] {len(hallucinated)} unverified citation(s) found: "
            + ", ".join(hallucinated[:5])
        )

    return cleaned, hallucinated


def sources_section(sources: list[dict]) -> str:
    """
    Build a Markdown sources section from validated source dicts.
    Only includes real crawled URLs — no hallucinations possible.
    """
    if not sources:
        return ""

    lines = ["\n\n## 📚 Sources & References\n"]
    seen: set[str] = set()
    i = 1
    for s in sources:
        url   = s.get("url", "")
        title = s.get("title", url)
        if url and url not in seen:
            seen.add(url)
            lines.append(f"{i}. [{title}]({url})")
            i += 1

    return "\n".join(lines)
