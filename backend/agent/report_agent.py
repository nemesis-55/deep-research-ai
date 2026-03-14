"""
Report Agent — Deep Research AI

Design pattern: Builder — assembles the final Markdown report incrementally.

Flow
────
  1. Group sources by research task
  2. Build per-section evidence from semantically ranked chunks
  3. Generate each section with Qwen writer (grounded-only, cite inline)
  4. Validate citations — strip hallucinated URLs
  5. Write Executive Summary + Conclusion
  6. Append validated sources section

Citation guarantee: every URL in the Sources section was actually scraped.
"""
import logging
from typing import Dict, List, Optional

from backend.model_loader import generate_text
from backend.tools.evidence_builder import (
    EvidenceResult,
    build_evidence,
    validate_citations,
    sources_section,
)
from backend.tools.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SECTION_PROMPT = """\
You are a world-class research analyst writing one section of a deep-dive report.

Report Topic: {query}
Section: {section_topic}

Research Evidence (use ONLY what is provided below — do not add external knowledge):
{evidence}

Write this section using the following structure:
### Major Developments
- Key events, announcements, and breakthroughs with specific dates and figures

### Industry Impact
- Effects on markets, organisations, and stakeholders; include data where present

### Expert Commentary
- Direct quotes or attributed views from named experts or institutions

### Strategic Implications
- Forward-looking analysis: risks, opportunities, and decisions decision-makers face

### Sources
- Inline citations as [Title](URL) using ONLY URLs that appear in the evidence above

Requirements:
- Start with ## heading matching the section topic
- Cite every claim inline as [Title](URL) using only URLs from the evidence
- Include specific facts, numbers, dates, and direct quotes where present
- Length: 400–700 words
- Do NOT invent URLs or cite sources not in the evidence"""

_EXEC_SUMMARY_PROMPT = """\
You are writing the Executive Summary of a major research report.

Topic: {query}

Section previews (already written — summarise these, do not add new facts):
{sections_preview}

Write a powerful Executive Summary (250–350 words):
- Open with the single most important finding
- Cover all key themes from the sections
- Include the most critical numbers or dates
- Use ## Executive Summary as the heading"""

_CONCLUSION_PROMPT = """\
You are writing the Conclusion of a major research report.

Topic: {query}

Key findings from the report:
{key_findings}

Write a concise Conclusion (150–250 words):
- Synthesise the most important insights
- Include forward-looking implications
- Do NOT introduce new facts not already in the key findings
- Use ## Conclusion as the heading"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReportAgent:
    """Builds a full structured research report from collected sources."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    # ── Private helpers ───────────────────────────────────────────────────────

    def _group_by_task(self, sources: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for s in sources:
            grouped.setdefault(s.get("task", "General Research"), []).append(s)
        return grouped

    def _collect_images(self, sources: List[Dict]) -> List[Dict]:
        seen: set = set()
        images: List[Dict] = []
        for s in sources:
            for img in s.get("images", []):
                url = img.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    images.append(img)
        return images[:20]

    def _pages_from_sources(self, task_sources: List[Dict]) -> List[Dict]:
        """Convert source dicts back into page dicts for evidence_builder."""
        pages = []
        for s in task_sources:
            text = s.get("analysis") or s.get("text", "")
            url  = s.get("url", "")
            if text and url and not url.startswith("synthesis://"):
                pages.append({
                    "url":   url,
                    "title": s.get("title", url),
                    "text":  text,
                })
        return pages

    def _build_image_gallery(self, images: List[Dict]) -> str:
        if not images:
            return ""
        lines = ["\n\n## 📷 Visual Evidence\n"]
        for img in images[:15]:
            url     = img.get("url", "")
            alt     = img.get("alt", "") or img.get("caption", "") or "Image"
            caption = img.get("caption", "") or alt
            if url:
                lines.append(f"![{alt}]({url})")
                if caption and caption != alt:
                    lines.append(f"*{caption}*\n")
        return "\n".join(lines)

    def _build_youtube_section(self, sources: List[Dict]) -> str:
        seen: set = set()
        embeds: List[Dict] = []
        for s in sources:
            for yt in s.get("youtube_embeds", []):
                url = yt.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    embeds.append(yt)
        if not embeds:
            return ""
        lines = ["\n\n## 🎥 Video Sources\n"]
        for yt in embeds[:5]:
            url        = yt.get("url", "")
            transcript = yt.get("transcript", "")
            lines.append(f"**[{url}]({url})**")
            if transcript:
                lines.append(f"\n> {transcript[:300]}…\n")
        return "\n".join(lines)

    # ── Director method ───────────────────────────────────────────────────────

    def generate_report(
        self,
        query:       str,
        all_sources: List[Dict],
        rag_context: str = "",
    ) -> str:
        logger.info(f"[Report] Generating for: {query} ({len(all_sources)} sources)")

        grouped    = self._group_by_task(all_sources)
        all_images = self._collect_images(all_sources)
        sections:  List[str] = []

        # Track all validated URLs so the final sources section is clean
        all_validated_sources: List[Dict] = []

        for task, task_sources in grouped.items():
            logger.info(f"[Report] Section: {task[:70]}")

            # ── Semantic evidence build for this section ───────────────────
            pages = self._pages_from_sources(task_sources)

            # Also splice in RAG context as a synthetic page (first section only)
            if rag_context and not sections:
                pages.insert(0, {
                    "url":   "rag://vector-store",
                    "title": "Vector Store RAG Context",
                    "text":  rag_context[:8000],
                })

            if not pages:
                continue

            ev: EvidenceResult = build_evidence(task, pages, top_k=20, max_chars=15_000)

            if not ev.context.strip():
                continue

            # Add this section's sources to the global validated list
            all_validated_sources.extend(ev.sources)

            # ── LLM section generation ─────────────────────────────────────
            # Budget: 600 tok @ ~10 tok/s (Qwen2.5-14B) ≈ 60 s per section
            # (was 800 tok @ 19.4 tok/s for 7B — same wall-clock, better quality)
            raw_section = generate_text(
                _SECTION_PROMPT.format(
                    query          = query,
                    section_topic  = task,
                    evidence       = ev.context,
                ),
                max_new_tokens=600, role="writer",
            )

            # ── Citation validation ────────────────────────────────────────
            # Only URLs actually crawled are permitted in the output
            allowed = ev.allowed_urls | frozenset({"rag://vector-store"})
            clean_section, hallucinated = validate_citations(raw_section, allowed, strict=False)
            if hallucinated:
                logger.warning(
                    f"[Report] Section '{task[:40]}': "
                    f"{len(hallucinated)} hallucinated URL(s) marked ⚠"
                )

            if clean_section.strip():
                sections.append(clean_section.strip())

        if not sections:
            return f"# Research Report: {query}\n\n⚠️ Insufficient data collected."

        # ── Executive summary ──────────────────────────────────────────────
        logger.info("[Report] Executive summary…")
        # Budget: 400 tok @ ~10 tok/s (Qwen2.5-14B) ≈ 40 s
        exec_summary = generate_text(
            _EXEC_SUMMARY_PROMPT.format(
                query            = query,
                sections_preview = "\n\n".join(s[:400] for s in sections[:6]),
            ),
            max_new_tokens=400, role="writer",
        )
        exec_summary, _ = validate_citations(
            exec_summary,
            frozenset(s["url"] for s in all_validated_sources),
            strict=True,
        )

        # ── Conclusion ─────────────────────────────────────────────────────
        logger.info("[Report] Conclusion…")
        # Budget: 300 tok @ ~10 tok/s (Qwen2.5-14B) ≈ 30 s
        conclusion = generate_text(
            _CONCLUSION_PROMPT.format(
                query        = query,
                key_findings = "\n".join(s[:200] for s in sections)[:4000],
            ),
            max_new_tokens=300, role="writer",
        )
        conclusion, _ = validate_citations(
            conclusion,
            frozenset(s["url"] for s in all_validated_sources),
            strict=True,
        )

        # ── Assemble final report ──────────────────────────────────────────
        # Deduplicate validated sources by URL
        seen_urls: set = set()
        deduped_sources: List[Dict] = []
        for s in all_validated_sources:
            url = s.get("url", "")
            if url and url not in seen_urls and not url.startswith(("rag://", "synthesis://")):
                seen_urls.add(url)
                deduped_sources.append(s)

        parts = [
            f"# 🔬 Deep Research Report: {query}\n",
            exec_summary.strip(),
            "\n---\n",
            "\n\n".join(sections),
            self._build_image_gallery(all_images),
            self._build_youtube_section(all_sources),
            sources_section(deduped_sources),
            conclusion.strip(),
        ]

        final = "\n\n".join(p for p in parts if p and p.strip())
        logger.info(f"[Report] Complete — {len(final):,} chars, {len(deduped_sources)} validated sources")
        return final
