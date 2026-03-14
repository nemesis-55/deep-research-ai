"""
Report Agent — Deep Research AI

Design pattern: Builder — assembles the final Markdown report incrementally:
  executive summary → per-task sections → image gallery → YouTube section
  → source references → conclusion.
  Each build step is a private _build_* method; generate_report() is the Director.
Uses the Qwen Writer model for all LLM calls.
"""
import logging
from typing import Dict, List

from backend.model_loader import generate_text
from backend.tools.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SECTION_PROMPT = """\
You are a world-class research analyst writing one section of a deep-dive report.

Report Topic: {query}
This Section: {section_topic}

Research Evidence:
{evidence}

Write a thorough section (500-800 words) with:
- ## heading matching the section topic
- ### sub-headings where appropriate
- Specific facts, numbers, quotes from the evidence
- Inline citations as [Title](URL)
- Bullet lists for key data points

Do NOT write an intro or conclusion. Use ONLY the evidence provided."""

_SECTION_WITH_RAG_PROMPT = """\
You are a world-class research analyst writing one section of a deep-dive report.

Report Topic: {query}
This Section: {section_topic}

Background Context (from deep web crawl — use as supporting evidence):
{rag_context}

Primary Research Evidence:
{evidence}

Write a thorough section (500-800 words) with:
- ## heading matching the section topic
- ### sub-headings where appropriate
- Specific facts, numbers, quotes from both the background context and primary evidence
- Inline citations as [Title](URL)
- Bullet lists for key data points

Prioritise the Primary Research Evidence for structure; use Background Context to add
depth, statistics, and corroborating details. Use ONLY provided evidence."""

_EXEC_SUMMARY_PROMPT = """\
You are writing the Executive Summary of a major research report.

Query: {query}

Section previews:
{sections_preview}

Write a powerful Executive Summary (300-400 words):
- Open with the single most important finding
- Cover all key themes
- Highlight critical data points

Use ## Executive Summary as the heading."""

_CONCLUSION_PROMPT = """\
You are writing the Conclusion of a major research report.

Query: {query}

Key Findings:
{key_findings}

Write a strong Conclusion (200-300 words):
- Synthesise the most important insights
- Forward-looking implications
- No new facts

Use ## Conclusion as the heading."""


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReportAgent:
    """Builds a full structured research report from collected sources."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    # ── Private builders ──────────────────────────────────────────────────────

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
        return images[:30]

    def _build_evidence(self, task_sources: List[Dict]) -> str:
        parts = []
        for s in task_sources[:5]:
            text  = s.get("analysis") or s.get("text", "")
            url   = s.get("url", "")
            title = s.get("title", url)
            if text:
                parts.append(f"### {title}\nURL: {url}\n\n{text[:2500]}")
        return "\n\n---\n\n".join(parts)[:8000]

    def _build_image_gallery(self, images: List[Dict]) -> str:
        if not images:
            return ""
        lines = ["\n\n## 📷 Visual Evidence\n",
                 "> All images sourced from research references.\n"]
        for img in images[:20]:
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
        for yt in embeds[:8]:
            url        = yt.get("url", "")
            transcript = yt.get("transcript", "")
            lines.append(f"**[{url}]({url})**")
            if transcript:
                lines.append(f"\n> {transcript[:400]}…\n")
        return "\n".join(lines)

    def _build_sources_section(self, sources: List[Dict]) -> str:
        seen: set = set()
        lines = ["\n\n## 📚 Sources & References\n"]
        i = 1
        for s in sources:
            url   = s.get("url", "")
            title = s.get("title", url)
            if url and url not in seen:
                seen.add(url)
                lines.append(f"{i}. [{title}]({url})")
                i += 1
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

        # Trim RAG context to fit comfortably in the prompt without crowding evidence
        rag_snippet = rag_context[:3000] if rag_context else ""

        for task, task_sources in grouped.items():
            logger.info(f"[Report] Section: {task[:70]}")
            evidence = self._build_evidence(task_sources)
            if not evidence.strip():
                continue

            if rag_snippet:
                prompt = _SECTION_WITH_RAG_PROMPT.format(
                    query        = query,
                    section_topic= task,
                    rag_context  = rag_snippet,
                    evidence     = evidence,
                )
            else:
                prompt = _SECTION_PROMPT.format(
                    query        = query,
                    section_topic= task,
                    evidence     = evidence,
                )

            section = generate_text(prompt, max_new_tokens=1200, role="writer")
            if section.strip():
                sections.append(section.strip())

        if not sections:
            return f"# Research Report: {query}\n\n⚠️ Insufficient data collected."

        # Executive summary
        logger.info("[Report] Executive summary…")
        exec_summary = generate_text(
            _EXEC_SUMMARY_PROMPT.format(
                query=query,
                sections_preview="\n\n".join(s[:500] for s in sections[:6]),
            ),
            max_new_tokens=700, role="writer",
        )

        # Conclusion
        logger.info("[Report] Conclusion…")
        conclusion = generate_text(
            _CONCLUSION_PROMPT.format(
                query=query,
                key_findings="\n".join(s[:250] for s in sections)[:5000],
            ),
            max_new_tokens=500, role="writer",
        )

        parts = [
            f"# 🔬 Deep Research Report: {query}\n",
            exec_summary.strip(),
            "\n---\n",
            "\n\n".join(sections),
            self._build_image_gallery(all_images),
            self._build_youtube_section(all_sources),
            self._build_sources_section(all_sources),
            conclusion.strip(),
        ]

        final = "\n\n".join(p for p in parts if p and p.strip())
        logger.info(f"[Report] Complete — {len(final):,} chars")
        return final
