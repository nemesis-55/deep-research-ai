"""
Research Agent — Deep Research AI

Template Method: research_task() defines the pipeline skeleton:
  search → scrape → score → analyse → store → extract entities → KG
"""
import logging
from typing import Callable, Dict, List, Optional

from backend.config_loader import get
from backend.model_loader import generate_text
from backend.tools.knowledge_graph import KnowledgeGraph, extract_entities_and_relations
from backend.tools.page_scraper import scrape_page
from backend.tools.source_scorer import score_source
from backend.tools.vector_store import VectorStore
from backend.tools.web_search import search_web

logger = logging.getLogger(__name__)

_CHUNK_PROMPT = """\
You are a senior research analyst extracting facts from a document chunk.

Research Task: {task}
Source: {url}
Chunk {chunk_num} of {total_chunks}:

{content}

Extract EVERY relevant fact, figure, quote, date, and data point.
Format as dense, specific bullet points. Include exact numbers, quotes, events."""

_SYNTHESIS_PROMPT = """\
You are a senior research analyst synthesizing chunk analyses.

Research Task: {task}
Source: {title} ({url})

Chunk Analyses:
{analyses}

Write a thorough synthesis (300-500 words) covering ALL key insights. Do not omit details."""

_CRITIQUE_PROMPT = """\
You are a critical research editor.

Research Task: {task}

Current Summary:
{summary}

Identify gaps, then write an improved summary.

=== IMPROVED SUMMARY ===
"""


class ResearchAgent:
    def __init__(self, vector_store: VectorStore, knowledge_graph: Optional[KnowledgeGraph] = None) -> None:
        self.vector_store   = vector_store
        self.kg             = knowledge_graph
        self.chunk_size     = get("research.chunk_size", 1000)
        self.chunk_overlap  = get("research.chunk_overlap", 150)
        self.max_pages      = get("research.max_pages_per_task", 5)
        self.max_results    = get("research.max_results_per_task", 8)
        self.follow_links   = get("research.follow_links", True)
        self.max_follow     = get("research.max_follow_links", 3)
        self.research_loops = get("research.research_loops", 3)
        self.min_score      = get("research.source_min_score", 50)

    def _chunk(self, text: str) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start : start + self.chunk_size])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _analyse_source(self, task: str, url: str, title: str, text: str) -> str:
        text   = text[: get("research.max_content_chars", 30000)]
        chunks = self._chunk(text)
        total  = len(chunks)
        analyses = []

        logger.debug(
            f"[ResearchAgent] _analyse_source  chunks={total}  "
            f"text_len={len(text):,}  url={url[:70]}"
        )

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"[ResearchAgent]   📝 Chunk {i}/{total}  ({len(chunk):,} chars) ← {url[:60]}")
            out = generate_text(
                _CHUNK_PROMPT.format(task=task, url=url, chunk_num=i, total_chunks=total, content=chunk),
                max_new_tokens=600, role="writer",
            )
            logger.debug(f"[ResearchAgent]   Chunk {i} output: {len(out):,} chars")
            analyses.append(out)

        if len(analyses) == 1:
            logger.debug(f"[ResearchAgent]   Single chunk — returning directly")
            return analyses[0]

        logger.info(f"[ResearchAgent]   🔀 Synthesising {len(analyses)} chunk analyses for {url[:60]}")
        synthesis = generate_text(
            _SYNTHESIS_PROMPT.format(
                task=task, title=title, url=url,
                analyses="\n\n---\n\n".join(analyses),
            ),
            max_new_tokens=1000, role="writer",
        )
        logger.debug(f"[ResearchAgent]   Synthesis output: {len(synthesis):,} chars")
        return synthesis

    def _self_critique(self, task: str, summary: str) -> str:
        result = generate_text(
            _CRITIQUE_PROMPT.format(task=task, summary=summary),
            max_new_tokens=1200, role="writer",
        )
        marker = "=== IMPROVED SUMMARY ==="
        if marker in result:
            improved = result.split(marker, 1)[1].strip()
            return improved if len(improved) > 100 else summary
        return result.strip() or summary

    def _store(self, task: str, url: str, title: str, text: str) -> None:
        self.vector_store.store_document(
            text=f"Task: {task}\nSource: {title}\nURL: {url}\n\n{text[:2000]}",
            metadata={"url": url, "title": title, "task": task},
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def research_task(
        self,
        task: str,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict]:
        """Full deep research for one task. Returns list of source dicts."""
        sources: List[Dict] = []

        def _status(msg: str) -> None:
            if status_callback:
                status_callback(msg)
            logger.info(f"[Research] {msg}")

        logger.info(f"[ResearchAgent] ══ research_task START ══════════════════════════════")
        logger.info(f"[ResearchAgent]   Task          : {task}")
        logger.info(f"[ResearchAgent]   max_pages     : {self.max_pages}")
        logger.info(f"[ResearchAgent]   max_results   : {self.max_results}")
        logger.info(f"[ResearchAgent]   research_loops: {self.research_loops}")
        logger.info(f"[ResearchAgent]   follow_links  : {self.follow_links}  max_follow={self.max_follow}")

        _status(f"🔍 Searching: {task[:80]}")
        results = search_web(task, max_results=self.max_results)

        logger.info(f"[ResearchAgent] Search returned {len(results)} results for: {task[:70]}")
        for idx, r in enumerate(results, 1):
            logger.info(f"  #{idx:>2}  {r.get('url', '(no url)')}  —  {r.get('title','')[:60]}")

        if not results:
            _status(f"⚠️ No results for: {task[:60]}")
            return sources

        for page_idx, result in enumerate(results[: self.max_pages], 1):
            url   = result.get("url", "")
            title = result.get("title", url)
            if not url:
                continue

            logger.info(
                f"[ResearchAgent] ── Page {page_idx}/{min(self.max_pages, len(results))} ──"
            )
            logger.info(f"[ResearchAgent]   URL   : {url}")
            logger.info(f"[ResearchAgent]   Title : {title[:80]}")

            _status(f"  📄 Scraping [{page_idx}/{min(self.max_pages, len(results))}]: {title[:60]}  🔗 {url[:70]}")
            try:
                page = scrape_page(
                    url,
                    follow_links=self.follow_links,
                    follow_links_depth=1,
                    max_follow_links=self.max_follow,
                )
            except Exception as e:
                logger.debug(f"[ResearchAgent]   ❌ Scrape error: {e}")
                _status(f"  ⚠️ Scrape failed {url[:50]}: {e}")
                continue

            text = page.get("text", "")
            logger.info(
                f"[ResearchAgent]   Scraped  : {len(text):,} chars  "
                f"followed={len(page.get('followed_sources', []))} links"
            )
            if not text or len(text) < 150:
                logger.debug(f"[ResearchAgent]   ⏭ SKIP (too short: {len(text)} chars)")
                continue

            _status(f"  🧠 Analysing: {title[:60]} ({len(text):,} chars)")
            analysis   = self._analyse_source(task, url, title, text)
            all_images = list(page.get("images", []))

            followed_sources = page.get("followed_sources", [])
            logger.debug(f"[ResearchAgent]   Followed links scraped: {len(followed_sources)}")
            for fi, followed in enumerate(followed_sources, 1):
                furl  = followed.get("url", "")
                ftext = followed.get("text", "")
                logger.debug(
                    f"[ResearchAgent]     Follow #{fi}  chars={len(ftext):,}  url={furl[:60]}"
                )
                all_images.extend(followed.get("images", []))
                if ftext and len(ftext) > 150:
                    self._store(task, furl, furl, ftext)

            self._store(task, url, title, analysis or text)

            # ── Entity extraction → Knowledge Graph ───────────────────────
            if self.kg is not None:
                extract_entities_and_relations(
                    text       = (analysis or text)[:4000],
                    source_url = url,
                    kg         = self.kg,
                )

            # ── Credibility score (early, per-source) ─────────────────────
            cred = score_source(url, result.get("snippet", ""), text)
            logger.info(
                f"[ResearchAgent]   ✅ Page {page_idx} done  "
                f"cred={cred}  analysis={len(analysis or ''):,} chars"
            )

            sources.append({
                "task":             task,
                "url":              url,
                "title":            title,
                "snippet":          result.get("snippet", ""),
                "text":             text[:3000],
                "analysis":         analysis,
                "images":           all_images[:8],
                "youtube_embeds":   page.get("youtube_embeds", []),
                "credibility_score": cred,
            })

        # Self-critique pass — improved synthesis stored as a standalone entry
        logger.info(
            f"[ResearchAgent] ── Self-critique  "
            f"loops={self.research_loops}  sources_before={len(sources)} ──"
        )
        if self.research_loops > 1 and sources:
            _status(f"  🔄 Self-critique: {task[:60]}")
            combined = "\n\n".join(
                s.get("analysis") or s.get("text", "")[:1000]
                for s in sources
                if s.get("analysis") or s.get("text")
            )
            if combined:
                logger.debug(f"[ResearchAgent]   Combined input for critique: {len(combined):,} chars")
                improved = self._self_critique(task, combined[:8000])
                logger.debug(f"[ResearchAgent]   Critique output            : {len(improved):,} chars")
                # Attach the critique as a synthetic source so the report
                # agent can use it without losing the original source entries
                sources.append({
                    "task":             task,
                    "url":              f"synthesis://{task[:40]}",
                    "title":            f"[Synthesis] {task[:70]}",
                    "snippet":          "",
                    "text":             improved,
                    "analysis":         improved,
                    "images":           [],
                    "youtube_embeds":   [],
                    "credibility_score": 75,
                })

        logger.info(
            f"[ResearchAgent] ══ research_task END  "
            f"sources={len(sources)} ══════════════════════════════"
        )
        _status(f"✅ Done: {task[:60]} — {len(sources)} sources")
        return sources
