"""
Research Agent — Deep Research AI

Flow per task:
  1. Generate 3 focused search queries
  2. Parallel multi-query DDG search
  3. Domain filter (url_filter)
  4. Parallel HTTP scrape (I/O-bound threads)
  5. Semantic chunk ranking + evidence assembly (build_evidence)
  6. Single LLM analysis call over top-ranked evidence
  7. Return validated source list
"""
import logging
from typing import Callable, Dict, List, Optional

from backend.config_loader import get
from backend.model_loader import generate_text
from backend.tools.knowledge_graph import KnowledgeGraph, extract_entities_and_relations
from backend.tools.source_scorer import score_source
from backend.tools.url_filter import is_allowed
from backend.tools.evidence_builder import build_evidence
from backend.tools.vector_store import VectorStore
from backend.tools.web_search_engine import multi_query_search, parallel_scrape_pages

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """\
You are a senior research analyst. Below is ranked evidence gathered from \
multiple sources for the research task.

Research Task: {task}

Evidence:
{content}

Write a concise factual analysis (300–500 words) covering all key facts, \
figures, dates, and insights from the evidence above. \
Cite sources by domain name where relevant. Do not invent facts."""


class ResearchAgent:
    def __init__(self, vector_store: VectorStore, knowledge_graph: Optional[KnowledgeGraph] = None) -> None:
        self.vector_store   = vector_store
        self.kg             = knowledge_graph
        self.max_pages      = get("research.max_pages_per_task", 10)
        self.max_results    = get("research.max_results_per_task", 10)
        self.follow_links   = get("research.follow_links", False)
        self.max_follow     = get("research.max_follow_links", 2)
        self.min_score      = get("research.source_min_score", 45)

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
        logger.info(f"[ResearchAgent]   Task      : {task}")
        logger.info(f"[ResearchAgent]   max_pages : {self.max_pages}")

        # ── 1. Generate 3 focused queries from the task ───────────────────
        _status(f"🔎 Generating search queries: {task[:70]}")
        queries = self._make_queries(task)
        logger.info(f"[ResearchAgent] Queries: {queries}")

        # ── 2. Multi-query search ─────────────────────────────────────────
        _status(f"🌐 Searching ({len(queries)} queries)…")
        candidates = multi_query_search(queries, results_per_query=self.max_results)
        logger.info(f"[ResearchAgent] Search returned {len(candidates)} unique URLs")

        # ── 3. Domain filter ──────────────────────────────────────────────
        before = len(candidates)
        candidates = [c for c in candidates if is_allowed(c.get("url", ""))]
        dropped = before - len(candidates)
        if dropped:
            logger.info(f"[ResearchAgent] Domain filter: dropped {dropped} low-quality URLs")

        if not candidates:
            _status(f"⚠️ No credible URLs found for: {task[:60]}")
            return sources

        # Cap to top N unique URLs (prioritise by pre-score)
        # max_pages * 1.5 gives a scrape buffer for 404s/stubs (benchmark: ~30% fail rate)
        scrape_budget = min(15, max(self.max_pages + 4, int(self.max_pages * 1.5)))
        candidates = sorted(
            candidates[:50],
            key=lambda c: score_source(c["url"], c.get("snippet", ""), ""),
            reverse=True,
        )[:scrape_budget]

        # ── 4. Scrape pages in parallel (I/O-bound — safe with threads) ─────
        _status(f"  📄 Scraping {min(self.max_pages, len(candidates))} pages in parallel…")
        pages = parallel_scrape_pages(
            candidates[: self.max_pages],
            min_chars=1000,
            follow_links=self.follow_links,
        )

        # Store in vector DB and extract entities for each page
        for page in pages:
            url   = page["url"]
            title = page["title"]
            text  = page["text"]
            self._store(task, url, title, text[:3000])
            if self.kg is not None:
                extract_entities_and_relations(text=text[:4000], source_url=url, kg=self.kg)

        logger.info(f"[ResearchAgent] Scraped {len(pages)} usable pages")

        if not pages:
            _status(f"⚠️ No usable content scraped for: {task[:60]}")
            return sources

        # ── 5. Semantic ranking + evidence assembly ────────────────────────
        _status(f"  🧠 Ranking & extracting evidence from {len(pages)} pages…")
        evidence = build_evidence(task, pages, top_k=20, max_chars=20_000)

        logger.info(
            f"[ResearchAgent] Evidence: {len(evidence.sources)} sources, "
            f"{len(evidence.context):,} chars"
        )

        # ── 6. Single LLM analysis over all ranked evidence ──────────────
        if evidence.context:
            _status(f"  ✍️  Analysing evidence…")
            analysis = generate_text(
                _ANALYSIS_PROMPT.format(
                    task=task,
                    content=evidence.context,
                ),
                max_new_tokens=800, role="writer",
            )
        else:
            analysis = ""

        # ── 7. Build source entries from validated evidence ───────────────
        all_images: List[Dict] = []
        for page in pages:
            all_images.extend(page.get("images", []))

        for src in evidence.sources:
            url   = src["url"]
            title = src.get("title", url)
            cred  = score_source(url, "", "")

            sources.append({
                "task":              task,
                "url":               url,
                "title":             title,
                "snippet":           "",
                "text":              "",
                "analysis":          analysis,
                "images":            all_images[:8],
                "youtube_embeds":    [],
                "credibility_score": cred,
            })

        # Attach full analysis to first source only (prevents huge duplication)
        if sources:
            sources[0]["analysis"] = analysis

        logger.info(
            f"[ResearchAgent] ══ research_task END  sources={len(sources)} ══"
        )
        _status(f"✅ Done: {task[:60]} — {len(sources)} sources")
        return sources

    # ── Query generator ───────────────────────────────────────────────────────

    def _make_queries(self, task: str) -> List[str]:
        """
        Convert a research sub-task into 3 focused search queries.
        Uses simple deterministic expansion — fast and reliable.
        The QueryPlanner (DeepSeek) handles the broader Phase-0 queries;
        per-task queries just need to be concise and varied.
        """
        import datetime
        base = task.strip().rstrip("?").strip()
        year = datetime.date.today().year
        return [
            base,
            f"{base} {year}",
            f"{base} analysis report",
        ]
