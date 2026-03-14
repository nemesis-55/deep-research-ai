"""
Research Agent — Deep Research AI

Flow per task:
  1. Generate 5 focused search queries via query_generator (DeepSeek-R1)
  2. Fan-out multi-query DDG search (10 results/query, cap 15 unique URLs)
  3. Source filter (blocklist + prefer-list sort) via source_filter
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
from backend.tools.query_generator import generate_queries
from backend.tools.source_filter import filter_and_sort
from backend.tools.source_scorer import score_source
from backend.tools.url_filter import is_allowed
from backend.tools.evidence_builder import build_evidence
from backend.tools.vector_store import VectorStore
from backend.tools.web_search import fan_out_search
from backend.tools.web_search_engine import parallel_scrape_pages

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
        prebuilt_queries: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Full deep research for one task. Returns list of source dicts.

        Args:
            task:              Research sub-task description.
            status_callback:   Optional SSE progress callback.
            prebuilt_queries:  Pre-generated search queries from the pipeline's
                               planner phase.  When provided, skips the
                               generate_queries() LLM call entirely — no model
                               swap needed mid-pipeline.
        """
        sources: List[Dict] = []

        def _status(msg: str) -> None:
            if status_callback:
                status_callback(msg)
            logger.info(f"[Research] {msg}")

        logger.info(f"[ResearchAgent] ══ research_task START ══════════════════════════════")
        logger.info(f"[ResearchAgent]   Task      : {task}")
        logger.info(f"[ResearchAgent]   max_pages : {self.max_pages}")

        # ── 1. Use pre-built queries (no LLM call) or generate on-the-fly ─
        if prebuilt_queries:
            queries = prebuilt_queries
            logger.info(
                f"[Research Pipeline] queries={len(queries)} (pre-built, no LLM): "
                + " | ".join(f'"{q}"' for q in queries)
            )
        else:
            _status(f"🔎 Generating search queries: {task[:70]}")
            queries = generate_queries(task, n=5)
            logger.info(
                f"[Research Pipeline] queries={len(queries)} (on-the-fly): "
                + " | ".join(f'"{q}"' for q in queries)
            )

        # ── 2. Fan-out multi-query search (10 results/query, cap 15 URLs) ─
        _status(f"🌐 Searching ({len(queries)} queries)…")
        raw_candidates = fan_out_search(queries, results_per_query=10, max_urls=15)
        logger.info(
            f"[Research Pipeline] search returned {len(raw_candidates)} raw URLs"
        )

        # ── 3. Source filter: blocklist + prefer-list sort + url_filter ───
        candidates = filter_and_sort(raw_candidates, cap=15)
        candidates = [c for c in candidates if is_allowed(c.get("url", ""))]
        logger.info(
            f"[Research Pipeline] urls after filter={len(candidates)}"
        )

        if not candidates:
            _status(f"⚠️ No credible URLs found for: {task[:60]}")
            return sources

        # Sort by source credibility score before capping scrape budget
        # (filter_and_sort already prefer-lists, this adds domain score on top)
        scrape_budget = min(self.max_pages, len(candidates))
        candidates = sorted(
            candidates,
            key=lambda c: score_source(c["url"], c.get("snippet", ""), ""),
            reverse=True,
        )[:scrape_budget]

        # ── 4. Scrape pages in parallel (I/O-bound — safe with threads) ─────
        _status(f"  📄 Scraping {scrape_budget} pages in parallel…")
        pages = parallel_scrape_pages(
            candidates,
            min_chars=1000,
            follow_links=self.follow_links,
        )
        logger.info(
            f"[Research Pipeline] chunks_after_scrape={len(pages)} pages"
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
            f"[Research Pipeline] sources={len(evidence.sources)} "
            f"chunks_ranked={min(20, len(pages))} "
            f"evidence_chars={len(evidence.context):,}"
        )

        # ── 6. Single LLM analysis over all ranked evidence ──────────────
        if evidence.context:
            _status(f"  ✍️  Analysing evidence…")
            analysis = generate_text(
                _ANALYSIS_PROMPT.format(
                    task=task,
                    content=evidence.context,
                ),
                max_new_tokens=600, role="writer",  # 600 tok @ ~10 tok/s ≈ 60 s
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
