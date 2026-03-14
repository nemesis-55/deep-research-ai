"""
Research Pipeline — Deep Research AI

Orchestrates the full deep-research flow as an async generator of SSE events.

Flow
────
  0. Deep multi-query crawl (Phase 0) — pre-populate vector store
       QueryPlanner → 10 queries → MultiSearch (100 URLs) → ArticleScraper
       (scrapes top 30, deduplicates, stores in ChromaDB)
       Refine loop: if stored < refine_threshold, generate new queries & repeat
  1. Ingest uploaded files → vector store
  2. Load planner (DeepSeek-R1 8B) → generate research plan → unload
  3. Load writer (Qwen2.5-7B) → execute each task:
       search → scrape → analyse → score → store → extract entities → KG
  4. Filter sources by credibility score
  5. Generate structured report  (writer now has RAG context from Phase 0)
  6. Emit knowledge graph snapshot

SSE event types
───────────────
  status | plan | progress | source | graph | report | done | error
"""
import asyncio
import logging
from typing import AsyncGenerator, List, Optional

from backend.agent.planner_agent import PlannerAgent
from backend.agent.query_planner import QueryPlanner
from backend.agent.report_agent import ReportAgent
from backend.agent.research_agent import ResearchAgent
from backend.config_loader import get
from backend.model_manager import (
    drain_think_queue,
    load_planner_model,
    swap_model,
    unload_model,
)
from backend.tools.article_scraper import ArticleScraper
from backend.tools.knowledge_graph import KnowledgeGraph
from backend.tools.page_scraper import _parse_local_file
from backend.tools.source_scorer import score_and_filter
from backend.tools.vector_store import VectorStore
from backend.tools.web_search_engine import multi_query_search

logger = logging.getLogger(__name__)


def _drain_think(loop) -> list:
    """Drain think-block queue and return SSE-ready event dicts."""
    items = drain_think_queue()
    events = []
    for item in items:
        events.append({
            "type":   "think",
            "role":   item.get("role", ""),
            "think":  item.get("think", item.get("text", "")),  # backwards compat
            "prompt": item.get("prompt", ""),
            "output": item.get("output", ""),
        })
    return events


class ResearchPipeline:
    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.kg           = KnowledgeGraph()
        self.planner      = PlannerAgent()
        self.reporter     = ReportAgent(self.vector_store)

    async def run(
        self,
        query:          str,
        uploaded_files: Optional[List[str]] = None,
    ) -> AsyncGenerator[dict, None]:

        loop        = asyncio.get_event_loop()
        all_sources: List[dict] = []

        _SEP = "═" * 65
        logger.info(f"\n{_SEP}")
        logger.info(f"  🔬 DEEP RESEARCH STARTED")
        logger.info(f"  Query: {query}")
        logger.info(f"{_SEP}")

        # Clear stale session data
        try:
            self.vector_store.clear()
            self.kg.clear()   # graph starts empty; this is a no-op unless a
                              # previous run added nodes without saving
        except Exception:
            pass

        # ── Phase 0: Deep multi-query web crawl ───────────────────────────
        deep_crawl_enabled  = get("deep_crawl.enabled", True)
        refine_threshold    = get("deep_crawl.refine_threshold", 8)
        num_queries         = get("deep_crawl.num_queries", 10)
        results_per_query   = get("deep_crawl.results_per_query", 10)

        deep_crawl_sources: List[dict] = []

        if deep_crawl_enabled:
            yield {"type": "status", "message": "🌐 Phase 0: Deep multi-query web crawl starting…"}

            # Load the planner model for query generation
            yield {"type": "status", "message": "🧠 Loading Planner for query generation…"}
            await loop.run_in_executor(None, load_planner_model)

            # Generate diverse search queries
            query_planner = QueryPlanner()
            yield {
                "type":    "status",
                "message": f"🔎 Generating {num_queries} search queries from planner LLM…",
            }
            try:
                queries: List[str] = await loop.run_in_executor(
                    None,
                    lambda: query_planner.generate_queries(query, n=num_queries),
                )
            except Exception as e:
                yield {"type": "status", "message": f"  ⚠️ Query generation failed ({e}), using fallback."}
                queries = query_planner._fallback(query, num_queries)

            yield {
                "type":    "status",
                "message": f"✅ {len(queries)} queries ready — starting fan-out search…",
            }

            # Fan-out search: N queries × M results = up to N×M candidate URLs
            def _do_multi_search() -> list:
                return multi_query_search(queries, results_per_query=results_per_query)

            candidates = await loop.run_in_executor(None, _do_multi_search)
            yield {
                "type":    "status",
                "message": f"🔗 {len(candidates)} unique URLs found — scraping top articles…",
            }

            # Scrape & store articles (Phase 0 — no per-source LLM analysis yet)
            scraper       = ArticleScraper(self.vector_store)
            status_buf_p0: List[str] = []

            def _do_scrape() -> list:
                return scraper.scrape_and_store(
                    candidates,
                    topic           = query,
                    status_callback = status_buf_p0.append,
                )

            deep_crawl_sources = await loop.run_in_executor(None, _do_scrape)

            for msg in status_buf_p0:
                yield {"type": "status", "message": msg}

            yield {
                "type":    "status",
                "message": (
                    f"📦 Phase 0 complete: {len(deep_crawl_sources)} articles stored "
                    f"in vector store ({self.vector_store.count()} total chunks)"
                ),
            }

            # ── Refine loop: if we got too few docs, search again ─────────
            if len(deep_crawl_sources) < refine_threshold:
                yield {
                    "type":    "status",
                    "message": (
                        f"🔁 Only {len(deep_crawl_sources)} articles — below threshold "
                        f"({refine_threshold}). Running refine pass…"
                    ),
                }
                try:
                    refine_queries: List[str] = await loop.run_in_executor(
                        None,
                        lambda: query_planner.generate_queries(
                            f"{query} detailed analysis research",
                            n=num_queries,
                        ),
                    )
                    # Exclude already-seen URLs
                    seen_urls = {s["url"] for s in deep_crawl_sources}

                    def _do_refine_search() -> list:
                        raw = multi_query_search(refine_queries, results_per_query=results_per_query)
                        return [r for r in raw if r["url"] not in seen_urls]

                    extra_candidates = await loop.run_in_executor(None, _do_refine_search)

                    if extra_candidates:
                        yield {
                            "type":    "status",
                            "message": f"🔗 Refine: {len(extra_candidates)} new URLs — scraping…",
                        }
                        status_buf_refine: List[str] = []

                        def _do_refine_scrape() -> list:
                            return scraper.scrape_and_store(
                                extra_candidates,
                                topic           = query,
                                status_callback = status_buf_refine.append,
                            )

                        extra_docs = await loop.run_in_executor(None, _do_refine_scrape)
                        for msg in status_buf_refine:
                            yield {"type": "status", "message": msg}

                        deep_crawl_sources.extend(extra_docs)
                        yield {
                            "type":    "status",
                            "message": f"✅ Refine complete: {len(deep_crawl_sources)} total articles",
                        }
                except Exception as e:
                    yield {"type": "status", "message": f"  ⚠️ Refine pass failed: {e}"}

        # ── Step 1: Ingest uploaded files ─────────────────────────────────
        if uploaded_files:
            yield {"type": "status", "message": f"📂 Ingesting {len(uploaded_files)} file(s)…"}
            for fp in uploaded_files:
                try:
                    parsed = await loop.run_in_executor(None, lambda p=fp: _parse_local_file(p))
                    text   = parsed.get("text", "")
                    if text:
                        self.vector_store.store_document(
                            text     = f"[Uploaded: {fp}]\n{text[:6000]}",
                            metadata = {"url": f"file://{fp}", "title": fp, "task": "uploaded"},
                        )
                        yield {"type": "status", "message": f"  ✅ {fp}"}
                    else:
                        yield {"type": "status", "message": f"  ⚠️ No text: {fp}"}
                except Exception as e:
                    yield {"type": "status", "message": f"  ❌ {fp}: {e}"}

        # ── Step 2: Planner — DeepSeek-R1 8B ─────────────────────────────
        # If Phase 0 already loaded the planner, this is a no-op (model stays loaded)
        if not deep_crawl_enabled:
            yield {"type": "status", "message": "🧠 Loading Planner (DeepSeek-R1 8B Q4)…"}
            await loop.run_in_executor(None, load_planner_model)

        yield {"type": "status", "message": "📋 Generating research plan…"}

        try:
            plan: List[str] = await loop.run_in_executor(
                None, lambda: self.planner.generate_plan(query)
            )
        except Exception as e:
            yield {"type": "error", "message": f"Planner failed: {e}"}
            return

        # Emit any think blocks the planner produced
        for ev in _drain_think(loop):
            yield ev

        yield {"type": "plan",   "plan": plan}
        yield {"type": "status", "message": f"✅ Plan ready — {len(plan)} tasks"}

        # ── Step 3: Swap to Writer — Qwen2.5-7B ──────────────────────────
        yield {"type": "status", "message": "🔄 Unloading Planner → Loading Writer (Qwen2.5-7B Q4)…"}
        await loop.run_in_executor(None, lambda: swap_model("writer"))
        yield {"type": "status", "message": "✅ Writer model ready"}

        # ── Step 4: Execute research tasks ────────────────────────────────
        researcher = ResearchAgent(self.vector_store, self.kg)

        for i, task in enumerate(plan, 1):
            yield {"type": "progress", "current": i, "total": len(plan), "task": task}
            yield {"type": "status",   "message": f"[{i}/{len(plan)}] 🔍 {task[:90]}"}

            status_buf: List[str] = []
            try:
                sources = await loop.run_in_executor(
                    None,
                    lambda t=task: researcher.research_task(t, status_callback=status_buf.append),
                )
            except Exception as e:
                yield {"type": "status", "message": f"  ❌ Task failed: {e}"}
                continue

            for msg in status_buf:
                yield {"type": "status", "message": msg}

            # Emit any think blocks produced during this task's AI calls
            for ev in _drain_think(loop):
                yield ev

            for s in sources:
                all_sources.append(s)
                yield {
                    "type":   "source",
                    "source": {
                        "title":            s.get("title", ""),
                        "url":              s.get("url", ""),
                        "task":             s.get("task", ""),
                        "snippet":          s.get("snippet", ""),
                        "credibility":      s.get("credibility_score", 0),
                        "images":           s.get("images", [])[:3],
                    },
                }

            # Emit incremental KG snapshot after each task
            yield {"type": "graph", "graph": self.kg.to_json()}

        # ── Step 5: Filter sources by credibility ─────────────────────────
        before = len(all_sources)
        all_sources = score_and_filter(all_sources)
        dropped     = before - len(all_sources)
        if dropped:
            yield {"type": "status", "message": f"🔎 Credibility filter: kept {len(all_sources)}/{before} sources"}

        # Merge Phase 0 deep-crawl sources (already scored) into the source list
        # so the report agent can reference them in citations.
        # Only add ones not already represented (avoid exact URL duplicates).
        if deep_crawl_sources:
            existing_urls = {s.get("url", "") for s in all_sources}
            unique_crawl  = [
                s for s in deep_crawl_sources
                if s.get("url", "") not in existing_urls
            ]
            if unique_crawl:
                all_sources = score_and_filter(all_sources + unique_crawl)
                yield {
                    "type":    "status",
                    "message": f"📎 Merged {len(unique_crawl)} Phase-0 articles → {len(all_sources)} total sources",
                }

        if not all_sources:
            yield {"type": "error", "message": "No credible sources collected. Cannot generate report."}
            return

        # ── Step 6: RAG context retrieval from vector store ───────────────
        # Pull the top-k most relevant passages from the pre-populated vector
        # store and prepend them to the report-writer context.
        rag_context = ""
        try:
            rag_k = get("deep_crawl.rag_top_k", 10)
            rag_context = self.vector_store.get_context_for_query(query, n_results=rag_k)
            if rag_context:
                yield {
                    "type":    "status",
                    "message": f"🧩 RAG: retrieved top-{rag_k} passages for report synthesis",
                }
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

        # ── Step 7: Generate report ───────────────────────────────────────
        yield {"type": "status", "message": f"📝 Writing report from {len(all_sources)} sources…"}
        try:
            report: str = await loop.run_in_executor(
                None,
                lambda: self.reporter.generate_report(query, all_sources, rag_context=rag_context),
            )
        except Exception as e:
            yield {"type": "error", "message": f"Report generation failed: {e}"}
            return

        # Emit any think blocks from report generation
        for ev in _drain_think(loop):
            yield ev

        # Save knowledge graph to disk
        try:
            await loop.run_in_executor(None, self.kg.save)
        except Exception:
            pass

        yield {"type": "report", "content": report}
        yield {"type": "graph",  "graph":   self.kg.to_json()}
        yield {
            "type":    "status",
            "message": (
                f"✅ Complete — {len(report):,} chars, "
                f"{len(all_sources)} sources, "
                f"{self.kg.node_count} entities, "
                f"{self.vector_store.count()} vector chunks"
            ),
        }
        yield {"type": "done"}
