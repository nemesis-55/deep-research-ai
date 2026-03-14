"""
Tools package — Deep Research AI

Exports:
    VectorStore          — ChromaDB-backed vector store
    KnowledgeGraph       — NetworkX entity-relation graph
    extract_entities_and_relations — LLM entity extractor
    score_source         — single-source credibility score
    score_and_filter     — filter + sort a list of sources by credibility
    search_web           — DuckDuckGo web search
    scrape_page          — HTML scraper + link follower
"""

from __future__ import annotations
from typing import TYPE_CHECKING

# Lazy imports — avoids pulling in sentence_transformers/torch at package load
# time. Each module is imported on first attribute access only.
def __getattr__(name: str):
    if name == "VectorStore":
        from backend.tools.vector_store import VectorStore
        return VectorStore
    if name in ("KnowledgeGraph", "extract_entities_and_relations"):
        from backend.tools.knowledge_graph import KnowledgeGraph, extract_entities_and_relations
        return locals()[name]
    if name in ("score_source", "score_and_filter"):
        from backend.tools.source_scorer import score_source, score_and_filter
        return locals()[name]
    if name == "search_web":
        from backend.tools.web_search import search_web
        return search_web
    if name == "scrape_page":
        from backend.tools.page_scraper import scrape_page
        return scrape_page
    raise AttributeError(f"module 'backend.tools' has no attribute {name!r}")

__all__ = [
    "VectorStore",
    "KnowledgeGraph",
    "extract_entities_and_relations",
    "score_source",
    "score_and_filter",
    "search_web",
    "scrape_page",
]
