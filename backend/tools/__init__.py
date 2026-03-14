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

from backend.tools.vector_store import VectorStore
from backend.tools.knowledge_graph import KnowledgeGraph, extract_entities_and_relations
from backend.tools.source_scorer import score_source, score_and_filter
from backend.tools.web_search import search_web
from backend.tools.page_scraper import scrape_page

__all__ = [
    "VectorStore",
    "KnowledgeGraph",
    "extract_entities_and_relations",
    "score_source",
    "score_and_filter",
    "search_web",
    "scrape_page",
]
