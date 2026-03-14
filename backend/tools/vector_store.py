"""
Vector Store — Deep Research AI

Design pattern: Repository — abstracts all ChromaDB operations behind a clean
domain interface.  Callers never touch ChromaDB or SentenceTransformer directly.

Embedding: BAAI/bge-small-en-v1.5 (~100 MB, stays loaded throughout session).
           Model name is read from config models.embedding.hf_repo so it can be
           swapped without touching code.
Chunking:  recursive character splitting (mirrors LangChain RecursiveCharacterTextSplitter).
"""
import logging
import uuid
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from backend.config_loader import get

logger = logging.getLogger(__name__)


# ── Chunking utility ──────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Recursive character text splitter (no external dependency)."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []

    def _split(t: str, sep_idx: int) -> None:
        if len(t) <= chunk_size or sep_idx >= len(separators):
            if t.strip():
                chunks.append(t.strip())
            return
        sep = separators[sep_idx]
        if not sep:
            start = 0
            while start < len(t):
                chunks.append(t[start : start + chunk_size])
                start += chunk_size - overlap
            return
        current = ""
        for part in t.split(sep):
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    _split(current, sep_idx + 1)
                current = part
        if current:
            _split(current, sep_idx + 1)

    _split(text, 0)

    # Deduplicate while preserving order
    seen: set = set()
    return [c for c in chunks if c not in seen and not seen.add(c)]  # type: ignore[func-returns-value]


# ── Repository ────────────────────────────────────────────────────────────────

class VectorStore:
    """ChromaDB-backed vector store with sentence-transformer embeddings."""

    def __init__(self) -> None:
        # ── Vector DB path ────────────────────────────────────────────────────
        _project_root = Path(__file__).parent.parent.parent
        raw_db = get("storage.vector_db", "")
        db_path = Path(raw_db) if raw_db else _project_root / "cache" / "vector_db"

        db_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Vector DB → {db_path.resolve()}")

        self._client = chromadb.PersistentClient(
            path=str(db_path.resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
        collection_name = get("vector_db.collection", "research_memory")
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # ── Embedding model cache ─────────────────────────────────────────────
        _project_root2 = Path(__file__).parent.parent.parent
        raw_cache = get("storage.hf_cache", "")
        cache_dir = Path(raw_cache).expanduser() if raw_cache else _project_root2 / "cache" / "hub"

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir_str = str(cache_dir.resolve())
        logger.info(f"Embedding cache → {cache_dir_str}")

        embedding_model = get("models.embedding.hf_repo", "BAAI/bge-small-en-v1.5")
        logger.info(f"Loading embedding model: {embedding_model}")

        def _load_embedder(model_name: str, local_only: bool) -> SentenceTransformer:
            return SentenceTransformer(
                model_name,
                cache_folder=cache_dir_str,
                trust_remote_code=True,
                local_files_only=local_only,
                # Force CPU for the tiny embedding model (~100 MB).
                # This keeps the full Metal GPU budget available for the 5 GB LLM
                # and avoids MPS/MLX Metal buffer contention on 16 GB M4.
                device="cpu",
            )

        try:
            # Try local-only first (fast, no network) — works when already cached
            try:
                self._embedder = _load_embedder(embedding_model, local_only=True)
                logger.info(f"Embedding model loaded from local cache (offline).")
            except Exception:
                # Not cached yet — allow network download
                logger.info(f"Embedding model not in cache — downloading…")
                self._embedder = _load_embedder(embedding_model, local_only=False)
        except Exception as err:
            fallback = "BAAI/bge-small-en-v1.5"
            logger.warning(f"Embedding model '{embedding_model}' failed ({err}), using fallback: {fallback}")
            try:
                self._embedder = _load_embedder(fallback, local_only=True)
            except Exception:
                self._embedder = _load_embedder(fallback, local_only=False)

        self._chunk_size    = get("vector_db.chunk_size", 1000)
        self._chunk_overlap = get("vector_db.chunk_overlap", 200)
        logger.info(f"VectorStore ready — collection: {collection_name}")

    # ── Repository interface ──────────────────────────────────────────────────

    def store_document(self, text: str, metadata: Dict = None) -> List[str]:
        """Chunk, embed, and store a document. Returns list of chunk IDs."""
        if not text or not text.strip():
            return []

        ids: List[str] = []
        for chunk in _chunk_text(text, self._chunk_size, self._chunk_overlap):
            doc_id = str(uuid.uuid4())
            try:
                embedding = self._embedder.encode(chunk, normalize_embeddings=True).tolist()
                self._collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata or {}],
                )
                ids.append(doc_id)
            except Exception as e:
                logger.warning(f"Failed to store chunk: {e}")
        return ids

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Semantic search. Returns list of {text, metadata, score}."""
        if self._collection.count() == 0:
            return []
        try:
            embedding = self._embedder.encode(query, normalize_embeddings=True).tolist()
            results   = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, self._collection.count()),
            )
            return [
                {
                    "text":     doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score":    1.0 - (results["distances"][0][i] if results.get("distances") else 0.0),
                }
                for i, doc in enumerate(results["documents"][0])
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_context_for_query(self, query: str, n_results: int = 5) -> str:
        """Retrieve relevant context as a formatted string."""
        docs = self.search_documents(query, n_results=n_results)
        if not docs:
            return ""
        parts = []
        for d in docs:
            source = d["metadata"].get("title") or d["metadata"].get("url") or "Unknown"
            parts.append(f"[{source}]\n{d['text']}")
        return "\n\n---\n\n".join(parts)

    def count(self) -> int:
        return self._collection.count()

    # Alias for callers that use the LangChain-style naming convention
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Alias for search_documents(). Returns list of {text, metadata, score}."""
        return self.search_documents(query, n_results=k)

    def clear(self) -> None:
        """Remove all documents from the collection."""
        try:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)
        except Exception as e:
            logger.warning(f"Vector store clear warning: {e}")
        logger.info("Vector store cleared.")
