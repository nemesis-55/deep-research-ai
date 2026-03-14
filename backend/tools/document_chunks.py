"""
Document Chunker & Semantic Ranker — Deep Research AI
=======================================================
Two responsibilities:

  1. chunk_document()   — split page text into overlapping 500-token chunks
                          with source metadata attached to every chunk.

  2. rank_chunks()      — embed all chunks + the query, compute cosine
                          similarity, return the top-N most relevant chunks.

Design
──────
- Uses the *already-loaded* BAAI/bge-small-en-v1.5 SentenceTransformer from
  the existing VectorStore's embedding model — no second model load.
- Falls back to TF-IDF keyword overlap ranking when the embedding model is
  unavailable (e.g. during unit tests).
- Pure utility — no FastAPI, no database writes.

Constants (all tuneable via config.yaml or direct kwargs)
─────────────────────────────────────────────────────────
  CHUNK_TOKENS    ≈ 500 words  (English prose: ~1 word ≈ 1.3 tokens)
  CHUNK_OVERLAP   ≈ 50  words  (10 % overlap to preserve context at boundaries)
  MAX_CHUNKS      = 20         (top-K after ranking; fits in 20 000-char budget)
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Embedding model singleton — loaded once per process ───────────────────────
# Avoids the 4-second SentenceTransformer init penalty on every rank_chunks()
# call that caused the repeated "BertModel LOAD REPORT" spam in benchmarks.
_embed_model = None
_embed_model_name: str = ""


def _get_embed_model(model_name: str):
    """Return the cached SentenceTransformer, loading it only on first call."""
    global _embed_model, _embed_model_name
    if _embed_model is None or _embed_model_name != model_name:
        import logging as _logging
        # Suppress the BertModel LOAD REPORT noise from sentence-transformers
        _logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
        _logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(_logging.ERROR)
        from sentence_transformers import SentenceTransformer
        logger.info(f"[Chunks] Loading embedding model: {model_name}")
        _embed_model = SentenceTransformer(model_name)
        _embed_model_name = model_name
        logger.info(f"[Chunks] Embedding model ready.")
    return _embed_model

# ── Tunable defaults ──────────────────────────────────────────────────────────
CHUNK_WORDS    = 400    # target words per chunk  (≈ 500 tokens)
CHUNK_OVERLAP  = 50     # overlap words between consecutive chunks
MAX_CHUNKS     = 20     # top-K chunks kept after semantic ranking
MAX_CONTEXT_CHARS = 20_000   # hard ceiling on combined evidence text


# ── Data type ──────────────────────────────────────────────────────────────────

class Chunk:
    """A piece of text with its provenance."""
    __slots__ = ("text", "url", "title", "domain", "chunk_idx", "score")

    def __init__(
        self,
        text:      str,
        url:       str,
        title:     str  = "",
        domain:    str  = "",
        chunk_idx: int  = 0,
        score:     float = 0.0,
    ) -> None:
        self.text      = text
        self.url       = url
        self.title     = title
        self.domain    = domain
        self.chunk_idx = chunk_idx
        self.score     = score

    def __repr__(self) -> str:  # pragma: no cover
        return f"Chunk(score={self.score:.3f} url={self.url[:50]!r} idx={self.chunk_idx})"


# ── Chunking ──────────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    """Split text into sentences (simple regex — fast, good enough for news prose)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def chunk_document(
    text:   str,
    url:    str,
    title:  str = "",
    domain: str = "",
    chunk_words:   int = CHUNK_WORDS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split `text` into overlapping word-window chunks.

    Strategy:
      - Split into sentences first so we never cut mid-sentence.
      - Accumulate sentences until the word budget is reached.
      - Slide forward by (chunk_words - chunk_overlap) words.

    Returns a list of Chunk objects in document order.
    """
    if not text or not text.strip():
        return []

    sentences = _sentence_split(text)
    if not sentences:
        return []

    chunks:    list[Chunk] = []
    buf:       list[str]   = []   # sentence accumulator
    buf_words: int         = 0
    step       = max(1, chunk_words - chunk_overlap)
    idx        = 0

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        wc   = len(sent.split())

        buf.append(sent)
        buf_words += wc

        if buf_words >= chunk_words:
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text      = chunk_text,
                    url       = url,
                    title     = title,
                    domain    = domain,
                    chunk_idx = idx,
                ))
                idx += 1

            # Roll back to find the overlap start point
            rollback_words = 0
            j = len(buf) - 1
            while j >= 0 and rollback_words < chunk_overlap:
                rollback_words += len(buf[j].split())
                j -= 1
            buf       = buf[j + 1:]
            buf_words = sum(len(s.split()) for s in buf)

        i += 1

    # Flush any remaining sentences
    if buf:
        chunk_text = " ".join(buf).strip()
        if chunk_text and len(chunk_text) > 100:
            chunks.append(Chunk(
                text      = chunk_text,
                url       = url,
                title     = title,
                domain    = domain,
                chunk_idx = idx,
            ))

    logger.debug(f"[Chunks] {url[:60]!r} → {len(chunks)} chunks")
    return chunks


def chunk_documents(pages: list[dict], **kwargs) -> list[Chunk]:
    """
    Chunk a list of scraped page dicts.

    Each dict must have 'text' and 'url'; 'title' and 'domain' are optional.
    kwargs are forwarded to chunk_document().
    """
    from urllib.parse import urlparse
    all_chunks: list[Chunk] = []
    for page in pages:
        text   = (page.get("text") or "").strip()
        url    = page.get("url", "")
        title  = page.get("title", url)
        domain = page.get("domain", "") or urlparse(url).netloc
        if text and url:
            all_chunks.extend(chunk_document(text, url, title, domain, **kwargs))
    logger.info(f"[Chunks] {len(pages)} pages → {len(all_chunks)} total chunks")
    return all_chunks


# ── Semantic ranking ──────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    """Cosine similarity between two 1-D numpy/list vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tfidf_rank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """
    Fallback ranker using simple TF-IDF keyword overlap.
    Used when the embedding model cannot be imported.
    """
    query_words = set(query.lower().split())
    for chunk in chunks:
        words     = chunk.text.lower().split()
        if not words:
            chunk.score = 0.0
            continue
        hits      = sum(1 for w in words if w in query_words)
        chunk.score = hits / len(words)
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def rank_chunks(
    query:      str,
    chunks:     list[Chunk],
    top_k:      int = MAX_CHUNKS,
) -> list[Chunk]:
    """
    Rank `chunks` by semantic similarity to `query` using bge-small embeddings.

    Falls back to TF-IDF overlap if the embedding model is unavailable.
    Returns the top-`top_k` chunks sorted by descending score.
    """
    if not chunks:
        return []

    logger.info(f"[Chunks] Ranking {len(chunks)} chunks  top_k={top_k}")

    # ── Try embedding-based ranking ───────────────────────────────────────
    try:
        from backend.config_loader import get

        model_name = get("models.embedding.hf_repo", "BAAI/bge-small-en-v1.5")
        _model = _get_embed_model(model_name)

        texts = [c.text for c in chunks]
        # bge models work best with a query prefix
        q_prefixed = f"Represent this sentence: {query}"

        all_texts   = [q_prefixed] + texts
        embeddings  = _model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        q_emb       = embeddings[0]
        chunk_embs  = embeddings[1:]

        for chunk, emb in zip(chunks, chunk_embs):
            chunk.score = float(_cosine(q_emb, emb))

        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        top    = ranked[:top_k]
        logger.info(
            f"[Chunks] Semantic ranking done — "
            f"top score={top[0].score:.3f}  bottom={top[-1].score:.3f}"
        )
        return top

    except Exception as e:
        logger.warning(f"[Chunks] Embedding ranking failed ({e}), falling back to TF-IDF")
        ranked = _tfidf_rank(query, chunks)
        return ranked[:top_k]


# ── Evidence builder helper ───────────────────────────────────────────────────

def build_evidence_context(
    chunks:        list[Chunk],
    max_chars:     int = MAX_CONTEXT_CHARS,
    dedupe_urls:   bool = True,
) -> str:
    """
    Combine ranked chunks into a single evidence string for the LLM prompt.

    Format per chunk:
        Source: <domain>
        Title:  <title>
        URL:    <url>
        <chunk text>

    Truncates at `max_chars` to stay within context window limits.
    If `dedupe_urls` is True, no more than 3 chunks from the same URL are
    included (prevents one long article from dominating the evidence).
    """
    parts:    list[str] = []
    total     = 0
    url_count: dict[str, int] = {}

    for chunk in chunks:
        url = chunk.url
        if dedupe_urls:
            url_count[url] = url_count.get(url, 0) + 1
            if url_count[url] > 3:
                continue

        block = (
            f"Source: {chunk.domain or chunk.url}\n"
            f"Title:  {chunk.title}\n"
            f"URL:    {chunk.url}\n"
            f"{chunk.text}"
        )
        if total + len(block) > max_chars:
            # Trim to fill exactly to the budget
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining])
            break

        parts.append(block)
        total += len(block)

    context = "\n\n---\n\n".join(parts)
    logger.info(
        f"[Chunks] Evidence context: {len(parts)} chunks, "
        f"{len(context):,} chars (limit={max_chars:,})"
    )
    return context
