"""
Knowledge Graph — Deep Research AI

Lightweight entity-relationship graph backed by NetworkX.
Persisted as JSON on the external drive between research sessions.

Schema
──────
  Node  : {id, type, label, sources: [url], confidence: float}
  Edge  : {source_id, target_id, relation, weight: float}

Public API
──────────
  KnowledgeGraph.add_entities(entities: list[dict])
  KnowledgeGraph.add_relation(src, relation, tgt, weight, source_url)
  KnowledgeGraph.get_related(entity: str, depth: int) → list[dict]
  KnowledgeGraph.to_json() → dict           # for SSE / frontend
  KnowledgeGraph.save()
  KnowledgeGraph.load()
  KnowledgeGraph.clear()
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from backend.config_loader import get

logger = logging.getLogger(__name__)

_DEFAULT_PATH = str(Path(__file__).parent.parent.parent / "cache" / "knowledge_graph.json")


class KnowledgeGraph:
    def __init__(self, auto_load: bool = False) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._path  = Path(get("storage.knowledge_graph", _DEFAULT_PATH))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Only auto-load persisted graph when explicitly requested.
        # The research pipeline always calls clear() immediately after __init__,
        # so loading then clearing was pure wasted disk I/O.
        if auto_load:
            self.load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        data = nx.node_link_data(self._graph)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[KG] Saved {self._graph.number_of_nodes()} nodes, "
                    f"{self._graph.number_of_edges()} edges → {self._path}")

    def load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                self._graph = nx.node_link_graph(data, directed=True)
                logger.info(f"[KG] Loaded {self._graph.number_of_nodes()} nodes "
                            f"from {self._path}")
            except Exception as e:
                logger.warning(f"[KG] Load failed ({e}) — starting fresh.")
                self._graph = nx.DiGraph()

    def clear(self) -> None:
        self._graph.clear()
        logger.info("[KG] Cleared.")

    # ── Entity management ─────────────────────────────────────────────────────

    def add_entities(self, entities: List[Dict], source_url: str = "") -> None:
        """
        Add a list of entity dicts to the graph.
        Expected dict keys: label (str), type (str), confidence (float).
        """
        for ent in entities:
            label      = str(ent.get("label", "")).strip()
            etype      = str(ent.get("type",  "entity")).lower()
            confidence = float(ent.get("confidence", 1.0))
            if not label:
                continue

            node_id = _normalise(label)
            if self._graph.has_node(node_id):
                # merge — bump confidence, add source
                node = self._graph.nodes[node_id]
                node["confidence"] = max(node.get("confidence", 0), confidence)
                if source_url and source_url not in node.get("sources", []):
                    node.setdefault("sources", []).append(source_url)
            else:
                self._graph.add_node(
                    node_id,
                    label      = label,
                    type       = etype,
                    confidence = confidence,
                    sources    = [source_url] if source_url else [],
                )

    def add_relation(
        self,
        src:        str,
        relation:   str,
        tgt:        str,
        weight:     float = 1.0,
        source_url: str   = "",
    ) -> None:
        src_id = _normalise(src)
        tgt_id = _normalise(tgt)
        for nid, label in [(src_id, src), (tgt_id, tgt)]:
            if not self._graph.has_node(nid):
                self._graph.add_node(nid, label=label, type="entity",
                                     confidence=0.7, sources=[])
        self._graph.add_edge(src_id, tgt_id, relation=relation,
                             weight=weight, source=source_url)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_related(self, entity: str, depth: int = 2) -> List[Dict]:
        node_id = _normalise(entity)
        if not self._graph.has_node(node_id):
            return []
        nodes = nx.ego_graph(self._graph, node_id, radius=depth)
        result = []
        for nid in nodes.nodes:
            n = self._graph.nodes[nid]
            edges = []
            for _, tgt, data in self._graph.edges(nid, data=True):
                edges.append({
                    "to":       self._graph.nodes[tgt].get("label", tgt),
                    "relation": data.get("relation", ""),
                    "weight":   data.get("weight", 1.0),
                })
            result.append({
                "id":         nid,
                "label":      n.get("label", nid),
                "type":       n.get("type", "entity"),
                "confidence": n.get("confidence", 1.0),
                "sources":    n.get("sources", []),
                "edges":      edges,
            })
        return result

    # ── Serialisation for SSE / frontend ─────────────────────────────────────

    def to_json(self) -> Dict:
        nodes = []
        for nid, data in self._graph.nodes(data=True):
            nodes.append({
                "id":         nid,
                "label":      data.get("label", nid),
                "type":       data.get("type", "entity"),
                "confidence": round(data.get("confidence", 1.0), 2),
                "sources":    data.get("sources", [])[:3],
            })
        edges = []
        for src, tgt, data in self._graph.edges(data=True):
            edges.append({
                "source":   src,
                "target":   tgt,
                "relation": data.get("relation", ""),
                "weight":   round(data.get("weight", 1.0), 2),
            })
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()


# ── Entity extractor (lightweight — NO model swap) ────────────────────────────
#
# Previously this called generate_text(role="planner") which caused a destructive
# model swap during the writer phase (planner → writer → planner → writer …).
# On 16 GB Apple Silicon that meant ~50 swaps × 45 s = ~40 min of pure overhead.
#
# Replacement strategy:
#   1. Regex patterns catch capitalised entities fast (CPU-only, zero latency).
#   2. If the currently-active model IS the writer (already in memory), a small
#      structured prompt is sent to IT — no swap at all.
#   3. spaCy is tried as a bonus if installed (pip install spacy + en_core_web_sm).

import re as _re

# Common English stopwords to filter out false-positive entity matches
_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "their", "our", "we", "they",
    "he", "she", "i", "you", "not", "also", "more", "new", "said", "than",
    "about", "after", "before", "between", "into", "through", "during",
    "including", "among", "across", "against", "without", "within",
})

_CAPITALIZED = _re.compile(r'\b([A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){0,3})\b')


def _regex_entities(text: str, max_entities: int = 20) -> list[dict]:
    """Extract capitalised multi-word tokens as candidate entities."""
    seen: dict[str, int] = {}
    for m in _CAPITALIZED.finditer(text):
        label = m.group(1).strip()
        words = label.lower().split()
        if all(w in _STOP for w in words):
            continue
        if len(label) < 3:
            continue
        seen[label] = seen.get(label, 0) + 1

    # Sort by frequency; assign confidence from frequency
    results = []
    for label, freq in sorted(seen.items(), key=lambda x: -x[1])[:max_entities]:
        conf = min(0.5 + freq * 0.05, 0.95)
        results.append({"label": label, "type": "entity", "confidence": round(conf, 2)})
    return results


def _spacy_entities(text: str, max_entities: int = 20) -> list[dict] | None:
    """Try spaCy NER — returns None if spaCy / model not installed."""
    try:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:4000])
        seen: dict[str, dict] = {}
        for ent in doc.ents:
            if ent.label_ in ("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
                               "ORDINAL", "CARDINAL"):
                continue
            label = ent.text.strip()
            if label in seen:
                seen[label]["confidence"] = min(seen[label]["confidence"] + 0.05, 0.95)
            else:
                type_map = {
                    "ORG": "company", "PERSON": "person", "GPE": "location",
                    "LOC": "location", "PRODUCT": "product", "WORK_OF_ART": "concept",
                    "LAW": "concept", "EVENT": "concept", "NORP": "concept",
                    "FAC": "location", "LANGUAGE": "concept",
                }
                seen[label] = {
                    "label": label,
                    "type": type_map.get(ent.label_, "entity"),
                    "confidence": 0.80,
                }
        return list(seen.values())[:max_entities]
    except Exception:
        return None


_ENTITY_PROMPT_WRITER = """\
Extract named entities from the text below.
Return ONLY valid JSON — no text before or after:
{{
  "entities": [
    {{"label": "entity name", "type": "company|person|technology|product|location|concept", "confidence": 0.9}}
  ],
  "relations": [
    {{"source": "entity A", "relation": "relation verb", "target": "entity B"}}
  ]
}}

Text (first 2000 chars):
{text}
"""


def extract_entities_and_relations(
    text: str,
    source_url: str,
    kg: KnowledgeGraph,
    max_entities: int = 20,
) -> None:
    """
    Extract entities + relations from *text* and add them to *kg*.

    Strategy (no model swap guaranteed):
      1. spaCy NER  — if installed (best quality, CPU, zero latency)
      2. Regex NER  — fast capitalised-token heuristic (CPU, always available)
      3. Writer LLM — only if writer is ALREADY the active model (no swap cost)

    The old approach (role="planner") caused N×2 model swaps during research,
    adding hours of overhead on 16 GB Apple Silicon. That path is removed.
    """
    try:
        entities: list[dict] = []
        relations: list[dict] = []

        # ── 1. Try spaCy (best, CPU-only) ─────────────────────────────────
        spacy_result = _spacy_entities(text, max_entities)
        if spacy_result is not None:
            entities = spacy_result
            logger.debug(f"[KG] spaCy: {len(entities)} entities from {source_url[:60]}")

        else:
            # ── 2. Regex fallback (always works, CPU-only) ─────────────────
            entities = _regex_entities(text, max_entities)
            logger.debug(f"[KG] Regex: {len(entities)} entities from {source_url[:60]}")

            # ── 3. Bonus: writer LLM if already loaded — gets us relations too
            try:
                from backend.model_manager import _active
                if _active is not None and _active.role in ("writer", "chat"):
                    from backend.model_loader import generate_text
                    raw = generate_text(
                        _ENTITY_PROMPT_WRITER.format(text=text[:2000]),
                        max_new_tokens=400,
                        role=_active.role,   # ← reuse CURRENT model, no swap
                    )
                    data = _parse_entity_json(raw)
                    if data.get("entities"):
                        entities = data["entities"][:max_entities]
                    relations = data.get("relations", [])[:max_entities]
            except Exception as e:
                logger.debug(f"[KG] LLM entity bonus skipped: {e}")

        kg.add_entities(entities, source_url=source_url)

        for rel in relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            r   = rel.get("relation", "related_to")
            if src and tgt:
                kg.add_relation(src, r, tgt, source_url=source_url)

        logger.info(f"[KG] +{len(entities)} entities, "
                    f"+{len(relations)} relations from {source_url[:60]}")
    except Exception as e:
        logger.warning(f"[KG] Entity extraction skipped: {e}")


def _parse_entity_json(raw: str) -> Dict:
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {}


def _normalise(label: str) -> str:
    """Lowercase + strip punctuation for stable node IDs."""
    return re.sub(r"[^\w\s]", "", label.lower()).strip().replace(" ", "_")
