# backend/app/documents/retrieval/hybrid.py

"""Hybrid retrieval utilities.

Combines vector and lexical candidates, deduplicates, scores, and ranks them.
"""

from typing import List, Dict, Tuple, Optional
from .config import VECTOR_WEIGHT, LEXICAL_WEIGHT, RELEVANCE_THRESHOLD, VECTOR_TOP_K
from .models import ChunkResult


def merge_hybrid(
    vector_results: List[Tuple[object, float]],  # (DocumentChunk, vector_score)
    lexical_results: List[Tuple[object, float]],  # (DocumentChunk, lexical_score)
    top_k: Optional[int] = None,
) -> List[ChunkResult]:
    """Merge vector and lexical results into a deterministic ranked list.

    - Deduplicate by ``DocumentChunk.id``.
    - Preserve individual scores.
    - Compute ``hybrid_score = VECTOR_WEIGHT * vector_score + LEXICAL_WEIGHT * lexical_score``.
    - If a score is missing, treat it as ``0`` for weighting.
    - Final ``score`` is the ``hybrid_score`` (or the available score if only one source).
    - Deterministic tie‑break by ``chunk.id``.
    - Apply ``RELEVANCE_THRESHOLD``: only keep items with ``hybrid_score`` >= threshold.
    - Return up to ``top_k`` items (default ``VECTOR_TOP_K``).
    """
    if top_k is None:
        top_k = VECTOR_TOP_K

    # Map chunk.id -> dict with collected data
    merged: Dict[int, Dict] = {}
    for chunk, v_score in vector_results:
        merged.setdefault(chunk.id, {
            "chunk": chunk,
            "vector_score": v_score,
            "lexical_score": None,
        })["vector_score"] = v_score
    for chunk, l_score in lexical_results:
        merged.setdefault(chunk.id, {
            "chunk": chunk,
            "vector_score": None,
            "lexical_score": l_score,
        })["lexical_score"] = l_score

    # Build ChunkResult objects with scoring
    candidates: List[ChunkResult] = []
    for entry in merged.values():
        chunk = entry["chunk"]
        v = entry.get("vector_score") or 0.0
        l = entry.get("lexical_score") or 0.0
        hybrid = VECTOR_WEIGHT * v + LEXICAL_WEIGHT * l
        # Determine final score: if both present use hybrid, else use whichever exists
        final_score = hybrid if (entry.get("vector_score") is not None and entry.get("lexical_score") is not None) else max(v, l)
        if hybrid < RELEVANCE_THRESHOLD:
            continue
        # Build result schema – extract needed fields from ORM models
        result = ChunkResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            vector_score=entry.get("vector_score"),
            lexical_score=entry.get("lexical_score"),
            hybrid_score=hybrid,
            score=final_score,
            filename=chunk.document.filename,
            document_type=chunk.document.document_type,
            source=chunk.document.source,
            source_pages=chunk.metadata_json.get("source_pages") if isinstance(chunk.metadata_json, dict) else None,
            metadata=chunk.metadata_json,
        )
        candidates.append(result)

    # Deterministic ranking: score desc, then chunk_id asc
    candidates.sort(key=lambda r: (-r.score, r.chunk_id))
    return candidates[:top_k]
