# backend/app/documents/retrieval/hybrid_retriever.py
"""Hybrid retrieval implementation.

Combines vector and lexical candidate lists, deduplicates by ``DocumentChunk.id`` and
calculates a hybrid score based on configured weights. The function returns a list of
``ChunkResult`` objects ready for downstream consumption.
"""

from typing import List, Tuple, Optional

from .config import VECTOR_WEIGHT, LEXICAL_WEIGHT, RELEVANCE_THRESHOLD, VECTOR_TOP_K
from .models import ChunkResult


def hybrid_retrieve(
    vector_results: List[Tuple[object, float]],  # (DocumentChunk, vector_score)
    lexical_results: List[Tuple[object, float]],  # (DocumentChunk, lexical_score)
    top_k: Optional[int] = None,
) -> List[ChunkResult]:
    """Merge vector and lexical results.

    Args:
        vector_results: List of tuples ``(DocumentChunk, vector_score)``.
        lexical_results: List of tuples ``(DocumentChunk, lexical_score)``.
        top_k: Number of final results to return. If ``None`` defaults to
            ``VECTOR_TOP_K`` (the final top‑k default).

    Returns:
        A list of ``ChunkResult`` objects sorted deterministically by ``score``
        descending and then ``chunk_id`` ascending.
    """
    if top_k is None:
        top_k = VECTOR_TOP_K

    # Merge by chunk id, preserving individual scores.
    merged: dict[int, dict] = {}
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

    # Build ChunkResult objects with hybrid scoring.
    results: List[ChunkResult] = []
    for entry in merged.values():
        chunk = entry["chunk"]
        v = entry.get("vector_score") or 0.0
        l = entry.get("lexical_score") or 0.0
        hybrid = VECTOR_WEIGHT * v + LEXICAL_WEIGHT * l
        # If both scores are present, use hybrid as final score, otherwise the
        # available score.
        final_score = hybrid if (entry.get("vector_score") is not None and entry.get("lexical_score") is not None) else max(v, l)
        if hybrid < RELEVANCE_THRESHOLD:
            continue
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
        results.append(result)

    # Deterministic ordering: score desc, then chunk_id asc.
    results.sort(key=lambda r: (-r.score, r.chunk_id))
    return results[:top_k]
