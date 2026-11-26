# backend/app/documents/retrieval/vector.py

"""Vector retrieval utilities.

Performs cosine-similarity search over ``DocumentChunk`` embeddings.

Business isolation and optional document filters are enforced **in SQL**
(the WHERE clause always includes ``Document.business_id``). Similarity is
computed in Python because embeddings are persisted as JSON arrays in this
schema; for a business's document collection this is exact and efficient.
"""

from typing import List, Tuple, Optional

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, and_
from backend.app.models.document_models import DocumentChunk, Document
from backend.app.documents.retrieval.config import VECTOR_TOP_K, VECTOR_CANDIDATE_POOL


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def vector_search(
    db: Session,
    query_vector: List[float],
    business_id: int,
    top_k: Optional[int] = None,
    filters: Optional[dict] = None,
) -> List[Tuple[DocumentChunk, float]]:
    """Return a list of ``(DocumentChunk, similarity_score)`` tuples.

    Similarity is cosine similarity (higher is better). Chunks with NULL or
    empty embeddings are excluded. Business isolation is enforced in SQL.
    """
    if top_k is None:
        top_k = VECTOR_CANDIDATE_POOL
    if top_k <= 0:
        top_k = VECTOR_TOP_K

    stmt = (
        select(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            and_(
                Document.business_id == business_id,
                DocumentChunk.embedding != None,  # exclude NULL embeddings
            )
        )
    )

    if filters:
        if "document_id" in filters:
            stmt = stmt.where(Document.id == filters["document_id"])  # type: ignore
        if "document_type" in filters:
            stmt = stmt.where(Document.document_type == filters["document_type"])  # type: ignore
        if "source" in filters:
            stmt = stmt.where(Document.source == filters["source"])  # type: ignore

    # Pool of business-scoped candidates (bounded by the candidate pool size).
    stmt = stmt.limit(max(top_k * 5, VECTOR_CANDIDATE_POOL))
    chunks = db.execute(stmt).scalars().all()

    q = np.asarray(query_vector, dtype=np.float64)
    scored: List[Tuple[DocumentChunk, float]] = []
    for chunk in chunks:
        emb = chunk.embedding
        if not emb:
            continue
        try:
            vec = np.asarray(emb, dtype=np.float64)
        except Exception:
            continue
        if vec.shape != q.shape:
            continue  # dimension mismatch: never crash on malformed data
        scored.append((chunk, _cosine(q, vec)))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    # Deterministic ordering: score desc, then chunk id asc.
    scored = sorted(scored, key=lambda pair: (-pair[1], pair[0].id))
    return scored[:top_k]
