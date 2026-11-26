# backend/app/documents/retrieval/lexical.py

"""Lexical (full‑text) retrieval utilities using PostgreSQL native functions.

Provides a function to perform tsquery search against DocumentChunk.text.
"""

from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, desc, text
from backend.app.models.document_models import DocumentChunk, Document
from backend.app.documents.retrieval.config import VECTOR_TOP_K, LEXICAL_CANDIDATE_POOL


# ts_rank returns a tiny epsilon (1e-20) rather than zero for some non-matching
# queries; genuine matches score above ~0.01. Anything under this floor is
# treated as no match.
MIN_MEANINGFUL_RANK = 1e-6


def lexical_search(
    db: Session,
    query: str,
    business_id: int,
    top_k: Optional[int] = None,
    filters: Optional[dict] = None,
) -> List[Tuple[DocumentChunk, float]]:
    """Return list of (DocumentChunk, ts_rank) for the given query.

    Uses ``websearch_to_tsquery`` for user‑friendly query parsing.
    Applies business isolation and optional filters.
    """
    if top_k is None:
        top_k = LEXICAL_CANDIDATE_POOL

    # Build tsquery using websearch_to_tsquery (handles phrases, AND/OR etc.)
    tsquery = func.websearch_to_tsquery('english', query)
    # Compute rank against the text column
    rank_expr = func.ts_rank(
        func.to_tsvector('english', DocumentChunk.text),
        tsquery,
    )

    stmt = (
        select(DocumentChunk, rank_expr)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            and_(
                Document.business_id == business_id,
                rank_expr > MIN_MEANINGFUL_RANK,  # epsilon floor: 1e-20 is a non-match
            )
        )
    )

    # Apply optional filters
    if filters:
        if "document_id" in filters:
            stmt = stmt.where(Document.id == filters["document_id"])  # type: ignore
        if "document_type" in filters:
            stmt = stmt.where(Document.document_type == filters["document_type"])  # type: ignore
        if "source" in filters:
            stmt = stmt.where(Document.source == filters["source"])  # type: ignore

    stmt = stmt.order_by(desc(rank_expr)).limit(top_k)

    results = db.execute(stmt).all()
    # each row: (DocumentChunk, rank)
    scored = [(row[0], float(row[1])) for row in results]
    if not scored:
        return []
    # Normalise ts_rank to the best match in this candidate pool (best = 1.0).
    # Raw ts_rank values are small (typically < 0.1) and not comparable across
    # queries; relative normalisation makes the hybrid threshold meaningful.
    best = max(score for _, score in scored) or 1.0
    if best <= 0:
        return []
    return [(chunk, score / best) for chunk, score in scored if score > 0]
