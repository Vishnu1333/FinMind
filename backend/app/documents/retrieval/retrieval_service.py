# backend/app/documents/retrieval/retrieval_service.py
"""Orchestrates retrieval workflow.

The service obtains a query embedding via an EmbeddingProvider (mocked in tests),
executes vector and lexical searches, merges results with the hybrid retriever,
validates evidence, and returns a result suitable for downstream RAG.
"""

from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from backend.app.embeddings.provider import EmbeddingProvider
from backend.app.documents.retrieval.vector import vector_search
from backend.app.documents.retrieval.lexical import lexical_search
from backend.app.documents.retrieval.hybrid_retriever import hybrid_retrieve
from backend.app.documents.retrieval.models import ChunkResult
from backend.app.documents.retrieval.evidence_validator import EvidenceValidator


class RetrievalResult:
    """Container for retrieval output.

    Attributes:
        chunks: List of ChunkResult objects that passed validation.
        insufficient_evidence: True when no chunk met the relevance threshold.
    """

    def __init__(self, chunks: List[ChunkResult], insufficient_evidence: bool = False):
        self.chunks = chunks
        self.insufficient_evidence = insufficient_evidence

    def dict(self) -> Dict[str, Any]:
        return {
            "chunks": [c.dict() for c in self.chunks],
            "insufficient_evidence": self.insufficient_evidence,
        }


class RetrievalService:
    """High‑level retrieval service.

    Usage example (in production code)::

        service = RetrievalService(db_session, embedding_provider)
        result = service.retrieve(
            query="financial risk analysis",
            business_id=1,
            top_k=5,
        )
    """

    def __init__(self, db: Session, embedding_provider: EmbeddingProvider):
        self.db = db
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        business_id: int,
        *,
        document_id: Optional[int] = None,
        document_type: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Execute full retrieval pipeline.

        Args:
            query: User query text.
            business_id: Required business identifier – enforced in SQL.
            document_id: Optional filter on Document.id.
            document_type: Optional filter on Document.document_type.
            source: Optional filter on Document.source.
            top_k: Number of final chunks to return (default 5).

        Returns:
            RetrievalResult containing validated chunks or an insufficient‑evidence flag.
        """
        if not business_id:
            raise ValueError("business_id is required for retrieval")

        # ------------------------------------------------------------------
        # 1. Embedding generation (mocked in tests)
        # ------------------------------------------------------------------
        query_vector = self.embedding_provider.embed_text(query)

        # Build filter dict for both vector and lexical searches
        filters: Dict[str, Any] = {}
        if document_id is not None:
            filters["document_id"] = document_id
        if document_type is not None:
            filters["document_type"] = document_type
        if source is not None:
            filters["source"] = source

        # ------------------------------------------------------------------
        # 2. Vector and lexical candidate retrieval (raw candidate pools)
        # ------------------------------------------------------------------
        # Vector search is only used with semantic embeddings. The deterministic
        # local fallback provider carries no meaning, and presenting its
        # similarity as relevance would be dishonest – lexical ranking drives
        # retrieval in that case.
        vector_candidates: List = []
        from ...embeddings.provider_factory import get_embedding_provider
        _, is_semantic = get_embedding_provider()
        if is_semantic:
            vector_candidates = vector_search(
                self.db, query_vector, business_id, top_k=None, filters=filters
            )
        lexical_candidates = lexical_search(
            self.db, query, business_id, top_k=None, filters=filters
        )

        # ------------------------------------------------------------------
        # 3. Hybrid merging and final top‑k selection
        # ------------------------------------------------------------------
        hybrid_chunks: List[ChunkResult] = hybrid_retrieve(
            vector_candidates, lexical_candidates, top_k=top_k
        )

        # ------------------------------------------------------------------
        # 4. Evidence validation – ensure each chunk is trustworthy
        # ------------------------------------------------------------------
        valid_chunks: List[ChunkResult] = []
        for chunk in hybrid_chunks:
            try:
                EvidenceValidator.validate(
                    db=self.db,
                    chunk_id=chunk.chunk_id,
                    business_id=business_id,
                )
                valid_chunks.append(chunk)
            except Exception:
                # Skip invalid evidence; do not raise – we may still have enough.
                continue

        insufficient = len(valid_chunks) == 0
        return RetrievalResult(chunks=valid_chunks, insufficient_evidence=insufficient)
