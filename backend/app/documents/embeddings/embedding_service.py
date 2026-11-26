# backend/app/documents/embeddings/embedding_service.py
"""Embedding service that persists embeddings for document chunks.

The service validates that the caller is authorised to embed chunks for the
specified document (i.e. the document belongs to the provided business) and
stores the computed vectors on the corresponding ``DocumentChunk`` rows.
"""

from __future__ import annotations

from typing import List, Dict

from sqlalchemy.orm import Session

from backend.app.models.document_models import Document, DocumentChunk

# Exceptions used by the test suite
class AuthorizationError(Exception):
    """Raised when a business attempts to embed chunks for a document it does not own."""

class EmbeddingService:
    def __init__(self, db_session: Session, provider: object) -> None:
        self.db = db_session
        self.provider = provider

    def embed_document_chunks(self, *, document_id: int, chunk_dicts: List[Dict], current_business_id: int) -> None:
        """Persist embeddings for the given list of chunk dictionaries.

        Parameters
        ----------
        document_id: int
            ID of the document whose chunks are being processed.
        chunk_dicts: List[Dict]
            Each dict must contain ``id`` (the chunk primary key) and ``text``.
        current_business_id: int
            ID of the business performing the operation; used for authorization.
        """
        # Verify document ownership
        document = self.db.query(Document).filter_by(id=document_id).one_or_none()
        if document is None:
            raise ValueError(f"Document with id {document_id} not found")
        if document.business_id != current_business_id:
            raise AuthorizationError("Business does not own the document")

        # Gather texts in the order of chunk_dicts
        texts = [chunk["text"] for chunk in chunk_dicts]
        embeddings = self.provider.embed_documents(texts)
        if len(embeddings) != len(chunk_dicts):
            raise ValueError("Provider returned mismatched number of embeddings")

        # Update each DocumentChunk
        for chunk_info, vector in zip(chunk_dicts, embeddings):
            chunk = self.db.query(DocumentChunk).filter_by(id=chunk_info["id"]).one_or_none()
            if chunk is None:
                continue  # skip missing chunks silently
            chunk.embedding = vector
        self.db.commit()
