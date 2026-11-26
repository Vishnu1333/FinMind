# backend/app/documents/retrieval/evidence_validator.py
"""Evidence validation utilities.

Ensures that a retrieved DocumentChunk and its parent Document satisfy a set of
integrity rules before being returned to downstream components.
"""

from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.models.document_models import DocumentChunk, Document
from backend.app.documents.evidence.exceptions import EvidenceValidatorError


class EvidenceValidator:
    """Validate that a chunk belongs to the requested business and contains
    well‑formed data.
    """

    @staticmethod
    def validate(db: Session, *, chunk_id: int, business_id: int) -> None:
        """Validate a single chunk.

        Raises ``EvidenceValidatorError`` if any check fails.
        """
        # Fetch chunk and its document in a single query for efficiency.
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id == chunk_id)
        )
        result = db.execute(stmt).first()
        if result is None:
            raise EvidenceValidatorError(f"Chunk with id {chunk_id} does not exist")

        chunk: DocumentChunk = result[0]
        document: Document = result[1]

        # Verify relationship consistency.
        if chunk.document_id != document.id:
            raise EvidenceValidatorError(
                f"Chunk {chunk_id} document_id {chunk.document_id} does not match Document.id {document.id}"
            )

        # Business ownership.
        if document.business_id != business_id:
            raise EvidenceValidatorError(
                f"Document {document.id} belongs to business {document.business_id}, not requested {business_id}"
            )

        # Non‑empty text.
        if not chunk.text or not chunk.text.strip():
            raise EvidenceValidatorError(f"Chunk {chunk_id} has empty text")

        # Metadata must be a dict if present.
        meta = chunk.metadata_json
        if meta is not None and not isinstance(meta, dict):
            raise EvidenceValidatorError(f"Chunk {chunk_id} metadata is not a dict")

        # source_pages validation.
        if isinstance(meta, dict) and "source_pages" in meta:
            pages = meta["source_pages"]
            if not isinstance(pages, list) or not all(isinstance(p, int) and p > 0 for p in pages):
                raise EvidenceValidatorError(
                    f"Chunk {chunk_id} metadata.source_pages must be a list of positive integers"
                )
        # All checks passed – function returns None.
        return None
