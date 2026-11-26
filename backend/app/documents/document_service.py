import os
import hashlib
from typing import Optional

from sqlalchemy.orm import Session

from ..models import business_models, document_models as models
from ..config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentService:
    """Service handling document lifecycle and business isolation."""

    def __init__(self, db: Session, business_id: int):
        self.db = db
        self.business_id = business_id

    def _hash_content(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def create_document(self, filename: str, content: bytes, source: Optional[str] = None) -> models.Document:
        """Create a new document record.

        - Calculates SHA‑256 hash of the raw content.
        - Prevents duplicate documents within the same business.
        - Sets initial status to ``pending``.
        """
        content_hash = self._hash_content(content)
        # duplicate check
        existing = (
            self.db.query(models.Document)
            .filter(models.Document.business_id == self.business_id, models.Document.content_hash == content_hash)
            .first()
        )
        if existing:
            raise ValueError("Duplicate document for this business")
        doc = models.Document(
            business_id=self.business_id,
            filename=filename,
            document_type="unknown",
            source=source,
            content_hash=content_hash,
            status="pending",
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_document(self, document_id: int) -> models.Document:
        """Retrieve a document ensuring it belongs to the service's business."""
        doc = (
            self.db.query(models.Document)
            .filter(models.Document.id == document_id, models.Document.business_id == self.business_id)
            .first()
        )
        if not doc:
            raise ValueError("Document not found or access denied")
        return doc

    def update_status(self, document_id: int, status: str, failure_reason: Optional[str] = None) -> None:
        doc = self.get_document(document_id)
        doc.status = status
        if failure_reason:
            doc.failure_reason = failure_reason[:500]
        self.db.commit()
