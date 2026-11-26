from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import pydantic

from .. import db as database, models
from ..dependencies import get_current_user
from ..documents.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(pydantic.BaseModel):
    id: int
    filename: str
    status: str
    document_type: str

    class Config:
        from_attributes = True


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    source: Optional[str] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    """Upload a document and run the full intelligence pipeline:

    extraction -> classification -> chunking (page-preserving) -> embeddings.
    """
    if not current_user.business_id:
        raise HTTPException(status_code=403, detail="User not associated with a business")

    doc_service = DocumentService(db, current_user.business_id)
    content = await file.read()

    try:
        doc = doc_service.create_document(filename=file.filename, content=content, source=source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # 1. Extraction (validated, page-aware)
        from ..documents.extraction.extraction_service import extract_document
        extracted = extract_document(file.filename, content)

        # 2. Classification (deterministic rules)
        from ..documents.classification.simple_classifier import classify_document
        classification = classify_document(extracted["text"], filename=file.filename)
        doc.document_type = classification.get("document_type", "other")

        # 3. Chunking (deterministic, page-preserving)
        from ..documents.chunking.character_chunker import chunk_document
        chunks = chunk_document(doc.id, extracted["pages"])

        # 4. Embeddings + persistence
        import os
        from ..embeddings.openai_provider import OpenAIEmbeddingProvider
        from ..embeddings.provider import EmbeddingProvider

        if os.getenv("EMBEDDING_API_KEY"):
            provider = OpenAIEmbeddingProvider()
        else:
            class LocalEmbeddingProvider(EmbeddingProvider):
                """Deterministic local embedding used when no embedding API is
                configured. Vector search degrades gracefully; lexical and
                hybrid retrieval remain fully functional."""

                def embed_text(self, text: str):
                    import hashlib

                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    vector = [float(b) for b in digest * 96]  # 1536 dims
                    norm = sum(v * v for v in vector) ** 0.5 or 1.0
                    return [v / norm for v in vector]

            provider = LocalEmbeddingProvider()

        for chunk in chunks:
            embedding = provider.embed_text(chunk["text"])
            db.add(
                models.DocumentChunk(
                    document_id=doc.id,
                    chunk_number=chunk["chunk_number"],
                    text=chunk["text"],
                    embedding=embedding,
                    metadata={
                        "format": extracted["format"],
                        "document_type": doc.document_type,
                        # 1-based page numbers for user display and evidence
                        # validation (extraction uses 0-based indices).
                        "source_pages": [p + 1 for p in chunk.get("source_pages", [])],
                    },
                )
            )

        doc_service.update_status(doc.id, "processed")
        db.commit()
        db.refresh(doc)
    except HTTPException:
        raise
    except Exception as e:
        doc_service.update_status(doc.id, "error", failure_reason=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return doc


@router.get("", response_model=List[DocumentResponse])
@router.get("/", response_model=List[DocumentResponse], include_in_schema=False)
def get_documents(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    if not current_user.business_id:
        raise HTTPException(status_code=403, detail="User not associated with a business")

    docs = db.query(models.Document).filter(models.Document.business_id == current_user.business_id).all()
    return docs
