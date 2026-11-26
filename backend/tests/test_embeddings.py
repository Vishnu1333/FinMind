import os
import json
from unittest.mock import patch, MagicMock

import pytest

from backend.app.db import engine, Base, SessionLocal
from backend.app.models.document_models import Document, DocumentChunk
from backend.app.embeddings.provider import EmbeddingProviderError
from backend.app.embeddings.openai_provider import OpenAIEmbeddingProvider
from backend.app.embeddings.validation import validate_vector
from backend.app.documents.embeddings.embedding_service import EmbeddingService, AuthorizationError

# Test module fixtures using PostgreSQL test database

@pytest.fixture
def db_session():
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture
def sample_data(db_session):
    # Create a business, document, and chunks
    from backend.app.models.business_models import Business
    business = Business(name="TestBiz")
    db_session.add(business)
    db_session.flush()
    doc = Document(
        business_id=business.id,
        filename="test.pdf",
        document_type="PDF",
        source=None,
        content_hash="hash123",
        status="processed",
    )
    db_session.add(doc)
    db_session.flush()
    chunks = []
    for i, txt in enumerate(["Chunk one", "Chunk two"]):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_number=i,
            text=txt,
            meta={"page": i + 1},
        )
        db_session.add(chunk)
        db_session.flush()
        chunks.append({"id": chunk.id, "text": txt})
    db_session.commit()
    return {
        "business_id": business.id,
        "document_id": doc.id,
        "chunks": chunks,
    }

def test_validate_vector_success():
    vec = [0.0] * 1536
    validate_vector(vec)

def test_validate_vector_wrong_dim():
    vec = [0.0] * 10
    with pytest.raises(ValueError):
        validate_vector(vec)

def test_openai_provider_success(monkeypatch):
    fake_response = MagicMock()
    fake_item = MagicMock()
    fake_item.embedding = [0.1] * 1536
    fake_response.data = [fake_item]
    with patch("openai.embeddings.create", return_value=fake_response) as mock_create:
        # Ensure env var present
        monkeypatch.setenv("EMBEDDING_API_KEY", "testkey")
        provider = OpenAIEmbeddingProvider()
        emb = provider.embed_text("hello")
        assert isinstance(emb, list) and len(emb) == 1536
        mock_create.assert_called_once()

def test_openai_provider_missing_key(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    with pytest.raises(EmbeddingProviderError):
        OpenAIEmbeddingProvider()

def test_embedding_service_success(db_session, sample_data, monkeypatch):
    # Mock provider to return deterministic vectors
    class DummyProvider:
        def embed_documents(self, texts):
            return [[float(i)] * 1536 for i, _ in enumerate(texts)]
    service = EmbeddingService(db_session, DummyProvider())
    service.embed_document_chunks(
        document_id=sample_data["document_id"],
        chunk_dicts=sample_data["chunks"],
        current_business_id=sample_data["business_id"],
    )
    for chunk_info in sample_data["chunks"]:
        chunk = db_session.query(DocumentChunk).filter_by(id=chunk_info["id"]).one()
        # Ensure embedding stored correctly (list or Vector) and has correct dimension
        embedding = chunk.embedding
        # Accept Vector type from pgvector or plain list
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        assert isinstance(embedding, list)
        assert len(embedding) == 1536

def test_embedding_service_cross_business(db_session, sample_data):
    class DummyProvider:
        def embed_documents(self, texts):
            return [[0.0] * 1536 for _ in texts]
    service = EmbeddingService(db_session, DummyProvider())
    with pytest.raises(AuthorizationError):
        service.embed_document_chunks(
            document_id=sample_data["document_id"],
            chunk_dicts=sample_data["chunks"],
            current_business_id=9999,
        )

def test_embedding_service_provider_error(db_session, sample_data):
    class BadProvider:
        def embed_documents(self, texts):
            raise EmbeddingProviderError("boom")
    service = EmbeddingService(db_session, BadProvider())
    with pytest.raises(EmbeddingProviderError):
        service.embed_document_chunks(
            document_id=sample_data["document_id"],
            chunk_dicts=sample_data["chunks"],
            current_business_id=sample_data["business_id"],
        )
    for chunk_info in sample_data["chunks"]:
        chunk = db_session.query(DocumentChunk).filter_by(id=chunk_info["id"]).one()
        assert chunk.embedding is None
