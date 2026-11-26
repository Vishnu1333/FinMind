# backend/app/embeddings/provider.py
"""Embedding provider abstraction.

Defines the base error class and an abstract provider interface used by the
embedding service.  Concrete implementations (e.g. OpenAI) implement the two
methods below.
"""

from __future__ import annotations

from typing import List, Protocol


class EmbeddingProviderError(Exception):
    """Raised when a provider cannot be instantiated or fails to embed."""


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implementations must provide methods to embed a single piece of text or a
    list of texts.  The return type for each vector is a ``list`` of ``float``
    with a length of 1536 (the OpenAI ada‑002 embedding dimension).
    """

    def embed_text(self, text: str) -> List[float]:
        """Return an embedding vector for a single string."""
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors for the supplied strings."""
        raise NotImplementedError
