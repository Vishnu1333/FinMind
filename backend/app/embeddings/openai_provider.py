# backend/app/embeddings/openai_provider.py
"""OpenAI implementation of the embedding provider.

The provider reads the ``EMBEDDING_API_KEY`` environment variable and uses the
``openai`` Python SDK to request embeddings.  It validates that the returned
vectors have the expected dimension (1536) and raises ``EmbeddingProviderError``
on failure.
"""

from __future__ import annotations

import os
from typing import List

# openai SDK is imported lazily in the provider

from .provider import EmbeddingProvider, EmbeddingProviderError
from .validation import validate_vector


class OpenAIEmbeddingProvider:
    """Concrete embedding provider using OpenAI's ``embeddings.create`` API.

    The class adheres to the :class:`EmbeddingProvider` protocol.  It can embed
    a single string via :meth:`embed_text` or a batch of strings via
    :meth:`embed_documents`.
    """

    def __init__(self) -> None:
        api_key = os.getenv("EMBEDDING_API_KEY")
        if not api_key:
            raise EmbeddingProviderError(
                "EMBEDDING_API_KEY environment variable is required for OpenAI embeddings"
            )
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise EmbeddingProviderError(
                "OpenAI SDK not installed. Install the 'openai' package to use this provider."
            ) from exc
        self._openai = openai
        self._openai.api_key = api_key
    def embed_text(self, text: str) -> List[float]:
        """Return a single embedding vector for *text*.

        The OpenAI API returns a list of objects; we extract the ``embedding``
        attribute from the first (and only) result.
        """
        response = self._openai.embeddings.create(input=text, model="text-embedding-ada-002")
        if not response.data:
            raise EmbeddingProviderError("OpenAI returned empty embedding data")
        vector = list(response.data[0].embedding)
        validate_vector(vector)
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of *texts*.

        The OpenAI API supports batch embedding.  The result is a list of vectors
        matching the input order.
        """
        if not texts:
            return []
        response = self._openai.embeddings.create(input=texts, model="text-embedding-ada-002")
        vectors: List[List[float]] = []
        for item in response.data:
            vec = list(item.embedding)
            validate_vector(vec)
            vectors.append(vec)
        return vectors
