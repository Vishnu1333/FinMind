"""Embedding provider factory.

Returns the configured embedding provider:

* ``OpenAIEmbeddingProvider`` when ``EMBEDDING_API_KEY`` is set (real
  semantic embeddings), or
* a deterministic local hashing provider otherwise. The local provider
  produces stable vectors but carries **no semantic meaning**, so callers
  must treat vector search as unavailable in that case (``is_real=False``)
  instead of presenting meaningless similarity as relevance.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple

from .provider import EmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider

_DIMENSIONS = 1536


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic content-hash embedding (no semantic meaning)."""

    def embed_text(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [float(b) for b in digest * (_DIMENSIONS // 32)]
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]


def get_embedding_provider() -> Tuple[EmbeddingProvider, bool]:
    """Return ``(provider, is_semantic)``.

    ``is_semantic`` is False when the provider is the local hashing fallback,
    in which case vector similarity must not be used for relevance ranking.
    """
    if os.getenv("EMBEDDING_API_KEY"):
        return OpenAIEmbeddingProvider(), True
    return LocalHashingEmbeddingProvider(), False
