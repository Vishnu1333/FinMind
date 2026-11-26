# backend/app/embeddings/validation.py
"""Utility for validating embedding vectors.

The provider is expected to return a list of ``float`` with a fixed dimension
(1536 for OpenAI ``text-embedding-ada-002``).  The tests use this helper to
ensure that incorrect dimensions raise a ``ValueError``.
"""

from __future__ import annotations

from typing import List

EXPECTED_DIMENSION = 1536


def validate_vector(vector: List[float]) -> None:
    """Validate that *vector* has the expected dimension and contains only finite numbers.

    Raises
    ------
    ValueError
        If the vector length does not match ``EXPECTED_DIMENSION`` or contains non‑numeric,
        NaN, or infinite values.
    """
    if len(vector) != EXPECTED_DIMENSION:
        raise ValueError(f"Embedding vector must have dimension {EXPECTED_DIMENSION}, got {len(vector)}")
    # Ensure all elements are real numbers and finite
    for i, v in enumerate(vector):
        if not isinstance(v, (int, float)):
            raise ValueError(f"Embedding vector element at index {i} is not numeric: {v!r}")
        if isinstance(v, float):
            if v != v:  # NaN check
                raise ValueError(f"Embedding vector element at index {i} is NaN")
            if v in (float('inf'), float('-inf')):
                raise ValueError(f"Embedding vector element at index {i} is infinite")
