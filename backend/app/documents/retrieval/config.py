# backend/app/documents/retrieval/config.py

"""Configuration constants for retrieval.

These can be overridden via environment variables if needed.
"""

import os
from dataclasses import dataclass

# Default constants
VECTOR_TOP_K = int(os.getenv('VECTOR_TOP_K', '5'))  # final top_k default
VECTOR_WEIGHT = float(os.getenv('VECTOR_WEIGHT', '0.6'))  # weight for vector score in hybrid
LEXICAL_WEIGHT = float(os.getenv('LEXICAL_WEIGHT', '0.4'))  # weight for lexical score in hybrid
RELEVANCE_THRESHOLD = float(os.getenv('RELEVANCE_THRESHOLD', '0.30'))  # minimum hybrid_score to be considered sufficient

# Candidate pool sizes for initial retrieval before hybrid merging
VECTOR_CANDIDATE_POOL = int(os.getenv('VECTOR_CANDIDATE_POOL', '20'))
LEXICAL_CANDIDATE_POOL = int(os.getenv('LEXICAL_CANDIDATE_POOL', '20'))

@dataclass
class RetrievalConfig:
    VECTOR_TOP_K: int = VECTOR_TOP_K
    VECTOR_WEIGHT: float = VECTOR_WEIGHT
    LEXICAL_WEIGHT: float = LEXICAL_WEIGHT
    RELEVANCE_THRESHOLD: float = RELEVANCE_THRESHOLD
    VECTOR_CANDIDATE_POOL: int = VECTOR_CANDIDATE_POOL
    LEXICAL_CANDIDATE_POOL: int = LEXICAL_CANDIDATE_POOL

