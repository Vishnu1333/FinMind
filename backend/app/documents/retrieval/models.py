# backend/app/documents/retrieval/models.py

"""Pydantic models for retrieval results."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class ChunkResult(BaseModel):
    chunk_id: int
    document_id: int
    text: str
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    score: float
    filename: str
    document_type: str
    source: Optional[str] = None
    source_pages: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None
