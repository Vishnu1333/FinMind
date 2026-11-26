from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Any, Dict
import pydantic

from .. import db as database, models
from ..dependencies import get_current_user
from ..documents.rag.rag_service import answer_question, RAGError

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

class QueryRequest(pydantic.BaseModel):
    query: str
    document_id: Optional[int] = None
    document_type: Optional[str] = None
    source: Optional[str] = None
    top_k: int = 5
    privacy_sensitive: bool = False

class SourceResponse(pydantic.BaseModel):
    document_id: int
    chunk_id: int
    filename: Optional[str] = None
    document_type: Optional[str] = None
    source_pages: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(pydantic.BaseModel):
    grounded: bool
    answer: str
    sources: List[SourceResponse]
    error: Optional[str] = None

@router.post("/query", response_model=QueryResponse)
def query_knowledge(
    request: QueryRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    if not current_user.business_id:
        raise HTTPException(status_code=403, detail="User not associated with a business")
        
    try:
        result = answer_question(
            question=request.query,
            business_id=current_user.business_id,
            document_id=request.document_id,
            document_type=request.document_type,
            source=request.source,
            top_k=request.top_k,
            privacy_sensitive=request.privacy_sensitive
        )
        return QueryResponse(**result)
    except RAGError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during RAG query")
