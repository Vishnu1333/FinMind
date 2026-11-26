from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, db as database
from ..dependencies import get_current_user
from pydantic import BaseModel

router = APIRouter()

class CategoryResponse(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True

@router.get('/categories', response_model=List[CategoryResponse])
def list_categories(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    # Categories are global; optionally could filter by business if needed.
    categories = db.query(models.Category).all()
    return categories
