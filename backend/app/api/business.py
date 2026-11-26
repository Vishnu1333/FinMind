from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from .. import models, db as database
from ..dependencies import get_current_user

router = APIRouter()

class BusinessResponse(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        orm_mode = True

@router.get('/business', response_model=BusinessResponse)
def get_business(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    business = db.query(models.Business).filter(models.Business.id == current_user.business_id).first()
    if not business:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Business not found")
    return business
