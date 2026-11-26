from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from .. import models, db as database
from ..auth_utils import verify_password
from ..dependencies import get_current_user

router = APIRouter()

# Response schemas
from pydantic import BaseModel
from datetime import datetime

class AccountResponse(BaseModel):
    id: int
    name: str
    balance: float
    business_id: int
    created_at: datetime

    class Config:
        from_attributes = True

@router.get("/accounts", response_model=List[AccountResponse])
def list_accounts(current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    accounts = db.query(models.Account).filter(models.Account.business_id == current_user.business_id).all()
    return accounts

@router.get("/accounts/{account_id}", response_model=AccountResponse)
def get_account(account_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    account = (
        db.query(models.Account)
        .filter(models.Account.id == account_id, models.Account.business_id == current_user.business_id)
        .first()
    )
    if not account:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")
    return account
