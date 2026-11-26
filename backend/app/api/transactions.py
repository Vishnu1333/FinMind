from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from .. import models, db as database
from ..dependencies import get_current_user
from pydantic import BaseModel

router = APIRouter()

class TransactionResponse(BaseModel):
    id: int
    account_id: int
    amount: float
    description: Optional[str]
    merchant: Optional[str]
    timestamp: datetime
    type: str
    category_id: Optional[int]
    category_name: Optional[str]

    class Config:
        from_attributes = True


def _to_response(t: models.Transaction) -> TransactionResponse:
    return TransactionResponse(
        id=t.id,
        account_id=t.account_id,
        amount=float(t.amount),
        description=t.description,
        merchant=t.merchant,
        timestamp=t.timestamp,
        type=t.type,
        category_id=t.category_id,
        category_name=t.category.name if t.category else None,
    )


@router.get("/transactions", response_model=List[TransactionResponse])
def list_transactions(
    account_id: Optional[int] = Query(None),
    category_id: Optional[int] = Query(None),
    type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    query = (
        db.query(models.Transaction)
        .join(models.Account)
        .filter(models.Account.business_id == current_user.business_id)
    )
    if account_id is not None:
        query = query.filter(models.Transaction.account_id == account_id)
    if category_id is not None:
        query = query.filter(models.Transaction.category_id == category_id)
    if type is not None:
        query = query.filter(models.Transaction.type == type)
    if search is not None and search.strip():
        like = f"%{search.strip().lower()}%"
        query = query.filter(
            (models.Transaction.description.ilike(like)) | (models.Transaction.merchant.ilike(like))
        )
    if min_amount is not None:
        query = query.filter(models.Transaction.amount >= min_amount)
    if max_amount is not None:
        query = query.filter(models.Transaction.amount <= max_amount)
    if start_date is not None:
        query = query.filter(models.Transaction.timestamp >= start_date)
    if end_date is not None:
        query = query.filter(models.Transaction.timestamp <= end_date)
    transactions = (
        query.order_by(models.Transaction.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [_to_response(t) for t in transactions]


@router.get("/transactions/{transaction_id}", response_model=TransactionResponse)
def get_transaction(
    transaction_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    transaction = (
        db.query(models.Transaction)
        .join(models.Account)
        .filter(
            models.Transaction.id == transaction_id,
            models.Account.business_id == current_user.business_id,
        )
        .first()
    )
    if not transaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    return _to_response(transaction)
