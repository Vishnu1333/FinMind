from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..dependencies import get_current_user
from .. import models, db as database
from ..services.financial_service import FinancialService
from pydantic import BaseModel, Field
from ..config import CURRENCY

router = APIRouter()

# ----- Response models -----

class AccountBalance(BaseModel):
    account_id: int
    name: str
    balance: Decimal

    model_config = {"from_attributes": True}

class SummaryResponse(BaseModel):
    currency: str = CURRENCY
    accounts: List[AccountBalance]
    total_balance: Decimal
    total_income: Decimal
    total_expenses: Decimal
    net_cash_flow: Decimal
    transaction_count: Optional[int] = None

class CashFlowResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    income: Decimal
    expenses: Decimal
    net: Decimal

class CategorySpending(BaseModel):
    category_id: int
    category_name: str
    amount: Decimal
    percentage: Decimal

class SpendingResponse(BaseModel):
    currency: str = CURRENCY
    start: datetime
    end: datetime
    breakdown: List[CategorySpending]
    total: Decimal

class RunwayResult(BaseModel):
    months: Optional[Decimal] = None
    insufficient_data: bool = False
    message: Optional[str] = None

# ----- Helper -----

def get_service(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    return FinancialService(db, user.business_id)

# ----- Endpoints -----

@router.get("/summary", response_model=SummaryResponse)
def analytics_summary(
    days: Optional[int] = Query(None, ge=1, le=365),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    service: FinancialService = Depends(get_service),
):
    if days is not None and start is None:
        start = datetime.utcnow() - timedelta(days=days)
    accounts = service.get_account_balances(start, end)
    total_balance = sum(a["balance"] for a in accounts)
    income = service.get_income(start, end)
    expenses = service.get_expenses(start, end)
    net = service.get_net_cash_flow(start, end)
    tx_count = (
        service.db.query(func.count(models.Transaction.id))
        .join(models.Account, models.Transaction.account_id == models.Account.id)
        .filter(models.Account.business_id == service.business_id)
        .scalar()
    ) or 0
    return SummaryResponse(
        accounts=[AccountBalance(**a) for a in accounts],
        total_balance=total_balance,
        total_income=income,
        total_expenses=expenses,
        net_cash_flow=net,
        transaction_count=tx_count,
    )

@router.get("/cash-flow", response_model=CashFlowResponse)
def analytics_cash_flow(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    service: FinancialService = Depends(get_service),
):
    start_dt, end_dt = service._resolve_date_range(start, end)
    income = service.get_income(start_dt, end_dt)
    expenses = service.get_expenses(start_dt, end_dt)
    net = service.get_net_cash_flow(start_dt, end_dt)
    return CashFlowResponse(
        period_start=start_dt,
        period_end=end_dt,
        income=income,
        expenses=expenses,
        net=net,
    )

@router.get("/spending", response_model=SpendingResponse)
def analytics_spending(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    service: FinancialService = Depends(get_service),
):
    start_dt, end_dt = service._resolve_date_range(start, end)
    breakdown = service.spending_by_category(start_dt, end_dt)
    total = sum(item["amount"] for item in breakdown)
    return SpendingResponse(
        start=start_dt,
        end=end_dt,
        breakdown=[CategorySpending(**b) for b in breakdown],
        total=total,
    )

@router.get("/runway", response_model=RunwayResult)
def analytics_runway(service: FinancialService = Depends(get_service)):
    return service.cash_runway()

@router.get("/accounts", response_model=List[AccountBalance])
def analytics_accounts(
    service: FinancialService = Depends(get_service),
):
    accounts = service.get_account_balances()
    return [AccountBalance(**a) for a in accounts]


class DailyPoint(BaseModel):
    date: str
    income: Decimal
    expenses: Decimal
    net: Decimal


class DailyCashFlowResponse(BaseModel):
    currency: str = CURRENCY
    days: int
    series: List[DailyPoint]


@router.get("/cash-flow/daily", response_model=DailyCashFlowResponse)
def analytics_cash_flow_daily(
    days: int = Query(90, ge=7, le=365),
    service: FinancialService = Depends(get_service),
):
    series = service.daily_cash_flow(days=days)
    return DailyCashFlowResponse(
        days=days,
        series=[DailyPoint(**p) for p in series],
    )


class TrendPoint(BaseModel):
    week: str
    income: Decimal
    expenses: Decimal
    net: Decimal


class TrendResponse(BaseModel):
    currency: str = CURRENCY
    weeks: int
    series: List[TrendPoint]


@router.get("/trends", response_model=TrendResponse)
def analytics_trends(
    weeks: int = Query(12, ge=4, le=52),
    service: FinancialService = Depends(get_service),
):
    series = service.weekly_trend(weeks=weeks)
    return TrendResponse(weeks=weeks, series=[TrendPoint(**p) for p in series])


class MerchantSpend(BaseModel):
    merchant: str
    total: Decimal
    count: int


class MerchantsResponse(BaseModel):
    currency: str = CURRENCY
    merchants: List[MerchantSpend]


@router.get("/merchants", response_model=MerchantsResponse)
def analytics_merchants(
    limit: int = Query(10, ge=1, le=50),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    service: FinancialService = Depends(get_service),
):
    rows = service.top_merchants(limit=limit, start=start, end=end)
    return MerchantsResponse(merchants=[MerchantSpend(**r) for r in rows])
