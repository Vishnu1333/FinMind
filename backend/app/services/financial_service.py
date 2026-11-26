"""Financial intelligence service layer for FinMind.

All calculations are performed directly from the authoritative PostgreSQL tables.
No hard‑coded values are used. Monetary arithmetic uses :class:`decimal.Decimal`.
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func

from .. import models
from ..config import CURRENCY

# Helper to ensure Decimal quantization to two decimal places
def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class FinancialService:
    """Service providing deterministic financial calculations.

    All methods expect a valid SQLAlchemy ``Session`` and the ``business_id``
    derived from the authenticated user.
    """

    def __init__(self, db: Session, business_id: int):
        self.db = db
        self.business_id = business_id

    # ---------------------------------------------------------------------
    # Date handling helpers
    # ---------------------------------------------------------------------
    def _resolve_date_range(self, start: Optional[datetime], end: Optional[datetime]) -> Tuple[datetime, datetime]:
        """Return inclusive start and exclusive end datetimes.

        * If *start* is ``None`` – use the earliest transaction timestamp for the business.
        * If *end* is ``None`` – use the current UTC time.
        """
        if start is None:
            earliest = (
                self.db.query(func.min(models.Transaction.timestamp))
                .join(models.Account, models.Transaction.account_id == models.Account.id)
                .filter(models.Account.business_id == self.business_id)
                .scalar()
            )
            start = earliest or datetime.utcnow()
        if end is None:
            end = datetime.utcnow()
        return start, end

    # ---------------------------------------------------------------------
    # Account balances
    # ---------------------------------------------------------------------
    def get_account_balances(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[dict]:
        """Return a list of accounts with their *authoritative* balance.

        The authoritative balance is taken from the latest ``LedgerEntry`` per
        account. If no ledger entry exists for an account we fall back to the
        cached ``Account.balance`` column.
        """
        # Subquery: the single latest ledger entry per account. A plain
        # max(entry_date) join can fan out when several entries share the
        # same date, so we rank deterministically by (entry_date, id).
        ranked = (
            self.db.query(
                models.LedgerEntry.account_id.label("account_id"),
                models.LedgerEntry.running_balance.label("running_balance"),
                func.row_number()
                .over(
                    partition_by=models.LedgerEntry.account_id,
                    order_by=(models.LedgerEntry.entry_date.desc(), models.LedgerEntry.id.desc()),
                )
                .label("rnk"),
            )
            .join(models.Account, models.LedgerEntry.account_id == models.Account.id)
            .filter(models.Account.business_id == self.business_id)
            .subquery()
        )
        latest_entries = self.db.query(ranked).filter(ranked.c.rnk == 1).subquery()
        # Join accounts with latest ledger (if any)
        query = (
            self.db.query(
                models.Account.id.label("account_id"),
                models.Account.name,
                func.coalesce(latest_entries.c.running_balance, models.Account.balance).label("balance"),
            )
            .outerjoin(latest_entries, models.Account.id == latest_entries.c.account_id)
            .filter(models.Account.business_id == self.business_id)
        )
        results = []
        for row in query.all():
            results.append({
                "account_id": row.account_id,
                "name": row.name,
                "balance": _quantize(row.balance),
            })
        return results

    def _total_balance(self) -> Decimal:
        balances = self.get_account_balances()
        total = sum(item["balance"] for item in balances)
        return _quantize(total)

    # ---------------------------------------------------------------------
    # Income / Expenses
    # ---------------------------------------------------------------------
    def _income_expenses(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Tuple[Decimal, Decimal]:
        start_dt, end_dt = self._resolve_date_range(start, end)
        base_query = (
            self.db.query(models.Transaction.type, func.sum(models.Transaction.amount).label("total"))
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(
                models.Account.business_id == self.business_id,
                models.Transaction.timestamp >= start_dt,
                models.Transaction.timestamp < end_dt,
            )
            .group_by(models.Transaction.type)
        )
        income = Decimal("0")
        expenses = Decimal("0")
        for typ, total in base_query:
            if total is None:
                continue
            if typ == models.TransactionType.CREDIT:
                income = Decimal(total)
            elif typ == models.TransactionType.DEBIT:
                expenses = Decimal(total)
        return _quantize(income), _quantize(expenses)

    def get_income(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Decimal:
        inc, _ = self._income_expenses(start, end)
        return inc

    def get_expenses(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Decimal:
        _, exp = self._income_expenses(start, end)
        return exp

    def get_net_cash_flow(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Decimal:
        inc, exp = self._income_expenses(start, end)
        return _quantize(inc - exp)

    # ---------------------------------------------------------------------
    # Spending by Category (expenses only)
    # ---------------------------------------------------------------------
    def spending_by_category(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[dict]:
        start_dt, end_dt = self._resolve_date_range(start, end)
        total_expenses = self.get_expenses(start_dt, end_dt)
        if total_expenses == Decimal("0"):
            return []
        query = (
            self.db.query(models.Category.id.label("category_id"), models.Category.name.label("category_name"), func.sum(models.Transaction.amount).label("amount"))
            .join(models.Transaction, models.Transaction.category_id == models.Category.id)
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(
                models.Account.business_id == self.business_id,
                models.Transaction.type == models.TransactionType.DEBIT,
                models.Transaction.timestamp >= start_dt,
                models.Transaction.timestamp < end_dt,
            )
            .group_by(models.Category.id, models.Category.name)
        )
        results = []
        for row in query.all():
            amount = _quantize(row.amount)
            percentage = _quantize((amount / total_expenses) * Decimal("100"))
            results.append({
                "category_id": row.category_id,
                "category_name": row.category_name,
                "amount": amount,
                "percentage": percentage,
            })
        return results

    # ---------------------------------------------------------------------
    # Daily / weekly series and merchant analytics
    # ---------------------------------------------------------------------
    def daily_cash_flow(self, days: int = 90) -> List[dict]:
        """Daily income, expense and net totals over the last N days (real rows)."""
        start = datetime.utcnow() - timedelta(days=int(days))
        rows = (
            self.db.query(
                func.date(models.Transaction.timestamp).label("day"),
                models.Transaction.type,
                func.sum(models.Transaction.amount).label("total"),
            )
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(
                models.Account.business_id == self.business_id,
                models.Transaction.timestamp >= start,
            )
            .group_by("day", models.Transaction.type)
            .all()
        )
        series: dict = {}
        for day, typ, total in rows:
            entry = series.setdefault(str(day), {"date": str(day), "income": Decimal("0"), "expenses": Decimal("0")})
            if typ == models.TransactionType.CREDIT:
                entry["income"] += Decimal(total or 0)
            else:
                entry["expenses"] += Decimal(total or 0)
        result = []
        for entry in sorted(series.values(), key=lambda e: e["date"]):
            result.append({
                "date": entry["date"],
                "income": _quantize(entry["income"]),
                "expenses": _quantize(entry["expenses"]),
                "net": _quantize(entry["income"] - entry["expenses"]),
            })
        return result

    def weekly_trend(self, weeks: int = 12) -> List[dict]:
        """Weekly income/expense/net trend (real rows)."""
        start = datetime.utcnow() - timedelta(weeks=int(weeks))
        rows = (
            self.db.query(
                func.date_trunc('week', models.Transaction.timestamp).label("week"),
                models.Transaction.type,
                func.sum(models.Transaction.amount).label("total"),
            )
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(
                models.Account.business_id == self.business_id,
                models.Transaction.timestamp >= start,
            )
            .group_by("week", models.Transaction.type)
            .all()
        )
        series: dict = {}
        for week, typ, total in rows:
            key = week.date().isoformat()
            entry = series.setdefault(key, {"week": key, "income": Decimal("0"), "expenses": Decimal("0")})
            if typ == models.TransactionType.CREDIT:
                entry["income"] += Decimal(total or 0)
            else:
                entry["expenses"] += Decimal(total or 0)
        result = []
        for entry in sorted(series.values(), key=lambda e: e["week"]):
            result.append({
                "week": entry["week"],
                "income": _quantize(entry["income"]),
                "expenses": _quantize(entry["expenses"]),
                "net": _quantize(entry["income"] - entry["expenses"]),
            })
        return result

    def top_merchants(self, limit: int = 10, start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[dict]:
        """Top merchants by total spend within a period (real rows)."""
        start_dt, end_dt = self._resolve_date_range(start, end)
        merchant = func.coalesce(models.Transaction.merchant, models.Transaction.description)
        rows = (
            self.db.query(
                merchant.label("merchant"),
                func.sum(models.Transaction.amount).label("total"),
                func.count(models.Transaction.id).label("count"),
            )
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(
                models.Account.business_id == self.business_id,
                models.Transaction.type == models.TransactionType.DEBIT,
                models.Transaction.timestamp >= start_dt,
                models.Transaction.timestamp < end_dt,
            )
            .group_by(merchant)
            .order_by(func.sum(models.Transaction.amount).desc())
            .limit(int(limit))
            .all()
        )
        return [
            {
                "merchant": row.merchant or "Unnamed",
                "total": _quantize(row.total),
                "count": int(row.count),
            }
            for row in rows
        ]
    def cash_runway(self) -> dict:
        """Calculate runway in months.

        Returns a dict compatible with the ``RunwayResult`` schema.
        """
        current_balance = self._total_balance()
        # Determine expense per full calendar month
        expense_query = (
            self.db.query(
                func.date_trunc('month', models.Transaction.timestamp).label('month'),
                func.sum(models.Transaction.amount).label('monthly_expense'),
            )
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(
                models.Account.business_id == self.business_id,
                models.Transaction.type == models.TransactionType.DEBIT,
            )
            .group_by('month')
            .order_by('month')
        )
        monthly_expenses = [Decimal(row.monthly_expense) for row in expense_query if row.monthly_expense is not None]
        if len(monthly_expenses) < 1:
            return {"months": None, "insufficient_data": True, "message": "Not enough expense history to compute runway."}
        avg_monthly_expense = sum(monthly_expenses) / Decimal(len(monthly_expenses))
        if avg_monthly_expense == Decimal("0"):
            return {"months": None, "insufficient_data": True, "message": "Average monthly expense is zero."}
        runway_months = _quantize(current_balance / avg_monthly_expense)
        return {"months": runway_months, "insufficient_data": False, "message": None}
