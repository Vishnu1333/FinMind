"""Business provisioning service – the simulated banking environment core.

When a new business registers, FinMind provisions the business with:

* a set of global spending categories,
* two starter accounts (operating + savings) with opening balances,
* two virtual business cards,
* a deterministic ~6 month transaction history with ledger entries,
* a welcome notification.

All figures are generated deterministically from a per-business seed so the
environment is reproducible. Every downstream financial calculation (balance,
cash flow, analytics, forecasts, anomalies) reads these rows through the
authoritative FinancialService – nothing is hard-coded at display time.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from .. import models

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Sales",
    "Payroll",
    "Rent",
    "Utilities",
    "Inventory",
    "Supplies",
    "Marketing",
    "Insurance",
    "Transfer",
    "Other",
]

# Deterministic card suffixes for the simulated environment. Card 4821 is the
# primary card used in the product's security demonstration workflow.
CARD_SUFFIXES = ["4821", "9932"]

HISTORY_DAYS = 182  # ~6 months of daily activity

MERCHANTS = {
    "Inventory": ["GoldenCrop Supplies", "Metro Wholesale", "PasarBesar Traders"],
    "Supplies": ["OfficeHub", "Kedai Alat Tulis"],
    "Marketing": ["Meta Ads", "Google Ads", "BillboardPro"],
    "Utilities": ["Tenaga Nasional", "Air Selangor"],
    "Rent": ["Menara Landlord", "Lot 42 Properties"],
    "Insurance": ["Amanah Insurans"],
    "Payroll": ["Staff Payroll"],
    "Sales": ["Retail Sales", "Online Store Sales", "Wholesale Order"],
    "Transfer": ["Internal Transfer"],
    "Other": ["Misc Expense"],
}


def _q(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class ProvisioningService:
    """Provisions a newly registered business with its simulated environment."""

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Categories (global)
    # ------------------------------------------------------------------
    def ensure_categories(self) -> Dict[str, models.Category]:
        categories: Dict[str, models.Category] = {}
        for name in CATEGORIES:
            cat = self.db.query(models.Category).filter(models.Category.name == name).first()
            if not cat:
                cat = models.Category(name=name)
                self.db.add(cat)
                self.db.flush()
            categories[name] = cat
        return categories

    # ------------------------------------------------------------------
    # Provision everything for a business
    # ------------------------------------------------------------------
    def provision(self, business: models.Business, user: models.User) -> None:
        categories = self.ensure_categories()

        # Accounts
        operating = models.Account(
            user_id=user.id,
            business_id=business.id,
            name="Operating Account",
            balance=Decimal("0"),
        )
        savings = models.Account(
            user_id=user.id,
            business_id=business.id,
            name="Business Savings",
            balance=Decimal("0"),
        )
        self.db.add_all([operating, savings])
        self.db.flush()

        # Opening balances (recorded as ledger entries with a null transaction)
        opening_operating = Decimal("12500.00")
        opening_savings = Decimal("48000.00")

        # Cards
        for suffix in CARD_SUFFIXES:
            self.db.add(models.Card(business_id=business.id, last_four=suffix, status="active"))

        # Transaction history
        txs = self._generate_history(business, categories, operating, savings)
        self._record_ledger(operating, savings, txs, opening_operating, opening_savings)

        # Welcome notification
        self.db.add(
            models.Notification(
                business_id=business.id,
                title="Welcome to FinMind",
                message=(
                    "Your simulated banking environment is ready. "
                    "Explore your accounts, cash flow and financial intelligence."
                ),
                read=0,
                severity="info",
            )
        )
        self.db.commit()
        logger.info("Provisioned business %s (%s)", business.id, business.name)

    # ------------------------------------------------------------------
    # History generation
    # ------------------------------------------------------------------
    def _generate_history(
        self,
        business: models.Business,
        categories: Dict[str, models.Category],
        operating: models.Account,
        savings: models.Account,
    ) -> List[models.Transaction]:
        rng = random.Random(100003 + business.id)
        txs: List[models.Transaction] = []
        today = datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0)
        start = today - timedelta(days=HISTORY_DAYS)

        # Weekly recurring date anchors
        d = start
        while d <= today:
            dow = d.weekday()

            # -- Sales: daily with weekday pattern and growth trend
            growth = 1.0 + (HISTORY_DAYS - (today - d).days) / 400.0  # ~+45% over 6 months
            if dow in (5, 6):  # weekend boost for retail
                base = rng.uniform(950, 1500)
            else:
                base = rng.uniform(550, 950)
            amount = _q(Decimal(str(base * growth)))
            merchant = rng.choice(MERCHANTS["Sales"])
            txs.append(
                self._tx(operating, amount, models.TransactionType.CREDIT, categories["Sales"],
                         merchant, d.replace(hour=rng.randint(9, 20)))
            )

            # -- Occasional second sale (wholesale orders ~ weekly)
            if rng.random() < 0.16:
                amount = _q(Decimal(str(rng.uniform(800, 2400) * growth)))
                txs.append(
                    self._tx(operating, amount, models.TransactionType.CREDIT, categories["Sales"],
                             "Wholesale Order", d.replace(hour=rng.randint(11, 17)))
                )

            # -- Inventory purchases ~2-3x per week
            if rng.random() < 0.38:
                amount = _q(Decimal(str(rng.uniform(350, 1100))))
                merchant = rng.choice(MERCHANTS["Inventory"])
                txs.append(
                    self._tx(operating, amount, models.TransactionType.DEBIT, categories["Inventory"],
                             merchant, d.replace(hour=rng.randint(8, 18)))
                )

            # -- Supplies ~ weekly
            if rng.random() < 0.12:
                amount = _q(Decimal(str(rng.uniform(80, 300))))
                txs.append(
                    self._tx(operating, amount, models.TransactionType.DEBIT, categories["Supplies"],
                             rng.choice(MERCHANTS["Supplies"]), d.replace(hour=rng.randint(9, 16)))
                )

            # -- Marketing spend ~ weekly, small
            if rng.random() < 0.10:
                amount = _q(Decimal(str(rng.uniform(60, 320))))
                txs.append(
                    self._tx(operating, amount, models.TransactionType.DEBIT, categories["Marketing"],
                             rng.choice(MERCHANTS["Marketing"]), d.replace(hour=rng.randint(9, 15)))
                )

            # -- Utilities monthly (1st of month)
            if d.day == 1:
                amount = _q(Decimal(str(rng.uniform(380, 620))))
                txs.append(
                    self._tx(operating, amount, models.TransactionType.DEBIT, categories["Utilities"],
                             MERCHANTS["Utilities"][0], d.replace(hour=10))
                )
                amount = _q(Decimal(str(rng.uniform(90, 180))))
                txs.append(
                    self._tx(operating, amount, models.TransactionType.DEBIT, categories["Utilities"],
                             MERCHANTS["Utilities"][1], d.replace(hour=10, minute=30))
                )

            # -- Rent monthly (3rd of month)
            if d.day == 3:
                txs.append(
                    self._tx(operating, Decimal("2400.00"), models.TransactionType.DEBIT,
                             categories["Rent"], MERCHANTS["Rent"][0], d.replace(hour=9))
                )

            # -- Payroll: 6th and 21st
            if d.day in (6, 21):
                txs.append(
                    self._tx(operating, Decimal("4200.00"), models.TransactionType.DEBIT,
                             categories["Payroll"], MERCHANTS["Payroll"][0], d.replace(hour=8))
                )

            # -- Insurance quarterly
            if d.day == 15 and d.month in (1, 4, 7, 10):
                txs.append(
                    self._tx(operating, Decimal("1850.00"), models.TransactionType.DEBIT,
                             categories["Insurance"], MERCHANTS["Insurance"][0], d.replace(hour=11))
                )

            # -- Monthly transfer to savings (28th)
            if d.day == 28:
                txs.append(
                    self._tx(operating, Decimal("2500.00"), models.TransactionType.DEBIT,
                             categories["Transfer"], MERCHANTS["Transfer"][0], d.replace(hour=15))
                )
                txs.append(
                    self._tx(savings, Decimal("2500.00"), models.TransactionType.CREDIT,
                             categories["Transfer"], MERCHANTS["Transfer"][0], d.replace(hour=15, minute=1))
                )

            d += timedelta(days=1)

        # -- Deliberate unusual events (for the anomaly detection workflow)
        # 1. A large one-off equipment purchase ~40 days ago
        spike_day = today - timedelta(days=40)
        txs.append(
            self._tx(operating, _q(Decimal("8950.00")), models.TransactionType.DEBIT,
                     categories["Other"], "VisionTech Equipment", spike_day.replace(hour=14))
        )
        # 2. A duplicate marketing charge 5 days ago
        dup_day = today - timedelta(days=5)
        for _ in range(2):
            txs.append(
                self._tx(operating, Decimal("748.50"), models.TransactionType.DEBIT,
                         categories["Marketing"], "Meta Ads", dup_day.replace(hour=16))
            )
        # 3. An unusually small late-night transfer 12 days ago
        odd_day = today - timedelta(days=12)
        txs.append(
            self._tx(operating, Decimal("4.20"), models.TransactionType.DEBIT,
                     categories["Other"], "Unknown Vendor", odd_day.replace(hour=2))
        )

        for t in txs:
            self.db.add(t)
        self.db.flush()
        return txs

    def _tx(
        self,
        account: models.Account,
        amount: Decimal,
        type_: str,
        category: models.Category,
        merchant: str,
        when: datetime,
    ) -> models.Transaction:
        return models.Transaction(
            account_id=account.id,
            amount=amount,
            description=merchant,
            merchant=merchant,
            timestamp=when,
            type=type_,
            category_id=category.id if category else None,
        )

    # ------------------------------------------------------------------
    # Ledger reconstruction (authoritative running balances)
    # ------------------------------------------------------------------
    def _record_ledger(
        self,
        operating: models.Account,
        savings: models.Account,
        txs: List[models.Transaction],
        opening_operating: Decimal,
        opening_savings: Decimal,
    ) -> None:
        balances: Dict[int, Decimal] = {
            operating.id: opening_operating,
            savings.id: opening_savings,
        }
        for t in sorted(txs, key=lambda x: x.timestamp or datetime.utcnow()):
            delta = t.amount if t.type == models.TransactionType.CREDIT else -t.amount
            new_balance = balances[t.account_id] + delta
            balances[t.account_id] = new_balance
            self.db.add(
                models.LedgerEntry(
                    transaction_id=t.id,
                    account_id=t.account_id,
                    amount=t.amount,
                    running_balance=_q(new_balance),
                    entry_date=t.timestamp,
                )
            )
        operating.balance = _q(balances[operating.id])
        savings.balance = _q(balances[savings.id])
