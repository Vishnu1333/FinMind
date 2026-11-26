"""Tests for business provisioning (the simulated banking environment)."""

import os
import sys

import pytest
from decimal import Decimal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import models  # noqa: E402
from backend.app.db import SessionLocal  # noqa: E402
from backend.app.services.provisioning_service import ProvisioningService  # noqa: E402


@pytest.fixture()
def session():
    db = SessionLocal()
    yield db
    db.close()


def _make_business(session, name, email):
    business = models.Business(name=name)
    session.add(business)
    session.flush()
    user = models.User(email=email, hashed_password="x", business_id=business.id)
    session.add(user)
    session.commit()
    session.refresh(business)
    session.refresh(user)
    return business, user


class TestProvisioning:
    def test_provision_creates_full_environment(self, session):
        business, user = _make_business(session, "Provision Test Sdn Bhd", "prov_test@finmind.com")
        ProvisioningService(session).provision(business, user)

        accounts = session.query(models.Account).filter_by(business_id=business.id).all()
        cards = session.query(models.Card).filter_by(business_id=business.id).all()
        txs = (
            session.query(models.Transaction)
            .join(models.Account)
            .filter(models.Account.business_id == business.id)
            .count()
        )
        ledger = (
            session.query(models.LedgerEntry)
            .join(models.Transaction)
            .join(models.Account)
            .filter(models.Account.business_id == business.id)
            .count()
        )
        notifications = session.query(models.Notification).filter_by(business_id=business.id).count()

        assert len(accounts) == 2
        assert {c.last_four for c in cards} == {"4821", "9932"}
        assert txs > 150
        assert ledger == txs
        assert notifications == 1
        # Both income and expenses exist
        credits = (
            session.query(models.Transaction)
            .join(models.Account)
            .filter(models.Account.business_id == business.id,
                    models.Transaction.type == models.TransactionType.CREDIT)
            .count()
        )
        debits = (
            session.query(models.Transaction)
            .join(models.Account)
            .filter(models.Account.business_id == business.id,
                    models.Transaction.type == models.TransactionType.DEBIT)
            .count()
        )
        assert credits > 20
        assert debits > 20

    def test_ledger_matches_account_balance(self, session):
        business, user = _make_business(session, "Ledger Test Sdn Bhd", "ledger_test@finmind.com")
        ProvisioningService(session).provision(business, user)
        for account in session.query(models.Account).filter_by(business_id=business.id).all():
            latest = (
                session.query(models.LedgerEntry)
                .filter_by(account_id=account.id)
                .order_by(models.LedgerEntry.entry_date.desc(), models.LedgerEntry.id.desc())
                .first()
            )
            assert latest is not None
            assert Decimal(str(account.balance)) == Decimal(str(latest.running_balance))

    def test_provisioning_is_deterministic(self, session):
        b1, u1 = _make_business(session, "Determinism A Sdn Bhd", "det_a@finmind.com")
        b2, u2 = _make_business(session, "Determinism B Sdn Bhd", "det_b@finmind.com")
        ProvisioningService(session).provision(b1, u1)
        ProvisioningService(session).provision(b2, u2)

        sums = []
        for b in (b1, b2):
            total = sum(
                t.amount for t in session.query(models.Transaction).join(models.Account)
                .filter(models.Account.business_id == b.id).all()
            )
            sums.append(Decimal(str(total)))
        assert len(sums) == 2  # both provisioned

    def test_categories_are_shared_and_unique(self, session):
        business, user = _make_business(session, "Category Test Sdn Bhd", "cat_test@finmind.com")
        svc = ProvisioningService(session)
        svc.provision(business, user)
        # Idempotent: provisioning a second business reuses global categories
        names = [c.name for c in session.query(models.Category).all()]
        assert len(names) == len(set(names))
        assert "Inventory" in names
