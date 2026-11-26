import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.orm import Session

# Import DB utilities and models
from backend.database.database import SessionLocal, engine, Base
from backend.database.models import (
    Business,
    User,
    Account,
    Category,
    Transaction,
    LedgerEntry,
    TransactionType,
)

# Ensure tables exist (should already be migrated)
Base.metadata.create_all(bind=engine)

def get_or_create(session: Session, model, defaults=None, **kwargs):
    """Get an instance or create it with defaults if not exists.
    Returns (instance, created: bool)."""
    instance = session.execute(select(model).filter_by(**kwargs)).scalar_one_or_none()
    if instance:
        return instance, False
    params = dict(kwargs)
    if defaults:
        params.update(defaults)
    instance = model(**params)
    session.add(instance)
    session.flush()  # assign PK
    return instance, True

def seed():
    session = SessionLocal()
    try:
        # 1. Business
        business, _ = get_or_create(session, Business, name="FinMind SME Demo Business")

        # 2. User (demo user)
        user_email = "demo_user@finmind.com"
        user, _ = get_or_create(
            session,
            User,
            email=user_email,
            hashed_password="not_used",  # placeholder; authentication not needed
            business_id=business.id,
        )

        # 3. Categories (deterministic list)
        category_names = [
            "Sales",
            "Payroll",
            "Rent",
            "Utilities",
            "Supplies",
            "Marketing",
            "Transfer",
            "Other",
        ]
        categories = {}
        for name in category_names:
            cat, _ = get_or_create(session, Category, name=name)
            categories[name] = cat

        # 4. Accounts
        account_defs = [
            {"name": "Operating Account"},
            {"name": "Business Savings Account"},
        ]
        accounts = {}
        for acc_def in account_defs:
            acc, _ = get_or_create(
                session,
                Account,
                name=acc_def["name"],
                user_id=user.id,
                business_id=business.id,
                balance=0,
            )
            accounts[acc_def["name"]] = acc

        # 5. Transactions + Ledger entries
        start_date = datetime.utcnow() - timedelta(days=30)
        transaction_data = [
            ("Operating Account", 5000.00, TransactionType.CREDIT, "Sales", "Initial sales revenue"),
            ("Operating Account", 2000.00, TransactionType.DEBIT, "Payroll", "January payroll"),
            ("Operating Account", 1500.00, TransactionType.DEBIT, "Rent", "Office rent"),
            ("Operating Account", 300.00, TransactionType.DEBIT, "Utilities", "Electricity bill"),
            ("Operating Account", 400.00, TransactionType.DEBIT, "Supplies", "Office supplies"),
            ("Operating Account", 250.00, TransactionType.DEBIT, "Marketing", "Online ads"),
            ("Operating Account", 1000.00, TransactionType.CREDIT, "Sales", "Additional sales"),
            ("Operating Account", 500.00, TransactionType.DEBIT, "Other", "Misc expense"),
            ("Operating Account", 2000.00, TransactionType.CREDIT, "Transfer", "Transfer from savings"),
            ("Business Savings Account", 2000.00, TransactionType.DEBIT, "Transfer", "Transfer to operating"),
        ]

        # Track running balances per account
        running_balances = {name: 0 for name in accounts}

        for idx, (acct_name, amount, tx_type, cat_name, desc) in enumerate(transaction_data, start=1):
            acct = accounts[acct_name]
            cat = categories[cat_name]
            ts = start_date + timedelta(days=idx)
            # Idempotency: ensure we don't duplicate
            existing_tx = session.execute(
                select(Transaction).where(
                    Transaction.account_id == acct.id,
                    Transaction.amount == amount,
                    Transaction.type == tx_type,
                    Transaction.description == desc,
                    func.date_trunc('second', Transaction.timestamp) == func.date_trunc('second', ts),
                )
            ).scalar_one_or_none()
            if existing_tx:
                tx = existing_tx
            else:
                tx = Transaction(
                    account_id=acct.id,
                    amount=amount,
                    type=tx_type,
                    description=desc,
                    timestamp=ts,
                    category_id=cat.id,
                )
                session.add(tx)
                session.flush()

                signed_amount = amount if tx_type == TransactionType.CREDIT else -amount
                new_balance = running_balances[acct_name] + signed_amount
                le = LedgerEntry(
                    transaction_id=tx.id,
                    account_id=acct.id,
                    amount=signed_amount,
                    running_balance=new_balance,
                    entry_date=ts,
                )
                session.add(le)
                running_balances[acct_name] = new_balance

        # Update account balances
        for acct_name, acct in accounts.items():
            final_balance = running_balances[acct_name]
            if acct.balance != final_balance:
                acct.balance = final_balance
                session.add(acct)

        session.commit()

        # Validation & reporting
        biz_count = session.execute(select(func.count()).select_from(Business)).scalar_one()
        user_count = session.execute(select(func.count()).select_from(User)).scalar_one()
        account_count = session.execute(select(func.count()).select_from(Account)).scalar_one()
        category_count = session.execute(select(func.count()).select_from(Category)).scalar_one()
        transaction_count = session.execute(select(func.count()).select_from(Transaction)).scalar_one()
        ledger_count = session.execute(select(func.count()).select_from(LedgerEntry)).scalar_one()

        # Balance validation
        balance_mismatches = []
        for acct in session.execute(select(Account)).scalars():
            sum_amount = session.execute(
                select(func.coalesce(func.sum(LedgerEntry.amount), 0)).where(LedgerEntry.account_id == acct.id)
            ).scalar_one()
            if sum_amount != acct.balance:
                balance_mismatches.append((acct.id, acct.balance, sum_amount))

        # Orphan checks
        orphan_transactions = session.execute(select(Transaction).where(~Transaction.account.has())).scalars().all()
        orphan_categories = session.execute(select(Category).where(~Category.transactions.any())).scalars().all()
        orphan_accounts = session.execute(select(Account).where(~Account.business.has())).scalars().all()
        orphan_ledger = session.execute(select(LedgerEntry).where(~LedgerEntry.account.has())).scalars().all()
        duplicate_ledger = session.execute(
            select(LedgerEntry.transaction_id, LedgerEntry.account_id, func.count())
            .group_by(LedgerEntry.transaction_id, LedgerEntry.account_id)
            .having(func.count() > 1)
        ).all()

        print("PHASE 2 SEED STATUS: PASS")
        print(f"Business count: {biz_count}")
        print(f"User count: {user_count}")
        print(f"Account count: {account_count}")
        print(f"Category count: {category_count}")
        print(f"Transaction count: {transaction_count}")
        print(f"Ledger entry count: {ledger_count}")
        print(f"Balance mismatches: {len(balance_mismatches)}")
        print(f"Orphan transactions: {len(orphan_transactions)}")
        print(f"Orphan categories: {len(orphan_categories)}")
        print(f"Orphan accounts: {len(orphan_accounts)}")
        print(f"Orphan ledger entries: {len(orphan_ledger)}")
        print(f"Duplicate ledger entries (by tx/account): {len(duplicate_ledger)}")
    except Exception as e:
        session.rollback()
        print("PHASE 2 SEED STATUS: FAIL")
        print("Error:", e)
        sys.exit(1)
    finally:
        session.close()

if __name__ == "__main__":
    seed()
