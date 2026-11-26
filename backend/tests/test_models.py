import pytest
from sqlalchemy.exc import IntegrityError
from backend.app.db import SessionLocal, engine, Base
from backend.app.models.business_models import Business, User, Account, Category, Transaction, LedgerEntry
from backend.app.models.document_models import Document, DocumentChunk
from datetime import datetime

# Ensure tables exist for ORM models
Base.metadata.create_all(engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a new database session for a test, roll back after test.
    Uses the real PostgreSQL test database as required.
    """
    connection = engine.connect()
    trans = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()

def test_business_crud(db_session):
    # Create
    b = Business(name="Acme Corp")
    db_session.add(b)
    db_session.flush()
    assert b.id is not None

    # Read
    fetched = db_session.query(Business).filter_by(name="Acme Corp").one()
    assert fetched.id == b.id

    # Delete
    db_session.delete(fetched)
    db_session.commit()
    remaining = db_session.query(Business).filter_by(name="Acme Corp").all()
    assert len(remaining) == 0

def test_business_unique_name(db_session):
    b1 = Business(name="UniqueBiz")
    db_session.add(b1)
    db_session.flush()
    b2 = Business(name="UniqueBiz")
    db_session.add(b2)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()

def test_user_business_relationship(db_session):
    biz = Business(name="BizCo")
    db_session.add(biz)
    db_session.flush()
    user = User(email="alice@example.com", hashed_password="hash", business_id=biz.id)
    db_session.add(user)
    db_session.commit()
    # Access via relationship
    assert user.business.id == biz.id
    assert biz.users[0].id == user.id

def test_account_relations(db_session):
    biz = Business(name="AccBiz")
    db_session.add(biz)
    db_session.flush()
    user = User(email="bob@example.com", hashed_password="hash", business_id=biz.id)
    db_session.add(user)
    db_session.flush()
    account = Account(user_id=user.id, business_id=biz.id, name="Cash", balance=1000)
    db_session.add(account)
    db_session.flush()
    assert account.user.id == user.id
    assert account.business.id == biz.id

def test_category_transaction_relationship(db_session):
    cat = Category(name="UtilitiesTest")
    db_session.add(cat)
    db_session.flush()
    biz = Business(name="TransBiz")
    db_session.add(biz)
    db_session.flush()
    user = User(email="carl@example.com", hashed_password="hash", business_id=biz.id)
    db_session.add(user)
    db_session.flush()
    account = Account(user_id=user.id, business_id=biz.id, name="Checking")
    db_session.add(account)
    db_session.flush()
    tx = Transaction(account_id=account.id, amount=50, description="Electric bill", type="DEBIT", category_id=cat.id)
    db_session.add(tx)
    db_session.flush()
    assert tx.account.id == account.id
    assert tx.category.id == cat.id
    # Ledger entry creation (simulate)
    le = LedgerEntry(transaction_id=tx.id, account_id=account.id, amount=tx.amount, running_balance=950)
    db_session.add(le)
    db_session.flush()
    assert le.transaction.id == tx.id
    assert le.account.id == account.id

def test_document_business_and_unique_hash(db_session):
    biz = Business(name="DocBiz")
    db_session.add(biz)
    db_session.flush()
    doc1 = Document(business_id=biz.id, filename="file1.pdf", document_type="PDF", content_hash="hash123", status="pending")
    db_session.add(doc1)
    db_session.flush()
    # Duplicate hash for same business should raise IntegrityError
    doc2 = Document(business_id=biz.id, filename="file2.pdf", document_type="PDF", content_hash="hash123", status="pending")
    db_session.add(doc2)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()
    # Different business can use same hash
    biz2 = Business(name="OtherBiz2")
    db_session.add(biz2)
    db_session.flush()
    doc3 = Document(business_id=biz2.id, filename="file3.pdf", document_type="PDF", content_hash="hash123", status="pending")
    db_session.add(doc3)
    db_session.flush()
    assert doc3.id is not None

def test_document_chunks_and_vector_column(db_session):
    biz = Business(name="ChunkBiz")
    db_session.add(biz)
    db_session.flush()
    doc = Document(business_id=biz.id, filename="notes.txt", document_type="TXT", content_hash="hash456", status="processed")
    db_session.add(doc)
    db_session.flush()
    chunk = DocumentChunk(document_id=doc.id, chunk_number=1, text="Hello world", embedding=None, metadata={"page": 1})
    db_session.add(chunk)
    db_session.flush()
    fetched = db_session.query(DocumentChunk).filter_by(document_id=doc.id).one()
    assert fetched.text == "Hello world"
    # Ensure relationship back to document works
    assert fetched.document.id == doc.id
    assert doc.document_chunks[0].id == fetched.id

def test_cascade_delete_business(db_session):
    biz = Business(name="CascadeBiz")
    db_session.add(biz)
    db_session.flush()
    user = User(email="del@example.com", hashed_password="hash", business_id=biz.id)
    db_session.add(user)
    db_session.flush()
    account = Account(user_id=user.id, business_id=biz.id, name="Savings")
    db_session.add(account)
    doc = Document(business_id=biz.id, filename="file.pdf", document_type="PDF", content_hash="hash789", status="pending")
    db_session.add(doc)
    db_session.flush()
    # Verify rows exist
    assert db_session.query(User).filter_by(business_id=biz.id).count() == 1
    assert db_session.query(Account).filter_by(business_id=biz.id).count() == 1
    assert db_session.query(Document).filter_by(business_id=biz.id).count() == 1
    # Delete business
    db_session.delete(biz)
    db_session.commit()
    # Cascades should have removed dependent rows
    assert db_session.query(User).filter_by(business_id=biz.id).count() == 0
    assert db_session.query(Account).filter_by(business_id=biz.id).count() == 0
    assert db_session.query(Document).filter_by(business_id=biz.id).count() == 0
