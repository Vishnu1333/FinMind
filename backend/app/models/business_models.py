from sqlalchemy import Column, Integer, String, DateTime, Numeric, ForeignKey, UniqueConstraint, Text, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.app.db import Base


class TransactionType:
    """Authoritative transaction type constants (stored as 6-char strings)."""
    CREDIT = "credit"  # money in
    DEBIT = "debit"    # money out


class Business(Base):
    __tablename__ = "businesses"
    __table_args__ = (UniqueConstraint('name', name='uq_business_name'),)
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    # Relationships
    users = relationship("User", back_populates="business", cascade="all, delete-orphan")
    accounts = relationship("Account", back_populates="business", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="business", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)

    business = relationship("Business", back_populates="users")
    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")

class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    balance = Column(Numeric(12, 2), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    user = relationship("User", back_populates="accounts")
    business = relationship("Business", back_populates="accounts")
    ledger_entries = relationship("LedgerEntry", back_populates="account", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")

class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)

    transactions = relationship("Transaction", back_populates="category")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    description = Column(String, nullable=True)
    merchant = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=True)
    type = Column(String(6), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id", ondelete="CASCADE"), nullable=True)

    account = relationship("Account", back_populates="transactions")
    category = relationship("Category", back_populates="transactions")
    ledger_entries = relationship("LedgerEntry", back_populates="transaction", cascade="all, delete-orphan")

class LedgerEntry(Base):
    __tablename__ = "ledger_entries"
    __table_args__ = (UniqueConstraint('transaction_id', 'account_id', name='uq_ledger_tx_account'),)
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    running_balance = Column(Numeric(12, 2), nullable=False)
    entry_date = Column(DateTime(timezone=True), nullable=True)

    transaction = relationship("Transaction", back_populates="ledger_entries")
    account = relationship("Account", back_populates="ledger_entries")

class Card(Base):
    __tablename__ = "cards"
    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    last_four = Column(String(4), nullable=False)
    status = Column(String, nullable=False, default="active") # active, blocked
    
    business = relationship("Business")

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    read = Column(Integer, default=0) # SQLite/Postgres compatible bool
    severity = Column(String, nullable=False, default="info")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    business = relationship("Business")

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="open")
    priority = Column(String, nullable=False, default="normal")
    category = Column(String, nullable=False, default="general")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    business = relationship("Business")

class UserSetting(Base):
    __tablename__ = "user_settings"
    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, unique=True)
    profile_data = Column(JSON, nullable=True)
    security_data = Column(JSON, nullable=True)
    notifications_data = Column(JSON, nullable=True)

    business = relationship("Business")


class AssistantConversation(Base):
    __tablename__ = "assistant_conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    messages = relationship(
        "AssistantMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AssistantMessage.id",
    )


class AssistantMessage(Base):
    __tablename__ = "assistant_messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, ForeignKey("assistant_conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role = Column(String(16), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    meta = Column(JSON, nullable=True)  # provider, refused, steps summary
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)

    conversation = relationship("AssistantConversation", back_populates="messages")
