# backend/app/db.py
"""SQLAlchemy database configuration and session handling.
Provides:
- ``engine`` – SQLAlchemy engine bound to ``DATABASE_URL``.
- ``Base`` – declarative base for ORM models.
- ``SessionLocal`` – session factory used throughout the app.
- ``get_db`` – FastAPI dependency (if needed) that yields a DB session.
"""
import os
import sys

# Ensure repository root is on sys.path regardless of execution CWD
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    _dotenv_path = os.path.join(_repo_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Determine database URL. The application runtime uses DATABASE_URL.
# The dedicated test database (TEST_DATABASE_URL) is used only when the
# pytest conftest installs it – it must never silently capture the runtime.
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL must be set in environment. FinMind is PostgreSQL-only."
    )

if DATABASE_URL.startswith("sqlite"):
    raise ValueError(
        "SQLite is not supported in FinMind. A PostgreSQL database URL is required."
    )

engine = create_engine(DATABASE_URL, echo=False, future=True)

# Base class for all ORM models
Base = declarative_base()

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

def get_db():
    """FastAPI dependency that yields a database session.
    Usage::
        from backend.app.db import get_db
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Import ORM models to ensure metadata is populated
from backend.app.models import business_models, document_models

# Database schema changes are managed by Alembic migrations. Do not run DDL
# at import time: this module is also imported by Alembic to load metadata.
