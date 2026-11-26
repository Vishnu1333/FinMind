import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# FinMind is PostgreSQL-only.
# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

# In pytest, TEST_DATABASE_URL is strictly required to prevent accidental writes to development DB.
# The runtime (backend/app/db.py) reads DATABASE_URL only, so the test conftest
# must explicitly point the runtime at the isolated test database.
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
if not TEST_DATABASE_URL:
    raise RuntimeError(
        "TEST_DATABASE_URL must be configured for tests. "
        "Tests must run on the dedicated isolated test database (e.g. finmind_test). "
        "Falling back to DATABASE_URL or SQLite is strictly prohibited."
    )

# Force the runtime DB configuration onto the isolated test database.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
DATABASE_URL = TEST_DATABASE_URL

if DATABASE_URL.startswith("sqlite"):
    raise ValueError(
        "SQLite is not supported in FinMind. A dedicated PostgreSQL test database URL is required."
    )

from backend.app.db import engine, Base, SessionLocal
from backend.app.models import business_models, document_models  # noqa: F401

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure tables exist on the PostgreSQL test database for the test session."""
    Base.metadata.create_all(bind=engine)
    yield
    # Tables are kept or dropped as required; avoid dropping schema in active DB


@pytest.fixture(autouse=True)
def clean_database():
    """Clean all tables between tests to ensure test isolation."""
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
