# This module now re-exports the main DB configuration from backend.app.db
# All database interactions should use the engine/session defined there.
# The DATABASE_URL must be set to a PostgreSQL URL; no SQLite fallback is provided.

from backend.app.db import engine, SessionLocal, Base, get_db  # noqa: F401
