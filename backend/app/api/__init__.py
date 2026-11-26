from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from backend.app.db import engine

router = APIRouter()


@router.get("/health")
def health_check():
    """Readiness probe that verifies the API can reach its PostgreSQL database."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="FinMind database connection is unavailable.",
        ) from exc

    return {"status": "ok", "database": "connected"}
