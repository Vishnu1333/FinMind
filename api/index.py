"""Vercel entrypoint for the FinMind FastAPI application."""

from backend.app.main import app

__all__ = ["app"]
