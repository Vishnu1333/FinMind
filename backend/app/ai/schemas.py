# backend/app/ai/schemas.py
"""Data structures and schemas for AI provider orchestration."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class RequestContext:
    """Explicit security and operational metadata for an AI request."""
    privacy_sensitive: bool = False
    offline_only: bool = False


@dataclass
class ProviderResponse:
    """Structured response returned by an AI provider."""
    text: str
    model: str
    provider: str
    raw_response: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.text
