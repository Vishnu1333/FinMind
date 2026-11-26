# backend/app/ai/provider.py
"""Unified provider interface and base exceptions for AI providers.

Provides the ``AIProvider`` protocol and controlled exceptions.
Providers do NOT have direct access to database, SQL, filesystem, shell,
or authorization logic.
"""

from __future__ import annotations

from typing import Protocol, Optional, runtime_checkable
from backend.app.ai.schemas import ProviderResponse


class AIProviderError(RuntimeError):
    """Base exception for all controlled AI provider failures."""
    pass


class AIConfigurationError(AIProviderError):
    """Raised when provider configuration or credentials are missing or invalid."""
    pass


class AITimeoutError(AIProviderError):
    """Raised when a provider request times out."""
    pass


class AIConnectionError(AIProviderError):
    """Raised when connection to provider service fails."""
    pass


class AIResponseFormatError(AIProviderError):
    """Raised when provider returns an unparseable or malformed response."""
    pass


@runtime_checkable
class AIProvider(Protocol):
    """Provider interface contract.

    Neutral interface supporting prompt, optional system instruction,
    temperature, and max tokens.
    """

    @property
    def name(self) -> str:
        """Name identifier of the provider (e.g. 'gemini', 'ollama')."""
        ...

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        """Generate a response given prompt and optional parameters.

        Args:
            prompt: Main prompt / user query.
            system_prompt: Optional system instruction.
            temperature: Optional sampling temperature.
            max_tokens: Optional maximum tokens to generate.

        Returns:
            ProviderResponse containing generated text and metadata.

        Raises:
            AIProviderError: On any failure.
        """
        ...
