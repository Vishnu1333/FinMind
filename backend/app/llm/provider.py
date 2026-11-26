# backend/app/llm/provider.py
"""LLM provider abstraction.

Provides a protocol ``LLMProvider`` with a ``generate`` method and concrete
implementations for Gemini and Ollama. Re-exports and integrates with
``backend.app.ai`` for unified provider management.
"""

from __future__ import annotations

import os
from typing import Protocol, Optional

from backend.app.ai.schemas import RequestContext, ProviderResponse
from backend.app.ai.provider import (
    AIProvider,
    AIProviderError,
    AIConfigurationError,
    AITimeoutError,
    AIConnectionError,
    AIResponseFormatError,
)
from backend.app.ai.gemini_provider import GeminiProvider
from backend.app.ai.ollama_provider import OllamaProvider
from backend.app.ai.security import AISecurityPolicy, SecurityPolicyError
from backend.app.ai.provider_manager import AIProviderManager, get_ai_manager


# Backwards-compatible exception alias
class LLMProviderError(AIProviderError):
    """Raised when a provider cannot be instantiated or fails during generation."""
    pass


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given a system prompt and a user prompt."""
        ...


class ManagedLLMAdapter:
    """Adapts AIProviderManager to the 2-argument generate(system_prompt, user_prompt) protocol."""

    def __init__(self, privacy_sensitive: bool = False, offline_only: bool = False) -> None:
        self.context = RequestContext(
            privacy_sensitive=privacy_sensitive,
            offline_only=offline_only,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            manager = get_ai_manager()
            resp = manager.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                context=self.context,
            )
            return resp.text
        except (SecurityPolicyError, AIProviderError) as exc:
            raise LLMProviderError(str(exc)) from exc


def get_provider(
    privacy_sensitive: bool = False,
    offline_only: bool = False,
) -> LLMProvider:
    """Factory that returns an LLM provider respecting security policy, routing, and configuration."""
    # Check if AI_PROVIDER_MODE is valid
    mode = os.getenv("AI_PROVIDER_MODE", "auto").strip().lower()
    try:
        AISecurityPolicy.validate_mode(mode)
    except SecurityPolicyError as exc:
        raise LLMProviderError(str(exc)) from exc

    return ManagedLLMAdapter(
        privacy_sensitive=privacy_sensitive,
        offline_only=offline_only,
    )
