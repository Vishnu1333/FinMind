# backend/app/ai/__init__.py
"""AI Providers & Orchestration module for FinMind.

Provides unified provider contracts, Gemini and Ollama providers,
security routing policy, and provider management.
"""

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

__all__ = [
    "RequestContext",
    "ProviderResponse",
    "AIProvider",
    "AIProviderError",
    "AIConfigurationError",
    "AITimeoutError",
    "AIConnectionError",
    "AIResponseFormatError",
    "GeminiProvider",
    "OllamaProvider",
    "AISecurityPolicy",
    "SecurityPolicyError",
    "AIProviderManager",
    "get_ai_manager",
]
