# backend/app/ai/provider_manager.py
"""AI Provider Manager and Router.

Orchestrates AI provider selection, routing, privacy enforcement, and fallback.
Key rules:
- Privacy-sensitive requests -> Ollama only. Fail closed if Ollama unavailable.
- Offline-only requests -> Ollama only.
- Cloud allowed (AUTO) -> Gemini first, Ollama fallback if Gemini fails.
- Direct modes ('gemini', 'ollama') -> use specified provider.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from backend.app.ai.provider import AIProvider, AIProviderError, AIConfigurationError
from backend.app.ai.schemas import RequestContext, ProviderResponse
from backend.app.ai.security import AISecurityPolicy, SecurityPolicyError
from backend.app.ai.gemini_provider import GeminiProvider
from backend.app.ai.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class AIProviderManager:
    """Manages AI provider routing and execution lifecycle."""

    def __init__(
        self,
        gemini_provider: Optional[AIProvider] = None,
        ollama_provider: Optional[AIProvider] = None,
    ) -> None:
        self._gemini = gemini_provider
        self._ollama = ollama_provider

    def get_gemini(self) -> AIProvider:
        if self._gemini is None:
            self._gemini = GeminiProvider()
        return self._gemini

    def get_ollama(self) -> AIProvider:
        if self._ollama is None:
            self._ollama = OllamaProvider()
        return self._ollama

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[RequestContext] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> ProviderResponse:
        """Route and execute generation according to security policy and operational mode.

        Args:
            prompt: User query / main prompt.
            system_prompt: Optional system prompt.
            context: Security context (privacy_sensitive, offline_only).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            mode: Override for AI_PROVIDER_MODE ('auto', 'gemini', 'ollama').
        """
        req_context = context or RequestContext()
        configured_mode = mode or os.getenv("AI_PROVIDER_MODE", "auto")

        # 1. Security Decision MUST happen BEFORE any provider selection or call
        routing_decision = AISecurityPolicy.evaluate_request(req_context, mode=configured_mode)

        # 2. Execute according to security decision
        if routing_decision == "ollama":
            # Mandatory local execution (privacy-sensitive or offline-only)
            try:
                ollama = self.get_ollama()
                return ollama.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                # Fail-closed behavior: NEVER fall back to Gemini for privacy-sensitive or offline requests
                if req_context.privacy_sensitive:
                    raise SecurityPolicyError(
                        f"Fail-closed: Privacy-sensitive request failed because local provider (Ollama) is unavailable: {exc}"
                    ) from exc
                raise AIProviderError(f"Local provider (Ollama) failed: {exc}") from exc

        if routing_decision == "gemini":
            gemini = self.get_gemini()
            return gemini.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if routing_decision == "auto":
            # Cloud is allowed. Try Gemini first; fallback to Ollama if Gemini fails.
            try:
                gemini = self.get_gemini()
                return gemini.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as gemini_err:
                logger.warning(
                    "Cloud provider (Gemini) failed on non-sensitive request. Falling back to Ollama. Error: %s",
                    str(gemini_err),
                )
                try:
                    ollama = self.get_ollama()
                    return ollama.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as ollama_err:
                    raise AIProviderError(
                        f"All providers failed in AUTO mode. Gemini error: {gemini_err}; Ollama fallback error: {ollama_err}"
                    ) from ollama_err

        raise SecurityPolicyError(f"Unhandled routing decision: {routing_decision}")


# Default global instance
_default_manager = AIProviderManager()


def get_ai_manager() -> AIProviderManager:
    """Return the application's AIProviderManager instance."""
    return _default_manager
