# backend/app/ai/security.py
"""Security routing and validation for AI requests.

Enforces data minimization, privacy boundaries, and fail-closed security rules
BEFORE any provider selection or network call takes place.
"""

from typing import Optional
from backend.app.ai.schemas import RequestContext


class SecurityPolicyError(RuntimeError):
    """Raised when an operation violates AI privacy or security policy."""
    pass


class AISecurityPolicy:
    """Enforces AI routing and privacy constraints."""

    VALID_MODES = {"auto", "gemini", "ollama"}

    @classmethod
    def validate_mode(cls, mode: str) -> str:
        """Validate AI_PROVIDER_MODE configuration."""
        normalized = (mode or "").strip().lower()
        if normalized not in cls.VALID_MODES:
            raise SecurityPolicyError(
                f"Invalid AI_PROVIDER_MODE '{mode}'. Allowed modes are: {sorted(cls.VALID_MODES)}"
            )
        return normalized

    @classmethod
    def evaluate_request(
        cls,
        context: RequestContext,
        mode: str = "auto",
    ) -> str:
        """Determine mandatory routing constraints based on security context.

        Rules:
        1. If privacy_sensitive is True:
           - Provider MUST be Ollama.
           - Cloud providers (Gemini) MUST NEVER be called.
           - If mode is explicitly 'gemini', fail-closed immediately.
        2. If offline_only is True:
           - Provider MUST be Ollama.
           - If mode is explicitly 'gemini', fail-closed immediately.
        3. If cloud is allowed:
           - mode == 'ollama' -> Ollama
           - mode == 'gemini' -> Gemini
           - mode == 'auto' -> Gemini first with Ollama fallback

        Returns:
            'ollama' | 'gemini' | 'auto'
        """
        validated_mode = cls.validate_mode(mode)

        if context.privacy_sensitive:
            if validated_mode == "gemini":
                raise SecurityPolicyError(
                    "Security policy violation: privacy_sensitive request cannot be routed to Gemini "
                    "when AI_PROVIDER_MODE is explicitly configured to 'gemini'."
                )
            return "ollama"

        if context.offline_only:
            if validated_mode == "gemini":
                raise SecurityPolicyError(
                    "Security policy violation: offline_only request cannot be routed to Gemini "
                    "when AI_PROVIDER_MODE is explicitly configured to 'gemini'."
                )
            return "ollama"

        return validated_mode
