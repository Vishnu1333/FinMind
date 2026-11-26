# backend/app/ai/ollama_provider.py
"""Local Ollama AI provider implementation.

Communicates with a local Ollama server via HTTP REST.
Configuration defaults:
- OLLAMA_BASE_URL: http://localhost:11434
- OLLAMA_MODEL: llama3 (or configured model)
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Optional, Any, Dict

from backend.app.ai.provider import (
    AIProvider,
    AIProviderError,
    AIConfigurationError,
    AITimeoutError,
    AIConnectionError,
    AIResponseFormatError,
)
from backend.app.ai.schemas import ProviderResponse


class OllamaProvider(AIProvider):
    """Production provider for local Ollama server."""

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"
    DEFAULT_TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if base_url is not None and not base_url.strip():
            raise AIConfigurationError("OLLAMA_BASE_URL must be specified.")
        if model is not None and not model.strip():
            raise AIConfigurationError("OLLAMA_MODEL must be specified.")

        resolved_url = base_url if base_url is not None else os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        resolved_model = model if model is not None else os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)

        if not resolved_url or not resolved_url.strip():
            raise AIConfigurationError("OLLAMA_BASE_URL must be specified.")
        if not resolved_model or not resolved_model.strip():
            raise AIConfigurationError("OLLAMA_MODEL must be specified.")

        self.base_url = resolved_url.rstrip("/")
        self.model = resolved_model
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT", str(self.DEFAULT_TIMEOUT_SECONDS)))

    @property
    def name(self) -> str:
        return "ollama"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        """Generate text from Ollama via POST /api/generate."""
        if not prompt:
            raise AIProviderError("Prompt cannot be empty.")

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        endpoint = f"{self.base_url}/api/generate"
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                if "error" in data:
                    raise AIProviderError(f"Ollama server returned error: {data['error']}")
                if "response" not in data:
                    raise AIResponseFormatError("Unexpected Ollama response: missing 'response' field.")
                return ProviderResponse(
                    text=data["response"],
                    model=self.model,
                    provider=self.name,
                    raw_response=data,
                )
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise AIProviderError(f"Ollama HTTP {exc.code} error: {err_body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            msg = str(exc)
            if "timed out" in msg.lower():
                raise AITimeoutError(f"Ollama generation timed out: {msg}") from exc
            raise AIConnectionError(f"Ollama server connection failed at {self.base_url}: {msg}") from exc
        except json.JSONDecodeError as exc:
            raise AIResponseFormatError(f"Malformed JSON from Ollama server: {exc}") from exc
        except AIProviderError:
            raise
        except Exception as exc:
            raise AIProviderError(f"Ollama generation failed: {exc}") from exc
