# backend/app/ai/gemini_provider.py
"""Google Gemini AI provider implementation.

Communicates with the Google Gemini API using configured credentials.
Credentials are read ONLY from the GEMINI_API_KEY environment variable.
Never logs API keys, passwords, or full prompt contents by default.
"""

from __future__ import annotations

import logging
import os
import sys
import json
import time
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

logger = logging.getLogger(__name__)


class _GeminiHTTPError(RuntimeError):
    """HTTP error from the Gemini API, carrying status code and body."""

    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}: {body}")
        self.code = code
        self.body = body


class GeminiProvider(AIProvider):
    """Production provider for Google Gemini API."""

    DEFAULT_MODEL = "gemini-1.5-flash"
    DEFAULT_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise AIConfigurationError("GEMINI_API_KEY is not configured in environment.")

        self.model = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout or float(os.getenv("GEMINI_TIMEOUT", str(self.DEFAULT_MODEL and self.DEFAULT_TIMEOUT_SECONDS)))

    @property
    def name(self) -> str:
        return "gemini"

    def _call_via_sdk(
        self,
        genai: Any,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        """Execute request using the google-generativeai SDK."""
        try:
            if hasattr(genai, "configure"):
                genai.configure(api_key=self.api_key)

            gen_config = {}
            if temperature is not None:
                gen_config["temperature"] = temperature
            if max_tokens is not None:
                gen_config["max_output_tokens"] = max_tokens

            kwargs: Dict[str, Any] = {"model_name": self.model}
            if system_prompt:
                kwargs["system_instruction"] = system_prompt
            if gen_config:
                kwargs["generation_config"] = gen_config

            model = genai.GenerativeModel(**kwargs)
            resp = model.generate_content(prompt)

            text_content = ""
            if hasattr(resp, "text") and resp.text:
                text_content = resp.text
            elif hasattr(resp, "candidates") and resp.candidates:
                candidate = resp.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    text_content = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))

            if not text_content:
                raise AIResponseFormatError("Empty or malformed text received from Gemini SDK response.")

            return ProviderResponse(
                text=text_content,
                model=self.model,
                provider=self.name,
                raw_response=getattr(resp, "to_dict", lambda: None)(),
            )
        except AIProviderError:
            raise
        except TimeoutError as exc:
            raise AITimeoutError(f"Gemini SDK generation timed out: {exc}") from exc
        except Exception as exc:
            msg = str(exc)
            if "timed out" in msg.lower() or "timeout" in msg.lower():
                raise AITimeoutError(f"Gemini generation timed out: {msg}") from exc
            if "connection" in msg.lower() or "failed to connect" in msg.lower():
                raise AIConnectionError(f"Gemini connection failed: {msg}") from exc
            raise AIProviderError(f"Gemini SDK generation failed: {msg}") from exc

    def _call_via_rest(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        """Execute request using direct REST API (compatible when google-generativeai is absent)."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        contents = [{"parts": [{"text": prompt}]}]
        body: Dict[str, Any] = {"contents": contents}

        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        gen_config: Dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if gen_config:
            body["generationConfig"] = gen_config

        req_data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                candidates = payload.get("candidates", [])
                if not candidates:
                    raise AIResponseFormatError("Gemini REST response contains no candidates.")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)
                if not text:
                    raise AIResponseFormatError("Gemini REST response returned empty text part.")
                return ProviderResponse(
                    text=text,
                    model=self.model,
                    provider=self.name,
                    raw_response=payload,
                )
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            # Do NOT print the api_key or full raw URL with key
            raise AIProviderError(f"Gemini API returned HTTP {exc.code}: {err_body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            msg = str(exc)
            if "timed out" in msg.lower():
                raise AITimeoutError(f"Gemini request timed out: {msg}") from exc
            raise AIConnectionError(f"Gemini connection failed: {msg}") from exc
        except json.JSONDecodeError as exc:
            raise AIResponseFormatError(f"Malformed JSON response from Gemini: {exc}") from exc
        except AIProviderError:
            raise
        except Exception as exc:
            raise AIProviderError(f"Gemini generation failed: {exc}") from exc

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ProviderResponse:
        """Generate response via Gemini SDK if installed, or fallback to standard REST API."""
        if not prompt:
            raise AIProviderError("Prompt cannot be empty.")

        # Check if google.generativeai module or mock is available
        genai = None
        if "google.generativeai" in sys.modules:
            genai = sys.modules["google.generativeai"]
        else:
            try:
                import importlib
                genai = importlib.import_module("google.generativeai")
            except ImportError:
                genai = None

        if genai is not None:
            return self._call_via_sdk(
                genai=genai,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._call_via_rest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Agentic function calling (REST) – used by the FinMind assistant loop.
    # ------------------------------------------------------------------
    def _chat_turn_once(
        self,
        model: str,
        contents: list,
        system_instruction: str,
        tools: Optional[list] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Execute one model turn with optional tool (function) declarations.

        ``contents`` is the Gemini REST conversation format: a list of
        ``{"role": "user"|"model", "parts": [...]}`` entries where parts may be
        ``{"text": ...}``, ``{"functionCall": {...}}`` or
        ``{"functionResponse": {...}}``.

        Returns ``{"text": str, "function_calls": [...], "parts": [...],
        "finish_reason": str}``. The raw ``parts`` (including any
        ``thoughtSignature`` required by newer models) are returned so the
        agent loop can echo the model turn back verbatim.
        """
        if not system_instruction:
            raise AIProviderError("A system instruction is required for the agent loop.")

        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
        }
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if model.startswith("gemini-2.5"):
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        body["generationConfig"] = generation_config

        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise _GeminiHTTPError(exc.code, err_body[:400]) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            msg = str(exc)
            if "timed out" in msg.lower():
                raise AITimeoutError(f"Gemini request timed out: {msg}") from exc
            raise AIConnectionError(f"Gemini connection failed: {msg}") from exc
        except json.JSONDecodeError as exc:
            raise AIResponseFormatError(f"Malformed JSON response from Gemini: {exc}") from exc

        candidates = payload.get("candidates") or []
        if not candidates:
            raise AIResponseFormatError(
                f"Gemini returned no candidates ({payload.get('promptFeedback', {}).get('blockReason', 'unknown reason')})."
            )
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        function_calls = [
            {
                "name": p["functionCall"]["name"],
                "args": p["functionCall"].get("args") or {},
                "id": p["functionCall"].get("id"),
            }
            for p in parts
            if "functionCall" in p
        ]
        return {
            "text": "".join(text_parts).strip(),
            "function_calls": function_calls,
            "parts": parts,
            "finish_reason": candidate.get("finishReason", ""),
        }

    def chat_turn(
        self,
        contents: list,
        system_instruction: str,
        tools: Optional[list] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        max_attempts_per_model: int = 2,
    ) -> Dict[str, Any]:
        """Execute one agent turn against the configured model, transparently
        falling back to alternative models on deprecation (404) or capacity
        (503/429) errors. The first model that answers wins; its model name is
        reported in the response so callers can observe which model served the
        request. Raises AIProviderError only when every model fails."""
        from ..config import GEMINI_FALLBACK_MODELS

        chain: list = [self.model]
        chain += [m for m in GEMINI_FALLBACK_MODELS if m != self.model]

        last_error: Optional[Exception] = None
        for model in chain:
            for attempt in range(max(1, max_attempts_per_model)):
                try:
                    response = self._chat_turn_once(
                        model=model,
                        contents=contents,
                        system_instruction=system_instruction,
                        tools=tools,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    response["model"] = model
                    if model != self.model:
                        logger.info("Gemini fallback model used: %s", model)
                    return response
                except _GeminiHTTPError as exc:
                    last_error = exc
                    # 404 = model unavailable/deprecated, 503/429 = capacity:
                    # both are recoverable by switching model / retrying.
                    if exc.code in (404, 503, 429):
                        time.sleep(0.8 * (attempt + 1))
                        continue
                    raise AIProviderError(
                        f"Gemini API returned HTTP {exc.code}: {exc.body}"
                    ) from exc
                except (AITimeoutError, AIConnectionError) as exc:
                    last_error = exc
                    break  # try the next model in the chain
        raise AIProviderError(
            f"Gemini request failed on all models: {last_error}"
        )
