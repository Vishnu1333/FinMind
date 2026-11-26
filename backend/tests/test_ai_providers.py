# backend/tests/test_ai_providers.py
"""Comprehensive test suite for AI providers, security routing, and error handling.

Tests all required Block 5 specifications:
A. Gemini selected for normal cloud-allowed AUTO request.
B. Ollama selected for offline request.
C. Ollama selected for privacy-sensitive request.
D. Privacy-sensitive request NEVER calls Gemini.
E. Privacy-sensitive + Ollama unavailable -> FAIL CLOSED.
F. Gemini unavailable + normal cloud-allowed request -> Ollama fallback.
G. Gemini unavailable + privacy-sensitive request -> MUST NOT fallback to Gemini.
H. Invalid AI_PROVIDER_MODE rejected.
I. Missing Gemini key produces controlled error.
J. Missing Ollama configuration produces controlled error where required.
K. Provider response parsing works.
L. Provider timeout/error is converted into controlled provider failure.
M. No secrets appear in logs/errors.
N. Existing RAG tests continue to pass.
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

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
from backend.app.llm.provider import get_provider, LLMProviderError


class env_context:
    """Helper context manager to temporarily set environment variables."""
    def __init__(self, **kwargs):
        self.new = kwargs
        self.old = {}

    def __enter__(self):
        for k, v in self.new.items():
            self.old[k] = os.getenv(k)
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v

    def __exit__(self, exc_type, exc, tb):
        for k, v in self.old.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Test Helpers / Mocks
# ---------------------------------------------------------------------------

class MockAIProvider(AIProvider):
    def __init__(self, name: str, return_text: str = "mocked response", should_fail: bool = False, fail_exc: Exception = None):
        self._name = name
        self.return_text = return_text
        self.should_fail = should_fail
        self.fail_exc = fail_exc or AIProviderError(f"{name} provider failed")
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str, system_prompt=None, temperature=None, max_tokens=None) -> ProviderResponse:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        if self.should_fail:
            raise self.fail_exc
        return ProviderResponse(text=self.return_text, model="test-model", provider=self.name)


# ---------------------------------------------------------------------------
# Requirement A: Gemini selected for normal cloud-allowed AUTO request
# ---------------------------------------------------------------------------

def test_a_gemini_selected_for_normal_cloud_auto():
    gemini_mock = MockAIProvider("gemini", return_text="Gemini normal answer")
    ollama_mock = MockAIProvider("ollama", return_text="Ollama answer")
    manager = AIProviderManager(gemini_provider=gemini_mock, ollama_provider=ollama_mock)

    context = RequestContext(privacy_sensitive=False, offline_only=False)
    resp = manager.generate("What is SME banking?", context=context, mode="auto")

    assert resp.text == "Gemini normal answer"
    assert resp.provider == "gemini"
    assert gemini_mock.call_count == 1
    assert ollama_mock.call_count == 0


# ---------------------------------------------------------------------------
# Requirement B: Ollama selected for offline request
# ---------------------------------------------------------------------------

def test_b_ollama_selected_for_offline_request():
    gemini_mock = MockAIProvider("gemini", return_text="Gemini answer")
    ollama_mock = MockAIProvider("ollama", return_text="Ollama offline answer")
    manager = AIProviderManager(gemini_provider=gemini_mock, ollama_provider=ollama_mock)

    context = RequestContext(privacy_sensitive=False, offline_only=True)
    resp = manager.generate("Compute ratio offline", context=context, mode="auto")

    assert resp.text == "Ollama offline answer"
    assert resp.provider == "ollama"
    assert gemini_mock.call_count == 0
    assert ollama_mock.call_count == 1


# ---------------------------------------------------------------------------
# Requirement C: Ollama selected for privacy-sensitive request
# ---------------------------------------------------------------------------

def test_c_ollama_selected_for_privacy_sensitive_request():
    gemini_mock = MockAIProvider("gemini", return_text="Gemini answer")
    ollama_mock = MockAIProvider("ollama", return_text="Ollama sensitive answer")
    manager = AIProviderManager(gemini_provider=gemini_mock, ollama_provider=ollama_mock)

    context = RequestContext(privacy_sensitive=True, offline_only=False)
    resp = manager.generate("Analyze private payroll", context=context, mode="auto")

    assert resp.text == "Ollama sensitive answer"
    assert resp.provider == "ollama"
    assert gemini_mock.call_count == 0
    assert ollama_mock.call_count == 1


# ---------------------------------------------------------------------------
# Requirement D: Privacy-sensitive request NEVER calls Gemini
# ---------------------------------------------------------------------------

def test_d_privacy_sensitive_request_never_calls_gemini():
    gemini_mock = MockAIProvider("gemini", return_text="Should never be called")
    ollama_mock = MockAIProvider("ollama", return_text="Secure Ollama response")
    manager = AIProviderManager(gemini_provider=gemini_mock, ollama_provider=ollama_mock)

    context = RequestContext(privacy_sensitive=True)
    # Even if mode is accidentally passed or configured as auto
    resp = manager.generate("Tax ID: 99999", context=context, mode="auto")

    assert gemini_mock.call_count == 0
    assert resp.provider == "ollama"


# ---------------------------------------------------------------------------
# Requirement E: Privacy-sensitive + Ollama unavailable -> FAIL CLOSED
# ---------------------------------------------------------------------------

def test_e_privacy_sensitive_ollama_unavailable_fails_closed():
    gemini_mock = MockAIProvider("gemini", return_text="Gemini answer")
    ollama_mock = MockAIProvider("ollama", should_fail=True, fail_exc=AIConnectionError("Ollama server down"))
    manager = AIProviderManager(gemini_provider=gemini_mock, ollama_provider=ollama_mock)

    context = RequestContext(privacy_sensitive=True)
    with pytest.raises(SecurityPolicyError) as exc_info:
        manager.generate("Confidential financial statement", context=context, mode="auto")

    assert "Fail-closed" in str(exc_info.value)
    # Crucial security verification: Gemini was NEVER called as a fallback
    assert gemini_mock.call_count == 0
    assert ollama_mock.call_count == 1


# ---------------------------------------------------------------------------
# Requirement F: Gemini unavailable + normal cloud-allowed request -> Ollama fallback
# ---------------------------------------------------------------------------

def test_f_gemini_unavailable_normal_cloud_request_falls_back_to_ollama():
    gemini_mock = MockAIProvider("gemini", should_fail=True, fail_exc=AITimeoutError("Gemini timed out"))
    ollama_mock = MockAIProvider("ollama", return_text="Ollama fallback answer")
    manager = AIProviderManager(gemini_provider=gemini_mock, ollama_provider=ollama_mock)

    context = RequestContext(privacy_sensitive=False, offline_only=False)
    resp = manager.generate("Public invoice summary", context=context, mode="auto")

    assert gemini_mock.call_count == 1
    assert ollama_mock.call_count == 1
    assert resp.text == "Ollama fallback answer"
    assert resp.provider == "ollama"


# ---------------------------------------------------------------------------
# Requirement G: Gemini unavailable + privacy-sensitive request -> MUST NOT fallback to Gemini
# ---------------------------------------------------------------------------

def test_g_privacy_sensitive_never_falls_back_to_gemini():
    gemini_mock = MockAIProvider("gemini", return_text="Gemini leakage")
    ollama_mock = MockAIProvider("ollama", should_fail=True, fail_exc=AIProviderError("Ollama failed"))
    manager = AIProviderManager(gemini_provider=gemini_mock, ollama_provider=ollama_mock)

    context = RequestContext(privacy_sensitive=True)
    with pytest.raises(SecurityPolicyError):
        manager.generate("Private account balances", context=context, mode="auto")

    assert gemini_mock.call_count == 0


# ---------------------------------------------------------------------------
# Requirement H: Invalid AI_PROVIDER_MODE rejected
# ---------------------------------------------------------------------------

def test_h_invalid_ai_provider_mode_rejected():
    with pytest.raises(SecurityPolicyError) as exc_info:
        AISecurityPolicy.validate_mode("super_ai")
    assert "Invalid AI_PROVIDER_MODE 'super_ai'" in str(exc_info.value)

    # Via get_provider
    with env_context(AI_PROVIDER_MODE="invalid_mode"):
        with pytest.raises(LLMProviderError):
            get_provider()


# ---------------------------------------------------------------------------
# Requirement I: Missing Gemini key produces controlled error
# ---------------------------------------------------------------------------

def test_i_missing_gemini_key_produces_controlled_error():
    with env_context(GEMINI_API_KEY=None):
        with pytest.raises(AIConfigurationError) as exc_info:
            GeminiProvider()
        assert "GEMINI_API_KEY is not configured" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Requirement J: Missing Ollama configuration produces controlled error where required
# ---------------------------------------------------------------------------

def test_j_missing_ollama_configuration_produces_controlled_error():
    with pytest.raises(AIConfigurationError) as exc_info:
        OllamaProvider(base_url="", model="llama3")
    assert "OLLAMA_BASE_URL must be specified" in str(exc_info.value)

    with pytest.raises(AIConfigurationError) as exc_info:
        OllamaProvider(base_url="http://localhost:11434", model="")
    assert "OLLAMA_MODEL must be specified" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Requirement K: Provider response parsing works
# ---------------------------------------------------------------------------

def test_k_provider_response_parsing_rest(monkeypatch):
    fake_json = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Parsed Gemini response"}]
                }
            }
        ]
    }

    class DummyResponse:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def read(self):
            return json.dumps(fake_json).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: DummyResponse())
    monkeypatch.delitem(sys.modules, "google.generativeai", raising=False)

    provider = GeminiProvider(api_key="test-api-key")
    resp = provider.generate("Hello", system_prompt="Be concise")
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "Parsed Gemini response"
    assert resp.provider == "gemini"


def test_k_provider_response_parsing_ollama(monkeypatch):
    fake_json = {
        "model": "llama3",
        "response": "Parsed Ollama response",
        "done": True,
    }

    class DummyResponse:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def read(self):
            return json.dumps(fake_json).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: DummyResponse())

    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")
    resp = provider.generate("Hello Ollama")
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "Parsed Ollama response"
    assert resp.provider == "ollama"


# ---------------------------------------------------------------------------
# Requirement L: Provider timeout/error is converted into controlled provider failure
# ---------------------------------------------------------------------------

def test_l_provider_timeout_converted_into_controlled_failure(monkeypatch):
    def timeout_urlopen(req, timeout=None):
        raise TimeoutError("Socket connection timed out")

    monkeypatch.setattr("urllib.request.urlopen", timeout_urlopen)

    ollama = OllamaProvider(base_url="http://localhost:11434")
    with pytest.raises(AITimeoutError) as exc_info:
        ollama.generate("Test prompt")
    assert "timed out" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Requirement M: No secrets appear in logs/errors
# ---------------------------------------------------------------------------

def test_m_no_secrets_appear_in_errors(monkeypatch):
    secret_key = "super_secret_secret_12345"
    provider = GeminiProvider(api_key=secret_key)

    import urllib.error
    # Mock HTTPError
    def failing_urlopen(req, timeout=None):
        fp = MagicMock()
        fp.read.return_value = b'{"error": "Unauthorized"}'
        raise urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + secret_key,
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=fp,
        )

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    monkeypatch.delitem(sys.modules, "google.generativeai", raising=False)

    with pytest.raises(AIProviderError) as exc_info:
        provider.generate("Test prompt")

    err_message = str(exc_info.value)
    assert secret_key not in err_message


# ---------------------------------------------------------------------------
# Requirement N: Adapter and get_provider backward compatibility
# ---------------------------------------------------------------------------

def test_n_legacy_get_provider_adapter(monkeypatch):
    with patch("backend.app.llm.provider.get_ai_manager") as mock_get_manager:
        mock_mgr = MagicMock()
        mock_mgr.generate.return_value = ProviderResponse(text="Adapted response", model="gemini-1.5-flash", provider="gemini")
        mock_get_manager.return_value = mock_mgr

        provider = get_provider(privacy_sensitive=False)
        result = provider.generate("System instruction", "User query")
        assert result == "Adapted response"
        mock_mgr.generate.assert_called_once()
