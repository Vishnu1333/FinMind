import json
from unittest.mock import patch

from backend.app.ai.gemini_provider import GeminiProvider


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({
            "candidates": [
                {"content": {"parts": [{"text": "gateway-ok"}]}}
            ]
        }).encode("utf-8")


def test_gemini_provider_routes_through_neon_ai_gateway(monkeypatch):
    monkeypatch.setenv("NEON_AI_GATEWAY_BASE_URL", "https://branch-api.ai.example")
    monkeypatch.setenv("NEON_AI_GATEWAY_TOKEN", "test-gateway-token")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("NEON_AI_GATEWAY_MODEL", raising=False)

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _FakeResponse()

    provider = GeminiProvider()
    with patch("backend.app.ai.gemini_provider.urllib.request.urlopen", side_effect=fake_urlopen):
        response = provider.generate("test prompt")

    assert response.text == "gateway-ok"
    assert provider.model == "gemini-3-flash"
    assert captured["url"] == (
        "https://branch-api.ai.example/gemini/v1beta/models/gemini-3-flash:generateContent"
    )
    assert captured["auth"] == "Bearer test-gateway-token"
    assert "test-gateway-token" not in captured["url"]
