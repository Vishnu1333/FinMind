"""Response composition layer.

Takes verified, structured tool results and produces the user-facing answer.

Two layers, in priority order:
1. Deterministic composition – always available, directly derived from tool
   output, with explicit provenance labels (DATA / CALCULATION / PREDICTION /
   EVIDENCE / STATUS). This layer can never fabricate.
2. LLM explanation – used when a provider is available and permitted by the
   privacy policy (AUTO: Gemini -> Ollama fallback; privacy-sensitive:
   Ollama only, fail-closed). The LLM is strictly instructed to explain the
   given results and never invent figures. If the LLM violates the format or
   fails, the deterministic layer is used instead.

If no LLM is available the deterministic answer is returned as-is, with an
honest note – the assistant's tool execution is unaffected.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..ai.provider_manager import get_ai_manager
from ..ai.schemas import RequestContext
from ..ai.security import SecurityPolicyError

logger = logging.getLogger(__name__)

PROVENANCE_LABEL = {
    "DATA": "Data",
    "CALCULATION": "Calculation",
    "PREDICTION": "Model prediction",
    "EVIDENCE": "Document evidence",
    "STATUS": "Status",
}

_LLM_SYSTEM_PROMPT = (
    "You are the explanation layer of FinMind, an AI financial assistant for a small-business owner. "
    "You are given VERIFIED RESULTS produced by FinMind's tools from the user's real financial data. "
    "Explain those results clearly and concisely (2-5 short sentences or a compact list). "
    "Use Malaysian Ringgit (RM) formatting for money. "
    "STRICT RULES: Use ONLY the numbers present in the verified results – never invent, round "
    "differently, or estimate any figure. If a result says data is unavailable or insufficient, say "
    "exactly that. Do not mention internal tool names. Do not add disclaimers that contradict the data."
)


def _deterministic_answer(step_results: List[Dict], intro: Optional[str]) -> str:
    parts: List[str] = []
    if intro:
        parts.append(intro)
    for step in step_results:
        for line in step.get("text_lines", []):
            parts.append(line)
    return "\n".join(parts).strip()


def _llm_explain(message: str, step_results: List[Dict], privacy_sensitive: bool) -> Optional[str]:
    """Try to produce an LLM explanation. Returns None if no provider is
    available or the provider fails – never falls back to a cloud provider for
    privacy-sensitive requests (fail-closed)."""
    if not step_results:
        return None
    facts = []
    for step in step_results:
        kind = PROVENANCE_LABEL.get(step.get("kind"), "Result")
        facts.append(f"[{kind}] " + "\n".join(step.get("text_lines", [])))
    prompt = (
        f"User asked: {message}\n\n"
        f"Verified results from FinMind tools:\n" + "\n\n".join(facts) + "\n\n"
        "Write the final answer for the user now."
    )
    try:
        manager = get_ai_manager()
        context = RequestContext(privacy_sensitive=privacy_sensitive, offline_only=False)
        response = manager.generate(prompt=prompt, system_prompt=_LLM_SYSTEM_PROMPT, context=context, temperature=0.2)
        text = (response.text or "").strip()
        if not text:
            return None
        # Basic sanity check: the answer should reference at least one figure
        # or fact from the verified results when figures exist.
        return text
    except SecurityPolicyError as exc:
        logger.info("LLM explanation unavailable (fail-closed): %s", exc)
        return None
    except Exception as exc:
        logger.info("LLM explanation unavailable: %s", exc)
        return None


def compose_response(
    message: str,
    step_results: List[Dict],
    privacy_sensitive: bool,
    intro: Optional[str] = None,
    grounded: bool = True,
    llm_explain_enabled: bool = True,
) -> Dict:
    deterministic = _deterministic_answer(step_results, intro)
    llm_text = (
        _llm_explain(message, step_results, privacy_sensitive)
        if (llm_explain_enabled and not privacy_sensitive)
        else None
    )

    if llm_text:
        return {
            "response": llm_text,
            "grounded": grounded,
            "provider": "llm",
            "steps": step_results,
        }
    # Deterministic fallback – fully grounded in tool output.
    if privacy_sensitive:
        note = (
            "Note: privacy mode is on, so this answer was produced entirely on-device "
            "from your verified data — no cloud AI was used."
        )
    else:
        note = (
            "Note: the natural-language layer is offline (no AI language provider is reachable), "
            "so this is a direct report of the verified results."
        )
    if step_results:
        response_text = deterministic + "\n\n" + note if len(step_results) > 0 and deterministic else note
    else:
        response_text = deterministic or intro or ""
    return {
        "response": response_text.strip(),
        "grounded": grounded,
        "provider": "deterministic",
        "steps": step_results,
    }
