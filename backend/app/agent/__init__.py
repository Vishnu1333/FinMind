"""backend/app/agent package – the FinMind assistant execution architecture.

USER -> AUTHENTICATED ASSISTANT API -> INTENT RESOLVER -> POLICY/PRIVACY
     -> AGENT ORCHESTRATOR -> SPECIALIST AGENT -> CONTROLLED TOOL
     -> BACKEND SERVICE -> DATABASE / ML / RAG -> RESULT
     -> LLM EXPLANATION (or deterministic structured fallback)
     -> USER
"""

from .orchestrator import AssistantOrchestrator  # noqa: F401
