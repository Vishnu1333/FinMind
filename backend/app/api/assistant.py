from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pydantic

from .. import db as database, models
from ..dependencies import get_current_user
from ..agent.orchestrator import AssistantOrchestrator

router = APIRouter(prefix="/assistant", tags=["assistant"])


class ChatRequest(pydantic.BaseModel):
    message: str = pydantic.Field(..., min_length=1, max_length=2000)
    privacy_sensitive: bool = False


class StepResult(pydantic.BaseModel):
    tool: str
    kind: str
    text_lines: list = []
    extra: dict = {}

    class Config:
        from_attributes = True


class PendingAction(pydantic.BaseModel):
    type: str
    last_four: str
    card_id: int


class ChatResponse(pydantic.BaseModel):
    response: str
    grounded: bool = False
    intent: str = "UNKNOWN"
    intents: list = []
    provider: str = "deterministic"
    refused: bool = False
    actions: list = []
    steps: list = []


@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(
    request: ChatRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    """Authenticated assistant endpoint.

    The user message is resolved to deterministic intents and executed through
    controlled, business-scoped tools. The response is grounded in verified
    tool output; the LLM only provides the natural-language explanation layer
    (when a provider is available and permitted by the privacy policy).
    """
    if not current_user.business_id:
        raise HTTPException(status_code=403, detail="User not associated with a business")

    orchestrator = AssistantOrchestrator(db)
    try:
        result = orchestrator.handle(
            user=current_user,
            message=request.message,
            privacy_sensitive=request.privacy_sensitive,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The assistant could not process this request.")

    # Normalise steps for the response contract (JSON-safe, compact).
    _BULKY_KEYS = {
        "anomalies", "sources", "breakdown", "merchants", "notifications",
        "tickets", "cards", "accounts", "runway", "evidence",
    }
    safe_steps = []
    for step in result.get("steps", []):
        safe_steps.append({
            "tool": step.get("tool"),
            "kind": step.get("kind"),
            "text_lines": step.get("text_lines", []),
            "data": {
                k: v for k, v in step.items()
                if k not in _BULKY_KEYS and k not in ("tool", "kind", "text_lines")
                and isinstance(v, (str, int, float, bool, type(None)))
            },
        })

    return ChatResponse(
        response=result["response"],
        grounded=result.get("grounded", False),
        intent=result.get("intent", "UNKNOWN"),
        intents=result.get("intents", []),
        provider=result.get("provider", "deterministic"),
        refused=result.get("refused", False),
        actions=result.get("actions", []),
        steps=safe_steps,
    )
