"""Assistant orchestrator – the secure dispatch core.

Flow: USER MESSAGE -> deterministic policy/privacy classification ->
agent dispatch -> verified multi-step tool execution -> response.

Dispatch policy:
  * Safety-critical flows (unblock refusal, security block, card-block
    confirmation) are ALWAYS deterministic and enforced before any LLM
    call. The LLM can never bypass them.
  * Privacy-sensitive requests are NEVER sent to a cloud provider. They are
    served by the local deterministic engine only; the cloud (Gemini) is not
    called at all (fail-closed at the network level).
  * Cloud-allowed requests use the real Gemini agent loop (multi-step
    function calling). If Gemini is unavailable, the deterministic engine
    answers with fully grounded tool output.
  * All tools receive business_id from the authenticated user; the message
    can never override it.

There is intentionally NO unblock/unfreeze/unlock/reactivate card tool.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from .. import models
from . import intents as I
from . import tools
from .responder import compose_response
from .gemini_agent import run_gemini_agent, _step_from_result, MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)

_CONFIRM_TTL_SECONDS = 300  # pending block confirmations expire after 5 min

# Pending block confirmations: card_id -> {user_id, last_four, expires}.
_pending_blocks: Dict[int, Dict] = {}


def register_pending_block(user_id: int, card_id: int, last_four: str) -> None:
    _pending_blocks[card_id] = {
        "user_id": user_id,
        "last_four": last_four,
        "expires": time.time() + _CONFIRM_TTL_SECONDS,
    }


def pop_pending_block(card_id: int, user_id: Optional[int] = None) -> bool:
    """Consume a valid pending confirmation. Returns True when one exists."""
    pending = _pending_blocks.get(card_id)
    if not pending:
        return False
    if time.time() > pending["expires"]:
        _pending_blocks.pop(card_id, None)
        return False
    if user_id is not None and pending["user_id"] != user_id:
        return False
    _pending_blocks.pop(card_id, None)
    return True


def pending_for_user(user_id: int, business_id: int, db: Session) -> Optional[models.Card]:
    """The most recent still-valid pending block card for this user, if any."""
    best = None
    for card_id, pending in list(_pending_blocks.items()):
        if pending["user_id"] != user_id or time.time() > pending["expires"]:
            if time.time() > pending["expires"]:
                _pending_blocks.pop(card_id, None)
            continue
        card = (
            db.query(models.Card)
            .filter(models.Card.id == card_id, models.Card.business_id == business_id)
            .first()
        )
        if card is not None:
            best = card
    return best


def cancel_pending_for_user(user_id: int) -> None:
    for card_id, pending in list(_pending_blocks.items()):
        if pending["user_id"] == user_id:
            _pending_blocks.pop(card_id, None)


HELP_TEXT = (
    "I can help you with:\n"
    "• Accounts and balances — “What's my current account balance?”\n"
    "• Transactions — “Show my last 10 transactions.”\n"
    "• Spending — “How much did I spend on inventory this month?”\n"
    "• Cash flow — “Analyze my cash flow.” or “How long will my cash last?”\n"
    "• Forecasts — “What will my cash flow look like over the next 30 days?”\n"
    "• Anomalies — “Show unusual transactions above RM 500.”\n"
    "• Documents — “Search my uploaded documents for payment terms.”\n"
    "• Notifications, support tickets and cards — including “Block my card ending in 4821.”\n\n"
    "What would you like to do?"
)

UNBLOCK_REFUSAL = (
    "For security reasons, blocked cards cannot be unblocked through the assistant or by AI. "
    "A blocked card stays blocked. If you need a replacement card, please raise a support ticket "
    "and our team will assist you."
)

GREETING_TEXT = (
    "Hello. I'm FinMind, your financial assistant. I can look at your accounts, transactions, "
    "cash flow, forecasts and documents, and I can manage cards, notifications and support tickets. "
    "How can I help?"
)


class AssistantOrchestrator:
    def __init__(self, db: Session):
        self.db = db
        self.resolver = I.IntentResolver(db)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def handle(
        self,
        user: models.User,
        message: str,
        privacy_sensitive: bool = False,
        conversation_id: Optional[int] = None,
    ) -> Dict:
        business_id = user.business_id
        if not business_id:
            return {
                "response": "Your account is not associated with a business yet, so I can't access any financial data.",
                "grounded": False,
                "intent": "NO_BUSINESS",
                "intents": [],
                "steps": [],
                "provider": "deterministic",
                "refused": False,
                "actions": [],
                "conversation_id": conversation_id,
            }

        # --- Policy / privacy classification --------------------------------
        # The explicit UI toggle is authoritative; message markers may only
        # ESCALATE privacy (never downgrade it).
        lmsg = (message or "").lower()
        escalated = any(
            marker in lmsg
            for marker in ("confidential", "don't share", "do not share", "keep this private", "privacy")
        )
        privacy = privacy_sensitive or escalated

        resolved = self.resolver.resolve(message)
        primary = resolved[0].intent if resolved else I.UNKNOWN
        intents_out = [r.intent for r in resolved]

        # --- Refusal family ---------------------------------------------------
        if any(r.intent == I.UNBLOCK_CARD for r in resolved):
            if len(resolved) == 1:
                return self._persisted(user, conversation_id, message, self._refusal(UNBLOCK_REFUSAL, intents_out))
            resolved = [r for r in resolved if r.intent != I.UNBLOCK_CARD]
            refusal_step = {"tool": "policy", "kind": "STATUS", "text_lines": [UNBLOCK_REFUSAL]}
            result = self._execute(user, business_id, resolved, privacy, message)
            result["steps"].append(refusal_step)
            result["response"] = refusal_step["text_lines"][0] + "\n\n" + result["response"]
            return self._persisted(user, conversation_id, message, result)

        if primary == I.SECURITY_BLOCK:
            return self._persisted(
                user, conversation_id, message,
                self._refusal(
                    "For security, I can't execute database or system commands. "
                    "I can only work through FinMind's controlled financial tools.",
                    intents_out,
                ),
            )

        # --- Confirmation flow for pending card blocks -------------------------
        if primary == I.CONFIRM:
            return self._handle_confirm(user, business_id, message, privacy, conversation_id)
        if primary == I.CANCEL:
            cancel_pending_for_user(user.id)
            return self._persisted(
                user, conversation_id, message,
                self._final(message, [], privacy, intro="Pending action cancelled.",
                            grounded=False, intents=[I.CANCEL], refused=False, actions=[]),
            )

        # --- BLOCK_CARD requires explicit confirmation (deterministic) ---------
        if primary == I.BLOCK_CARD and len(resolved) == 1:
            return self._handle_block_request(user, business_id, resolved[0], privacy, intents_out, conversation_id)

        # --- Gemini agent (cloud-allowed, conversation-aware) ------------------
        if not privacy and not self._llm_disabled():
            history = self._load_history(user, conversation_id)
            llm_result = run_gemini_agent(
                db=self.db,
                user=user,
                message=message,
                history=history,
                privacy_sensitive=privacy,
            )
            if llm_result is not None:
                llm_result["intents"] = intents_out
                llm_result["intent"] = intents_out[0] if intents_out else "UNKNOWN"
                llm_result["refused"] = False
                llm_result["actions"] = []
                return self._persisted(user, conversation_id, message, llm_result)

        # --- Deterministic engine (privacy mode, Gemini unavailable, or fallback)
        if primary == I.GREETING:
            return self._persisted(
                user, conversation_id, message,
                self._final(message, [], privacy, intro=GREETING_TEXT,
                            grounded=False, intents=intents_out, refused=False, actions=[]),
            )
        if primary == I.HELP:
            return self._persisted(
                user, conversation_id, message,
                self._final(message, [], privacy, intro=HELP_TEXT,
                            grounded=False, intents=[I.HELP], refused=False, actions=[]),
            )
        if primary == I.UNKNOWN and all(r.intent == I.UNKNOWN for r in resolved):
            clarification = (
                "I'm not sure what you'd like me to do. "
                + HELP_TEXT.split("What would you like")[0].rstrip() +
                "What would you like to do?"
            )
            return self._persisted(
                user, conversation_id, message,
                self._final(message, [], privacy, intro=clarification,
                            grounded=False, intents=[I.UNKNOWN], refused=False, actions=[]),
            )

        result = self._execute(user, business_id, resolved, privacy, message)
        return self._persisted(user, conversation_id, message, result)

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------
    def _conversation(self, user: models.User, conversation_id: Optional[int]) -> models.AssistantConversation:
        if conversation_id is not None:
            convo = (
                self.db.query(models.AssistantConversation)
                .filter(
                    models.AssistantConversation.id == int(conversation_id),
                    models.AssistantConversation.user_id == user.id,
                    models.AssistantConversation.business_id == user.business_id,
                )
                .first()
            )
            if convo is not None:
                return convo
        convo = models.AssistantConversation(user_id=user.id, business_id=user.business_id)
        self.db.add(convo)
        self.db.flush()
        return convo

    def _load_history(self, user: models.User, conversation_id: Optional[int]) -> List[Dict[str, str]]:
        if conversation_id is None:
            return []
        convo = (
            self.db.query(models.AssistantConversation)
            .filter(
                models.AssistantConversation.id == int(conversation_id),
                models.AssistantConversation.user_id == user.id,
            )
            .first()
        )
        if convo is None:
            return []
        msgs = (
            self.db.query(models.AssistantMessage)
            .filter(models.AssistantMessage.conversation_id == convo.id)
            .order_by(models.AssistantMessage.id.desc())
            .limit(MAX_HISTORY_MESSAGES)
            .all()
        )
        history = [{"role": m.role, "content": m.content} for m in reversed(msgs)]
        return history

    def _persisted(self, user: models.User, conversation_id: Optional[int], message: str, result: Dict) -> Dict:
        try:
            convo = self._conversation(user, conversation_id)
            self.db.add(models.AssistantMessage(
                conversation_id=convo.id, role="user", content=message, meta=None,
            ))
            meta = {
                "provider": result.get("provider"),
                "refused": bool(result.get("refused")),
                "grounded": bool(result.get("grounded")),
                "intents": result.get("intents", []),
                "steps": [
                    {"tool": s.get("tool"), "kind": s.get("kind"), "text_lines": s.get("text_lines", [])}
                    for s in result.get("steps", [])
                ],
            }
            self.db.add(models.AssistantMessage(
                conversation_id=convo.id, role="assistant", content=result["response"], meta=meta,
            ))
            self.db.commit()
            result["conversation_id"] = convo.id
        except Exception:
            logger.exception("Conversation persistence failed; continuing without memory")
            self.db.rollback()
            result.setdefault("conversation_id", conversation_id)
        return result

    # ------------------------------------------------------------------
    # Block-card workflow (deterministic – never delegated to the LLM)
    # ------------------------------------------------------------------
    def _handle_block_request(self, user, business_id, resolved_intent, privacy, intents_out, conversation_id) -> Dict:
        last_four = resolved_intent.slots.get("last_four")
        cards = self.db.query(models.Card).filter(models.Card.business_id == business_id).all()

        if not cards:
            return self._persisted(
                user, conversation_id, resolved_intent.text,
                self._final(resolved_intent.text, [], privacy,
                            intro="Your business has no cards issued, so there is nothing to block.",
                            grounded=True, intents=intents_out, refused=False, actions=[]),
            )

        if not last_four:
            card_list = ", ".join(f"•••• {c.last_four} ({c.status})" for c in cards)
            return self._persisted(
                user, conversation_id, resolved_intent.text,
                self._final(resolved_intent.text, [], privacy,
                            intro=f"Which card would you like to block? Your cards: {card_list}.",
                            grounded=True, intents=intents_out, refused=False, actions=[]),
            )

        matches = [c for c in cards if c.last_four == last_four]
        if not matches:
            card_list = ", ".join(f"•••• {c.last_four} ({c.status})" for c in cards)
            return self._persisted(
                user, conversation_id, resolved_intent.text,
                self._final(resolved_intent.text, [], privacy,
                            intro=f"You don't have a card ending in {last_four}. Your cards: {card_list}.",
                            grounded=True, intents=intents_out, refused=False, actions=[]),
            )
        card = matches[0]
        if card.status == "blocked":
            return self._persisted(
                user, conversation_id, resolved_intent.text,
                self._final(resolved_intent.text, [], privacy,
                            intro=f"Card •••• {last_four} is already blocked.",
                            grounded=True, intents=intents_out, refused=False, actions=[]),
            )

        register_pending_block(user.id, card.id, last_four)
        action = {"type": "confirm_block", "last_four": last_four, "card_id": card.id}
        intro = (
            f"You're about to block card •••• {last_four}. Blocking takes effect immediately and "
            "cannot be undone by the assistant. Reply “confirm” or press the button to proceed."
        )
        return self._persisted(
            user, conversation_id, resolved_intent.text,
            self._final(resolved_intent.text, [], privacy, intro=intro,
                        grounded=True, intents=intents_out, refused=False, actions=[action]),
        )

    def _handle_confirm(self, user, business_id, message, privacy, conversation_id) -> Dict:
        card = pending_for_user(user.id, business_id, self.db)
        if card is None:
            return self._persisted(
                user, conversation_id, message,
                self._final(message, [], privacy,
                            intro="There's no pending action to confirm. Tell me what you'd like to do.",
                            grounded=False, intents=[I.CONFIRM], refused=False, actions=[]),
            )
        result = tools.block_card(self.db, business_id, {"last_four": card.last_four})
        step = _step_from_result("block_card", result)
        if not result.get("success"):
            if result.get("error_code") == "NOT_FOUND":
                intro = result["message"]
            else:
                intro = "The confirmation window has expired. Please request the action again."
            return self._persisted(
                user, conversation_id, message,
                self._final(message, [], privacy, intro=intro,
                            grounded=False, intents=[I.BLOCK_CARD], refused=False, actions=[]),
            )
        return self._persisted(
            user, conversation_id, message,
            self._final(message, [step], privacy, grounded=True,
                        intents=[I.BLOCK_CARD], refused=False, actions=[]),
        )

    # ------------------------------------------------------------------
    # Multi-step execution for ordinary intents (deterministic engine)
    # ------------------------------------------------------------------
    def _execute(self, user, business_id, resolved, privacy, message) -> Dict:
        intents_out = [r.intent for r in resolved]
        steps: List[Dict] = []
        errors: List[str] = []

        for r in resolved:
            tool_name, args = self._map_intent(r)
            if tool_name is None:
                continue
            fn = tools.TOOL_REGISTRY[tool_name]
            try:
                contract = fn(self.db, business_id, args)
            except tools.ToolError as exc:
                errors.append(str(exc))
                continue
            except Exception:
                logger.exception("Deterministic tool %s failed", tool_name)
                errors.append("That request couldn't be completed. Please try again.")
                continue
            steps.append(_step_from_result(tool_name, contract))
            if not contract.get("success", True):
                errors.append(contract.get("message", "The request could not be completed."))

        intro = None
        if errors:
            intro = "\n".join(errors)
        return self._final(
            message, steps, privacy, intro=intro,
            grounded=bool(steps), intents=intents_out, refused=False, actions=[],
        )

    @staticmethod
    def _map_intent(r) -> tuple:
        s = r.slots
        mapping = {
            I.ACCOUNT_BALANCE: ("get_account_balance", {}),
            I.ACCOUNTS_LIST: ("get_accounts", {}),
            I.TRANSACTIONS: ("get_transactions", {
                "limit": s.get("limit") or 10,
                "type": s.get("type"),
                "category": s.get("category"),
                "start": s.get("start"),
                "end": s.get("end"),
                "min_amount": s.get("min_amount"),
            }),
            I.SPENDING_ANALYSIS: ("analyze_spending", {
                "start": s.get("start"), "end": s.get("end"),
                "label": s.get("label"), "category": s.get("category"),
            }),
            I.SPENDING_BY_CATEGORY: ("analyze_spending_by_category", {
                "start": s.get("start"), "end": s.get("end"), "label": s.get("label"),
            }),
            I.MERCHANT_ANALYSIS: ("analyze_merchants", {
                "start": s.get("start"), "end": s.get("end"), "label": s.get("label"),
            }),
            I.CASH_FLOW_ANALYSIS: ("analyze_cash_flow", {
                "start": s.get("start"), "end": s.get("end"), "label": s.get("label"),
            }),
            I.CASH_RUNWAY: ("get_cash_runway", {}),
            I.FORECAST: ("get_forecast", {"days": s.get("days") or 30}),
            I.ANOMALIES: ("get_anomalies", {"min_amount": s.get("min_amount"), "limit": 10}),
            I.DOCUMENTS_LIST: ("list_documents", {}),
            I.DOCUMENT_SEARCH: ("search_knowledge", {"query": s.get("query") or r.text}),
            I.DOCUMENT_RAG: ("search_knowledge", {"query": s.get("query") or r.text}),
            I.NOTIFICATIONS_LIST: ("get_notifications", {}),
            I.NOTIFICATIONS_MARK_READ: ("mark_notification_read", {"all": s.get("all", True)}),
            I.TICKETS_LIST: ("list_helpdesk_tickets", {}),
            I.CREATE_TICKET: ("create_helpdesk_ticket", {
                "description": s.get("description") or r.text, "title": s.get("title"),
            }),
            I.CARDS_LIST: ("list_cards", {}),
        }
        return mapping.get(r.intent, (None, None))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _llm_disabled() -> bool:
        import os
        return os.getenv("FINMIND_DISABLE_LLM_AGENT", "").strip() in ("1", "true", "yes")

    def _refusal(self, text: str, intents_out) -> Dict:
        return {
            "response": text,
            "grounded": True,
            "intent": intents_out[0] if intents_out else "UNBLOCK_CARD",
            "intents": intents_out,
            "steps": [{"tool": "policy", "kind": "STATUS", "text_lines": [text]}],
            "provider": "deterministic",
            "refused": True,
            "actions": [],
        }

    def _final(self, message, steps, privacy, intro=None, grounded=False,
               intents=None, refused=False, actions=None) -> Dict:
        if steps:
            grounded = grounded and all(step.get("kind") != "EVIDENCE" or step.get("grounded", True) for step in steps)
        composed = compose_response(
            message, steps, privacy_sensitive=privacy, intro=intro, grounded=grounded,
            llm_explain_enabled=False,
        )
        return {
            "response": composed["response"],
            "grounded": composed["grounded"],
            "intent": intents[0] if intents else "UNKNOWN",
            "intents": intents or [],
            "steps": composed["steps"],
            "provider": composed["provider"],
            "refused": refused,
            "actions": actions or [],
        }
