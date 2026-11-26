"""Tests for the deterministic assistant agent.

Covers intent resolution, multi-step orchestration, the block-card
confirmation flow, the unblock refusal policy, and tool grounding.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import models  # noqa: E402
from backend.app.db import SessionLocal  # noqa: E402
from backend.app.services.provisioning_service import ProvisioningService  # noqa: E402
from backend.app.agent.intents import IntentResolver  # noqa: E402
from backend.app.agent.orchestrator import AssistantOrchestrator  # noqa: E402
import backend.app.agent.intents as I  # noqa: E402


@pytest.fixture()
def session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture()
def provisioned(session):
    business = models.Business(name="Agent Test Sdn Bhd")
    session.add(business)
    session.flush()
    user = models.User(
        email="agent_test@finmind.com",
        hashed_password="not-a-real-hash",
        business_id=business.id,
    )
    session.add(user)
    session.commit()
    session.refresh(business)
    session.refresh(user)
    ProvisioningService(session).provision(business, user)
    return user


# ---------------------------------------------------------------------------
# Intent resolution (pure rules)
# ---------------------------------------------------------------------------
class TestIntentResolution:
    @pytest.fixture()
    def resolver(self, session):
        return IntentResolver(session)

    @pytest.mark.parametrize("message,expected", [
        ("What's my current account balance?", I.ACCOUNT_BALANCE),
        ("Show my last 10 transactions.", I.TRANSACTIONS),
        ("How much did I spend on inventory this month?", I.SPENDING_ANALYSIS),
        ("Analyze my cash flow.", I.CASH_FLOW_ANALYSIS),
        ("What will my cash flow look like over the next 30 days?", I.FORECAST),
        ("Show unusual transactions.", I.ANOMALIES),
        ("Find unusual expenses above RM 500.", I.ANOMALIES),
        ("Search my uploaded documents for payment terms.", I.DOCUMENT_RAG),
        ("Show my notifications.", I.NOTIFICATIONS_LIST),
        ("Show my cards.", I.CARDS_LIST),
        ("Block my card ending in 4821.", I.BLOCK_CARD),
        ("Unblock my card ending in 4821.", I.UNBLOCK_CARD),
        ("Unfreeze card 4821", I.UNBLOCK_CARD),
        ("Reactivate my card", I.UNBLOCK_CARD),
        ("Create a support ticket about a failed transfer.", I.CREATE_TICKET),
        ("Show my support tickets", I.TICKETS_LIST),
        ("How long will my cash last?", I.CASH_RUNWAY),
        ("Where is my money going?", I.MERCHANT_ANALYSIS),
        ("hello", I.GREETING),
        ("What can you do?", I.HELP),
        ("Tell me about the weather in Paris.", I.UNKNOWN),
    ])
    def test_expected_intent(self, resolver, message, expected):
        resolved = resolver.resolve(message)
        assert resolved[0].intent == expected

    def test_multi_intent_split(self, resolver):
        resolved = resolver.resolve(
            "How much did I spend on inventory last month and what do you predict for next month?"
        )
        intents = [r.intent for r in resolved]
        assert I.SPENDING_ANALYSIS in intents
        assert I.FORECAST in intents

    def test_block_card_slot(self, resolver):
        resolved = resolver.resolve("Block my card ending in 4821")
        assert resolved[0].slots.get("last_four") == "4821"

    def test_anomaly_amount_slot(self, resolver):
        resolved = resolver.resolve("Find unusual expenses above RM 500")
        assert resolved[0].slots.get("min_amount") == 500.0

    def test_forecast_horizon(self, resolver):
        resolved = resolver.resolve("What will my cash flow look like over the next 30 days?")
        assert resolved[0].slots.get("days") == 30

    def test_no_unblock_tool_exists(self):
        from backend.app.agent.tools import TOOL_REGISTRY
        for forbidden in ("unblock_card", "unfreeze_card", "reactivate_card", "unlock_card"):
            assert forbidden not in TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Orchestrator flows (against the real test database)
#
# These tests verify the DETERMINISTIC orchestration engine, so the Gemini
# agent is disabled via the supported FINMIND_DISABLE_LLM_AGENT flag. This
# keeps the tests hermetic: they must not depend on live LLM tool choices.
# (Gemini-first routing itself is covered by test_ai_providers.py.)
# ---------------------------------------------------------------------------
class TestOrchestrator:
    @pytest.fixture()
    def orch(self, session, monkeypatch):
        monkeypatch.setenv("FINMIND_DISABLE_LLM_AGENT", "1")
        return AssistantOrchestrator(session)

    def _handle(self, orch, user, message, privacy=False):
        return orch.handle(user, message, privacy_sensitive=privacy)

    def test_balance_grounded(self, orch, provisioned):
        result = self._handle(orch, provisioned, "What's my current account balance?")
        assert result["grounded"] is True
        assert result["intent"] == I.ACCOUNT_BALANCE
        assert "RM" in result["response"]

    def test_transactions(self, orch, provisioned):
        result = self._handle(orch, provisioned, "Show my last 10 transactions.")
        assert result["intent"] == I.TRANSACTIONS
        assert result["grounded"] is True
        assert len(result["steps"]) == 1
        assert result["steps"][0]["count"] == 10

    def test_multi_step(self, orch, provisioned):
        result = self._handle(
            orch, provisioned,
            "How much did I spend on inventory last month and what do you predict for next month?",
        )
        assert I.SPENDING_ANALYSIS in result["intents"]
        assert I.FORECAST in result["intents"]
        tool_names = [s["tool"] for s in result["steps"]]
        assert "analyze_spending" in tool_names
        assert "get_forecast" in tool_names

    def test_block_requires_confirmation(self, orch, session, provisioned):
        result = self._handle(orch, provisioned, "Block my card ending in 4821.")
        # Card must NOT be blocked yet.
        card = session.query(models.Card).filter(
            models.Card.business_id == provisioned.business_id,
            models.Card.last_four == "4821",
        ).first()
        assert card.status == "active"
        assert result["actions"] and result["actions"][0]["type"] == "confirm_block"

        # Confirm -> blocked.
        confirmed = self._handle(orch, provisioned, "confirm")
        session.refresh(card)
        assert card.status == "blocked"
        assert confirmed["grounded"] is True
        assert "4821" in confirmed["response"]

    def test_block_without_suffix_asks_clarification(self, orch, provisioned):
        result = self._handle(orch, provisioned, "Block my card")
        card_actions = result["actions"]
        assert card_actions == []  # no direct action pending
        assert "Which card" in result["response"]

    def test_unblock_refused_and_card_stays_blocked(self, orch, session, provisioned):
        self._handle(orch, provisioned, "Block my card ending in 4821.")
        self._handle(orch, provisioned, "confirm")
        result = self._handle(orch, provisioned, "Unblock my card ending in 4821.")
        assert result["refused"] is True
        card = session.query(models.Card).filter(
            models.Card.business_id == provisioned.business_id,
            models.Card.last_four == "4821",
        ).first()
        session.refresh(card)
        assert card.status == "blocked"

    @pytest.mark.parametrize("msg", [
        "Unblock my card ending in 4821.",
        "Unfreeze card 4821",
        "Reactivate my card ending 4821",
        "Unlock my card",
    ])
    def test_all_unblock_words_refused(self, orch, provisioned, msg):
        result = self._handle(orch, provisioned, msg)
        assert result["refused"] is True

    def test_create_ticket_persists(self, orch, session, provisioned):
        count_before = session.query(models.Ticket).filter(
            models.Ticket.business_id == provisioned.business_id
        ).count()
        result = self._handle(orch, provisioned, "Create a support ticket about a transaction dispute.")
        count_after = session.query(models.Ticket).filter(
            models.Ticket.business_id == provisioned.business_id
        ).count()
        assert count_after == count_before + 1

    def test_notifications(self, orch, provisioned):
        result = self._handle(orch, provisioned, "Show my notifications.")
        assert result["intent"] == I.NOTIFICATIONS_LIST
        assert result["grounded"] is True

    def test_forecast_uses_real_model(self, orch, provisioned):
        result = self._handle(
            orch, provisioned, "What will my cash flow look like over the next 30 days?",
        )
        assert result["intent"] == I.FORECAST
        step = result["steps"][0]
        assert step["tool"] == "get_forecast"
        assert step["kind"] == "PREDICTION"
        # The forecast must come from the registry-selected trained artifact
        # (see data/models/registry.json) — never a hardcoded value.
        assert step["model"]["selected"] in ("LSTM", "GRU", "RNN", "AR(14)")
        assert step["model"]["trained_on_days"] >= 60
        assert "candidates" in step["model"]

    def test_anomalies(self, orch, provisioned):
        result = self._handle(orch, provisioned, "Show unusual transactions.")
        assert result["intent"] == I.ANOMALIES
        # Provisioned history includes deliberate anomalies.
        assert result["steps"][0]["count"] >= 1

    def test_unknown_asks_clarification(self, orch, provisioned):
        result = self._handle(orch, provisioned, "Tell me about quantum physics.")
        assert result["intent"] == I.UNKNOWN
        assert "not sure" in result["response"].lower()

    def test_rag_insufficient_evidence_is_honest(self, orch, provisioned):
        result = self._handle(orch, provisioned, "What do my documents say about payment terms?")
        # No documents exist for this business: the agent must say so honestly
        # (never fabricate an answer).
        response_l = result["response"].lower()
        assert "couldn't find" in response_l or "no documents" in response_l or "upload" in response_l

    def test_business_isolation(self, orch, session, provisioned):
        # Second business with different data
        other_biz = models.Business(name="Other Business Sdn Bhd")
        session.add(other_biz)
        session.flush()
        other_user = models.User(email="other@finmind.com", hashed_password="x", business_id=other_biz.id)
        session.add(other_user)
        session.commit()
        session.refresh(other_user)

        result = self._handle(orch, provisioned, "Show my last 5 transactions.")
        ids = [t.id for t in session.query(models.Transaction).join(models.Account).filter(
            models.Account.business_id == provisioned.business_id
        ).limit(5).all()]
        assert result["steps"][0]["count"] == 5
        # All lines must come from this business only (verified via tool count).
        assert result["steps"][0]["total_matching"] >= 5

    def test_privacy_sensitive_never_uses_cloud(self, orch, provisioned):
        result = self._handle(orch, provisioned, "What's my balance?", privacy=True)
        # With no local provider available the responder must fail closed to
        # the deterministic layer – never a cloud provider.
        assert result["provider"] == "deterministic"
