"""Deterministic intent resolution for the FinMind assistant.

The resolver NEVER lets the LLM decide what action to take. It uses ordered
rule patterns and slot extraction (limits, timeframes, categories, card
suffixes, amounts) to produce a structured plan. Ambiguous or unknown input
yields an explicit clarification request instead of a guessed action.

Multi-intent input ("A and B") is split into sequential steps so the
orchestrator can run controlled multi-step workflows.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from .. import models

# ---------------------------------------------------------------------------
# Intent constants
# ---------------------------------------------------------------------------
GREETING = "GREETING"
HELP = "HELP"
ACCOUNT_BALANCE = "ACCOUNT_BALANCE"
ACCOUNTS_LIST = "ACCOUNTS_LIST"
TRANSACTIONS = "TRANSACTIONS"
SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
SPENDING_BY_CATEGORY = "SPENDING_BY_CATEGORY"
MERCHANT_ANALYSIS = "MERCHANT_ANALYSIS"
CASH_FLOW_ANALYSIS = "CASH_FLOW_ANALYSIS"
CASH_RUNWAY = "CASH_RUNWAY"
FORECAST = "FORECAST"
ANOMALIES = "ANOMALIES"
DOCUMENTS_LIST = "DOCUMENTS_LIST"
DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
DOCUMENT_RAG = "DOCUMENT_RAG"
NOTIFICATIONS_LIST = "NOTIFICATIONS_LIST"
NOTIFICATIONS_MARK_READ = "NOTIFICATIONS_MARK_READ"
TICKETS_LIST = "TICKETS_LIST"
CREATE_TICKET = "CREATE_TICKET"
CARDS_LIST = "CARDS_LIST"
BLOCK_CARD = "BLOCK_CARD"
UNBLOCK_CARD = "UNBLOCK_CARD"  # recognised only to be REFUSED
CONFIRM = "CONFIRM"
CANCEL = "CANCEL"
SECURITY_BLOCK = "SECURITY_BLOCK"
UNKNOWN = "UNKNOWN"

CARD_UN_PATTERNS = r"un\s*-?\s*block|unblock|un\s*-?\s*freeze|unfreeze|un\s*-?\s*lock|unlock|re\s*-?\s*activate|reactivate|re\s*-?\s*enable|reinstate|restore"
CARD_BLOCK_PATTERN = r"\b(block|freeze|lock)\b"

# Direct SQL command attempts are never treated as data requests.
_SQL_COMMAND = re.compile(
    r"\b(drop|delete|truncate|alter|insert|update|grant|revoke)\b[\s,`'\"]+"
    r"(table|tables|database|databases|schema|column|columns|from|into)\b",
    re.IGNORECASE,
)


class ResolvedIntent:
    def __init__(self, intent: str, slots: Optional[dict] = None, text: str = ""):
        self.intent = intent
        self.slots = slots or {}
        self.text = text

    def __repr__(self):
        return f"ResolvedIntent({self.intent}, {self.slots})"


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
}


class IntentResolver:
    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Slot extractors
    # ------------------------------------------------------------------
    def _categories(self) -> List[str]:
        return [c.name for c in self.db.query(models.Category).all()]

    @staticmethod
    def _extract_limit(text: str) -> Optional[int]:
        m = re.search(r"\b(?:last|recent|latest|top|show me)\s+(\d{1,3})\b", text)
        if m:
            return max(1, min(int(m.group(1)), 100))
        m = re.search(r"\b(\d{1,3})\s+(?:transactions|expenses|payments)\b", text)
        if m:
            return max(1, min(int(m.group(1)), 100))
        for word, n in _NUM_WORDS.items():
            if re.search(rf"\b{word}\s+transactions\b", text):
                return n
        return None

    @staticmethod
    def _extract_card_suffix(text: str) -> Optional[str]:
        m = re.search(r"\b(?:card|cards)?\s*(?:ending|end(?:ing)? in|with)?\s*[\*\-#x ]*(\d{4})\b", text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_amount(text: str) -> Optional[float]:
        m = re.search(r"(?:rm|myr|\$)\s*(\d+(?:[.,]\d+)?)", text, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))
        m = re.search(
            r"\b(?:above|over|exceeding|more than|greater than|at least|minimum of)\s+(?:rm\s*)?(\d+(?:[.,]\d+)?)",
            text, re.IGNORECASE,
        )
        if m:
            return float(m.group(1).replace(",", ""))
        return None

    @staticmethod
    def _extract_forecast_days(text: str) -> int:
        m = re.search(r"\bnext\s+(\d{1,3})\s*days?\b", text, re.IGNORECASE)
        if m:
            return max(1, min(int(m.group(1)), 90))
        if re.search(r"\bnext\s+(?:month|30\s*days)\b", text, re.IGNORECASE):
            return 30
        if re.search(r"\bnext\s+week\b", text, re.IGNORECASE):
            return 7
        m = re.search(r"\b(\d{1,3})\s*[- ]?\s*day\b", text, re.IGNORECASE)
        if m:
            return max(1, min(int(m.group(1)), 90))
        return 30

    @staticmethod
    def _resolve_period(text: str):
        """Return (start_dt, end_dt, label) for relative timeframes."""
        now = datetime.utcnow()
        text_l = text.lower()

        if "last month" in text_l or "previous month" in text_l:
            first_of_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = first_of_this
            start = (first_of_this - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return start, end, "last month"
        if "this month" in text_l or "current month" in text_l:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return start, now, "this month"
        m = re.search(r"last\s+(\d{1,3})\s*days?", text_l)
        if m:
            n = int(m.group(1))
            return now - timedelta(days=n), now, f"the last {n} days"
        if "last week" in text_l:
            start = now - timedelta(days=7)
            return start, now, "the last 7 days"
        if "this week" in text_l:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, now, "this week"
        if "today" in text_l:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, now, "today"
        if "yesterday" in text_l:
            y = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return y, y + timedelta(days=1), "yesterday"
        # Default: trailing 30 days
        return now - timedelta(days=30), now, "the last 30 days"

    def _extract_category(self, text: str) -> Optional[str]:
        text_l = text.lower()
        for name in self._categories():
            nl = name.lower()
            if re.search(rf"\b{re.escape(nl)}\b", text_l) or re.search(
                rf"\b{re.escape(nl.rstrip('s'))}(?:s|es)?\b", text_l
            ):
                return name
        # Common informal aliases
        aliases = {
            "stock": "Inventory",
            "inventory": "Inventory",
            "supplies": "Supplies",
            "salary": "Payroll",
            "staff": "Payroll",
            "wages": "Payroll",
            "ads": "Marketing",
            "advertising": "Marketing",
            "rent": "Rent",
            "utility": "Utilities",
            "electricity": "Utilities",
            "insurance": "Insurance",
            "utilities": "Utilities",
        }
        for alias, cat in aliases.items():
            if re.search(rf"\b{alias}\b", text_l):
                return cat
        return None

    # ------------------------------------------------------------------
    # Single-clause resolution
    # ------------------------------------------------------------------
    def resolve_clause(self, text: str) -> ResolvedIntent:
        t = text.strip()
        t_l = t.lower()
        slots: dict = {}

        # --- Greetings / help (only when essentially standalone) ---
        if re.fullmatch(r"\s*(hi|hello|hey|good (morning|afternoon|evening))[!. ]*", t_l):
            return ResolvedIntent(GREETING, text=t)

        # --- SQL command attempts: refused before any other matching ---
        if _SQL_COMMAND.search(t):
            return ResolvedIntent(SECURITY_BLOCK, text=t)
        if re.fullmatch(
            r"\s*(what can you do\??|help( me)?\??|how do you work\??|what are you capable of\??)\s*",
            t_l,
        ):
            return ResolvedIntent(HELP, text=t)

        # --- Card operations (highest priority: security-sensitive) ---
        has_card_word = bool(re.search(r"\bcard", t_l))
        if has_card_word or re.search(CARD_UN_PATTERNS, t_l):
            if re.search(CARD_UN_PATTERNS, t_l):
                return ResolvedIntent(UNBLOCK_CARD, text=t)
            if re.search(CARD_BLOCK_PATTERN, t_l) and (has_card_word or self._extract_card_suffix(t_l)):
                suffix = self._extract_card_suffix(t_l)
                slots = {}
                if suffix:
                    slots["last_four"] = suffix
                return ResolvedIntent(BLOCK_CARD, slots, t)
            if re.search(r"\b(my )?cards?\b.*\b(status|list|show|details)\b|\b(show|list|my) .*\bcards?\b|\bcards?\b\s*\?", t_l):
                return ResolvedIntent(CARDS_LIST, text=t)

        # --- Confirmation / cancellation of pending actions ---
        if re.fullmatch(
            r"\s*(yes|confirm|confirmed|confirm it|yes,? (block|do) it|go ahead|proceed|i confirm)[.! ]*",
            t_l,
        ):
            return ResolvedIntent(CONFIRM, text=t)
        if re.search(r"\bconfirm (?:blocking )?(?:card )?(?:ending )?(?:in )?(\d{4})", t_l):
            return ResolvedIntent(
                CONFIRM, {"last_four": re.search(r"\bconfirm (?:blocking )?(?:card )?(?:ending )?(?:in )?(\d{4})", t_l).group(1)}, t,
            )
        if re.fullmatch(r"\s*(no|cancel|stop|never ?mind|don'?t)[.! ]*", t_l):
            return ResolvedIntent(CANCEL, text=t)

        # --- Anomalies ---
        if re.search(r"\b(unusual|anomal|suspicious|odd|irregular|strange|fraud)\w*", t_l):
            amount = self._extract_amount(t_l)
            if amount:
                slots["min_amount"] = amount
            return ResolvedIntent(ANOMALIES, slots, t)

        # --- Notifications ---
        if re.search(r"\bmark\b.*\bnotifications?\b.*\bread\b|\bmark\b.*\bread\b.*\bnotifications?\b", t_l):
            if re.search(r"\ball\b", t_l):
                slots["all"] = True
            return ResolvedIntent(NOTIFICATIONS_MARK_READ, slots, t)
        if re.search(r"\b(notifications?|alerts?)\b", t_l):
            if re.search(r"\bunread|summar(y|ise|ize)\b", t_l):
                slots["summary"] = True
            return ResolvedIntent(NOTIFICATIONS_LIST, slots, t)

        # --- Help desk ---
        if re.search(r"\b(create|open|new|raise|file|submit|make)\b.*\b(ticket|support ticket|complaint|request)\b", t_l):
            desc = None
            m = re.search(r"\bticket\b\s*(?:about|for|regarding|re:?|on)\s+(.+)$", t_l, re.IGNORECASE)
            if m:
                desc = m.group(1).strip().rstrip("?.!")
            if desc:
                slots["description"] = desc
            return ResolvedIntent(CREATE_TICKET, slots, t)
        if re.search(r"\b(my )?(support )?tickets?\b.*\b(status|list|show|open)\b|\bshow\b.*\btickets?\b", t_l):
            return ResolvedIntent(TICKETS_LIST, text=t)

        # --- Documents / RAG ---
        if re.search(r"\b(documents?|files?|uploads?|invoices?|statements?|contracts?)\b", t_l):
            if re.search(r"\b(list|show|my|uploaded)\b.*\b(documents?|files?|uploads?)\b", t_l) and not re.search(
                r"\b(about|say|tell|what|which|payment terms|find|search for)\b", t_l
            ):
                return ResolvedIntent(DOCUMENTS_LIST, text=t)
            # RAG: any substantive question about document content
            if re.search(r"\b(say|says|tell|what|which|how|search|find|look up|payment terms|due|policy|terms|read)\b", t_l):
                query = re.sub(r"^\s*(please\s+)?(can you\s+)?", "", t, flags=re.IGNORECASE)
                slots["query"] = query
                return ResolvedIntent(DOCUMENT_RAG, slots, t)
            if re.search(r"\bsearch\b", t_l):
                m = re.search(r"\bsearch\b.*?\b(for|about|in)\b\s+(.+)$", t_l)
                slots["query"] = (m.group(2) if m else t).strip()
                return ResolvedIntent(DOCUMENT_SEARCH, slots, t)
            return ResolvedIntent(DOCUMENTS_LIST, text=t)

        # --- Forecasts ---
        if re.search(r"\b(forecast|predict|projection|outlook|expect)\w*\b", t_l) or re.search(
            r"\bhow (much|will)\b.*\b(look|be|spend|cash flow)\b.*\b(next|coming|upcoming)\b", t_l
        ) or re.search(r"\bwill my cash flow look\b", t_l):
            slots["days"] = self._extract_forecast_days(t_l)
            return ResolvedIntent(FORECAST, slots, t)

        # --- Spending ---
        if re.search(r"\bhow much\b.*\bspend\b|\bspending on\b|\bhow much did i (pay|spend)\b", t_l):
            start, end, label = self._resolve_period(t_l)
            category = self._extract_category(t_l)
            slots = {"start": start, "end": end, "label": label}
            if category:
                slots["category"] = category
            return ResolvedIntent(SPENDING_ANALYSIS, slots, t)
        if re.search(r"\b(spending (analysis|breakdown|by category|summary)|category (analysis|breakdown))\b", t_l):
            start, end, label = self._resolve_period(t_l)
            return ResolvedIntent(SPENDING_BY_CATEGORY, {"start": start, "end": end, "label": label}, t)
        if re.search(r"\b(merchant|vendors?|suppliers?|stores?)\b.*\b(analysis|top|breakdown|spending)\b|\bwhere.*money (going|goes)\b", t_l):
            start, end, label = self._resolve_period(t_l)
            return ResolvedIntent(MERCHANT_ANALYSIS, {"start": start, "end": end, "label": label}, t)

        # --- Cash flow ---
        if re.search(r"\b(analyze|analysis of|show|explain|about|my)\b.*\bcash ?flow\b|\bcash ?flow\b.*\b(analys|analyz|explain|summary|review)\b", t_l):
            start, end, label = self._resolve_period(t_l)
            return ResolvedIntent(CASH_FLOW_ANALYSIS, {"start": start, "end": end, "label": label}, t)
        if re.search(r"\b(runway|how long will (my )?(cash|money) last)\b", t_l):
            return ResolvedIntent(CASH_RUNWAY, text=t)

        # --- Transactions ---
        if re.search(r"\b(transactions?|expenses?|payments?|purchases?|deposits?|activity)\b", t_l):
            slots = {}
            limit = self._extract_limit(t_l)
            if limit:
                slots["limit"] = limit
            if re.search(r"\bexpenses?\b", t_l):
                slots["type"] = models.TransactionType.DEBIT
            category = self._extract_category(t_l)
            if category:
                slots["category"] = category
            amount = self._extract_amount(t_l)
            if amount:
                slots["min_amount"] = amount
            return ResolvedIntent(TRANSACTIONS, slots, t)

        # --- Accounts / balance ---
        if re.search(r"\b(balance|how much money|funds available|total balance)\b", t_l):
            return ResolvedIntent(ACCOUNT_BALANCE, text=t)
        if re.search(r"\b(list|show|my) (my )?accounts?\b|\baccounts?\b\s*\?", t_l):
            return ResolvedIntent(ACCOUNTS_LIST, text=t)
        if re.search(r"\baccount\b", t_l):
            return ResolvedIntent(ACCOUNTS_LIST, text=t)

        return ResolvedIntent(UNKNOWN, text=t)

    # ------------------------------------------------------------------
    # Multi-clause resolution
    # ------------------------------------------------------------------
    def resolve(self, message: str) -> List[ResolvedIntent]:
        """Resolve a full user message into an ordered list of intents."""
        message = message.strip()
        if not message:
            return [ResolvedIntent(UNKNOWN, text="")]

        # Split multi-step requests on sentence-like separators.
        clauses = re.split(r"\s*(?:,\s*)?\band also\b|\s*\band then\b|(?<=[.?])\s+|\s*;\s*", message)
        clauses = [c for c in (c.strip() for c in clauses) if c]

        # Try clause splitting on plain " and " only if both sides resolve.
        if len(clauses) == 1:
            parts = re.split(r"\s+and\s+", message, flags=re.IGNORECASE)
            if len(parts) > 1:
                resolved_parts = [self.resolve_clause(p) for p in parts]
                if all(r.intent != UNKNOWN for r in resolved_parts) and len(
                    {r.intent for r in resolved_parts}
                ) > 1:
                    return resolved_parts
            return [self.resolve_clause(message)]

        resolved: List[ResolvedIntent] = []
        for clause in clauses:
            r = self.resolve_clause(clause)
            if r.intent != UNKNOWN:
                resolved.append(r)
        if not resolved:
            return [self.resolve_clause(message)]
        return resolved
