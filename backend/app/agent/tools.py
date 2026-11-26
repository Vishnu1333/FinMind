"""Controlled tool registry for the FinMind assistant.

Every tool:
  * is business_id-scoped (authorisation enforced in the query itself),
  * talks to real backend services (FinancialService, ForecastService,
    AnomalyService, RAG, ORM models) – never raw SQL from model output,
    shell execution, Python eval, filesystem access or secrets,
  * returns a strict structured contract consumed by the Gemini agent
    (function responses) and the deterministic fallback (steps):

        success: {"success": true, "source": <service>, "kind": <KIND>,
                  "data": {...}, "text_lines": [...]}
        failure: {"success": false, "error_code": <CODE>, "message": "..."}

There is intentionally NO unblock/unfreeze/reactivate card tool.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from .. import models
from ..services.financial_service import FinancialService
from ..services.forecast_service import get_cached_forecast, InsufficientDataError
from ..services.anomaly_service import AnomalyService
from ..documents.rag.rag_service import answer_question, RAGError


class ToolError(RuntimeError):
    """Raised for user-actionable invalid input (validated contract)."""


def _fmt_myr(value) -> str:
    try:
        return f"RM {float(value):,.2f}"
    except Exception:
        return str(value)


def _parse_date(value, field: str) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ToolError(f"Invalid {field} value: expected an ISO date such as 2026-09-01.")


def _ok(kind: str, source: str, data: dict, text_lines: List[str]) -> dict:
    return {"success": True, "source": source, "kind": kind, "data": data, "text_lines": text_lines}


def _fail(error_code: str, message: str) -> dict:
    return {"success": False, "error_code": error_code, "message": message}


def _tx_line(t) -> str:
    sign = "+" if t.type == models.TransactionType.CREDIT else "-"
    when = t.timestamp.date() if t.timestamp else "—"
    cat = t.category.name if t.category else ""
    label = t.merchant or t.description or "Transaction"
    return f"{when} {sign}{_fmt_myr(t.amount)} — {label}{(f' (in {cat})') if cat else ''}".rstrip()


def _tx_dict(t) -> dict:
    return {
        "transaction_id": t.id,
        "date": t.timestamp.date().isoformat() if t.timestamp else None,
        "type": t.type,
        "amount": str(t.amount),
        "merchant": t.merchant,
        "description": t.description,
        "category": t.category.name if t.category else None,
        "account_id": t.account_id,
    }


def _query_transactions(db: Session, business_id: int, args: dict):
    query = (
        db.query(models.Transaction)
        .join(models.Account)
        .filter(models.Account.business_id == business_id)
    )
    if args.get("type"):
        t = str(args["type"]).lower()
        if t not in ("credit", "debit"):
            raise ToolError("type must be 'credit' or 'debit'.")
        query = query.filter(models.Transaction.type == t)
    if args.get("category"):
        query = query.join(models.Category).filter(models.Category.name.ilike(str(args["category"])))
    if args.get("search"):
        term = f"%{str(args['search'])}%"
        from sqlalchemy import or_
        query = query.filter(or_(
            models.Transaction.merchant.ilike(term),
            models.Transaction.description.ilike(term),
        ))
    if args.get("start"):
        query = query.filter(models.Transaction.timestamp >= _parse_date(args["start"], "start"))
    if args.get("end"):
        query = query.filter(models.Transaction.timestamp < _parse_date(args["end"], "end"))
    if args.get("min_amount") is not None:
        query = query.filter(models.Transaction.amount >= float(args["min_amount"]))
    if args.get("max_amount") is not None:
        query = query.filter(models.Transaction.amount <= float(args["max_amount"]))
    return query.order_by(models.Transaction.timestamp.desc())


# ---------------------------------------------------------------------------
# ACCOUNT tools
# ---------------------------------------------------------------------------
def get_accounts(db: Session, business_id: int, args: dict) -> dict:
    service = FinancialService(db, business_id)
    accounts = service.get_account_balances()
    data = {
        "accounts": [
            {"account_id": a["account_id"], "name": a["name"], "balance": str(a["balance"])}
            for a in accounts
        ]
    }
    lines = [f"{a['name']}: {_fmt_myr(a['balance'])}" for a in accounts] or ["No accounts found."]
    return _ok("DATA", "financial_service", data, lines)


def get_account_balance(db: Session, business_id: int, args: dict) -> dict:
    service = FinancialService(db, business_id)
    accounts = service.get_account_balances()
    account_id = args.get("account_id")
    if account_id is not None:
        for a in accounts:
            if a["account_id"] == int(account_id):
                return _ok("DATA", "financial_service",
                           {"account": a["name"], "account_id": a["account_id"],
                            "balance": str(a["balance"]), "currency": "MYR"},
                           [f"Your balance for {a['name']} is {_fmt_myr(a['balance'])}."])
        return _fail("NOT_FOUND", f"Account {account_id} was not found for your business.")
    total = sum((a["balance"] for a in accounts), Decimal("0"))
    data = {
        "currency": "MYR",
        "total_balance": str(total),
        "accounts": [{"account_id": a["account_id"], "name": a["name"], "balance": str(a["balance"])}
                     for a in accounts],
    }
    lines = [f"Your current total balance is {_fmt_myr(total)} across {len(accounts)} account(s)."]
    for a in accounts:
        lines.append(f"  {a['name']}: {_fmt_myr(a['balance'])}")
    return _ok("DATA", "financial_service", data, lines)


def get_account_details(db: Session, business_id: int, args: dict) -> dict:
    account_id = args.get("account_id")
    service = FinancialService(db, business_id)
    accounts = service.get_account_balances()
    if account_id is None:
        counts = (
            db.query(models.Account.id, models.Transaction.id)
            .join(models.Transaction, models.Transaction.account_id == models.Account.id)
            .filter(models.Account.business_id == business_id)
            .all()
        )
        activity: Dict[int, int] = {}
        for acc_id, _ in counts:
            activity[acc_id] = activity.get(acc_id, 0) + 1
        data = {"accounts": [
            {"account_id": a["account_id"], "name": a["name"], "balance": str(a["balance"]),
             "transaction_count": activity.get(a["account_id"], 0)}
            for a in accounts
        ]}
        lines = [
            f"{a['name']} (#{a['account_id']}): balance {_fmt_myr(a['balance'])}, "
            f"{activity.get(a['account_id'], 0)} transactions"
            for a in accounts
        ] or ["No accounts found."]
        return _ok("DATA", "financial_service", data, lines)
    for a in accounts:
        if a["account_id"] == int(account_id):
            recent = (
                db.query(models.Transaction)
                .join(models.Account)
                .filter(
                    models.Account.business_id == business_id,
                    models.Transaction.account_id == int(account_id),
                )
                .order_by(models.Transaction.timestamp.desc())
                .limit(5)
                .all()
            )
            data = {"account": a["name"], "account_id": a["account_id"],
                    "balance": str(a["balance"]),
                    "recent_transactions": [_tx_dict(t) for t in recent]}
            lines = [f"{a['name']} (#{a['account_id']}): balance {_fmt_myr(a['balance'])}"]
            if recent:
                lines.append("Recent activity:")
                lines.extend(f"  {_tx_line(t)}" for t in recent)
            return _ok("DATA", "financial_service", data, lines)
    return _fail("NOT_FOUND", f"Account {account_id} was not found for your business.")


# ---------------------------------------------------------------------------
# TRANSACTION tools
# ---------------------------------------------------------------------------
def get_transactions(db: Session, business_id: int, args: dict) -> dict:
    limit = int(args.get("limit") or 10)
    limit = max(1, min(limit, 50))
    query = _query_transactions(db, business_id, args)
    txs = query.limit(limit).all()
    total = query.count()
    data = {"count": len(txs), "total_matching": total,
            "transactions": [_tx_dict(t) for t in txs]}
    lines = [_tx_line(t) for t in txs] or ["No matching transactions found."]
    return _ok("DATA", "financial_service", data, lines)


def search_transactions(db: Session, business_id: int, args: dict) -> dict:
    if not any(args.get(k) for k in ("search", "category", "type", "start", "end",
                                     "min_amount", "max_amount")):
        args = dict(args)
        args.setdefault("limit", 10)
    return get_transactions(db, business_id, args)


# ---------------------------------------------------------------------------
# ANALYTICS tools
# ---------------------------------------------------------------------------
def analyze_spending(db: Session, business_id: int, args: dict) -> dict:
    service = FinancialService(db, business_id)
    start, end = _parse_date(args.get("start"), "start"), _parse_date(args.get("end"), "end")
    category = args.get("category")
    breakdown = service.spending_by_category(start, end)
    label = args.get("label") or "the selected period"
    if category:
        name = str(category).strip()
        row = next((b for b in breakdown if b["category_name"].lower() == name.lower()), None)
        if row is None:
            names = ", ".join(b["category_name"] for b in breakdown) or "none exist yet"
            return _fail("NO_DATA", f"No spending recorded for category '{category}' in {label}. "
                                    f"Categories with spending: {names}.")
        data = {"category": row["category_name"], "total": str(row["amount"]),
                "percentage_of_spending": str(row["percentage"]),
                "transaction_count": None, "period": label}
        return _ok("CALCULATION", "financial_service", data, [
            f"Total {row['category_name']} spending for {label}: {_fmt_myr(row['amount'])} "
            f"({float(row['percentage']):.1f}% of all spending)."
        ])
    expenses = service.get_expenses(start, end)
    top = max(breakdown, key=lambda b: b["amount"]) if breakdown else None
    data = {
        "total_spending": str(expenses),
        "period": label,
        "by_category": [
            {"category": b["category_name"], "amount": str(b["amount"]),
             "percentage": str(b["percentage"])} for b in breakdown
        ],
        "top_category": top["category_name"] if top else None,
    }
    lines = [f"Total expenses for {label}: {_fmt_myr(expenses)}."]
    if top:
        lines.append(f"Largest category: {top['category_name']} at {_fmt_myr(top['amount'])} "
                     f"({float(top['percentage']):.1f}% of spending).")
    return _ok("CALCULATION", "financial_service", data, lines)


def analyze_spending_by_category(db: Session, business_id: int, args: dict) -> dict:
    service = FinancialService(db, business_id)
    start, end = _parse_date(args.get("start"), "start"), _parse_date(args.get("end"), "end")
    breakdown = service.spending_by_category(start, end)
    label = args.get("label") or "the selected period"
    data = {"breakdown": [
        {"category": b["category_name"], "amount": str(b["amount"]),
         "percentage": str(b["percentage"])} for b in breakdown
    ]}
    lines = [f"Spending by category for {label}:"] if breakdown else [f"No expense data for {label}."]
    lines.extend(f"{b['category_name']}: {_fmt_myr(b['amount'])} ({float(b['percentage']):.1f}%)"
                 for b in breakdown)
    return _ok("CALCULATION", "financial_service", data, lines)


def analyze_merchants(db: Session, business_id: int, args: dict) -> dict:
    service = FinancialService(db, business_id)
    rows = service.top_merchants(limit=10, start=_parse_date(args.get("start"), "start"),
                                  end=_parse_date(args.get("end"), "end"))
    label = args.get("label") or "the selected period"
    data = {"merchants": [{"merchant": r["merchant"], "total": str(r["total"]),
                           "count": r["count"]} for r in rows]}
    lines = [f"Top merchants by spend for {label}:"] if rows else [f"No merchant spending data for {label}."]
    lines.extend(f"{r['merchant']}: {_fmt_myr(r['total'])} over {r['count']} payments" for r in rows)
    return _ok("CALCULATION", "financial_service", data, lines)


def analyze_cash_flow(db: Session, business_id: int, args: dict) -> dict:
    service = FinancialService(db, business_id)
    start, end = _parse_date(args.get("start"), "start"), _parse_date(args.get("end"), "end")
    start_dt, end_dt = service._resolve_date_range(start, end)
    income = service.get_income(start_dt, end_dt)
    expenses = service.get_expenses(start_dt, end_dt)
    net = service.get_net_cash_flow(start_dt, end_dt)
    label = args.get("label") or f"{start_dt:%d %b %Y} to {end_dt:%d %b %Y}"
    data = {"income": str(income), "expenses": str(expenses), "net": str(net),
            "period": label, "currency": "MYR"}
    return _ok("CALCULATION", "financial_service", data, [
        f"Income for {label}: {_fmt_myr(income)}",
        f"Expenses for {label}: {_fmt_myr(expenses)}",
        f"Net cash flow: {_fmt_myr(net)}",
    ])


def get_cash_runway(db: Session, business_id: int, args: dict) -> dict:
    service = FinancialService(db, business_id)
    result = service.cash_runway()
    if result.get("insufficient_data"):
        line = result.get("message") or "Not enough expense history to compute runway."
        return _ok("STATUS", "financial_service", {"runway": result}, [line])
    months = float(result["months"])
    data = {"runway_months": result["months"], "currency": "MYR"}
    return _ok("CALCULATION", "financial_service", data, [
        f"Cash runway: approximately {months:.1f} months at the current burn rate."
    ])


# ---------------------------------------------------------------------------
# PREDICTION tools
# ---------------------------------------------------------------------------
def get_forecast(db: Session, business_id: int, args: dict) -> dict:
    days = int(args.get("days") or 30)
    days = max(1, min(days, 90))
    try:
        result = get_cached_forecast(db, business_id, days)
    except InsufficientDataError as exc:
        return _fail("NO_DATA", str(exc))
    model = result["model"]
    total = sum(p["predicted_cashflow"] for p in result["forecast"])
    data = {
        "days": days,
        "total_projected": round(total, 2),
        "average_daily_net": round(total / max(days, 1), 2),
        "series": result["forecast"],
        "model": {
            "selected": model["selected"],
            "artifact": model.get("artifact"),
            "trained_on_days": model["trained_on_days"],
            "business_context_days": model.get("business_context_days"),
            "holdout_mae_scaled": model["holdout_mae_scaled"],
            "candidates": model.get("candidates", {}),
        },
    }
    lines = [
        f"{days}-day cash-flow forecast (model prediction):",
        f"  Projected net cash flow over the horizon: {_fmt_myr(total)}.",
        f"  Average predicted daily net: {_fmt_myr(total / max(days, 1))}.",
        f"  Model: {model['selected']} (registry artifact {model.get('artifact') or 'n/a'}), "
        f"selected by offline holdout evaluation; trained on {model['trained_on_days']} days of "
        f"daily financial series, applied to {model.get('business_context_days', 'the')} days of "
        f"this business's own history.",
    ]
    return _ok("PREDICTION", "forecast_service", data, lines)


# ---------------------------------------------------------------------------
# ANOMALY tools
# ---------------------------------------------------------------------------
def _detect(db: Session, business_id: int, args: dict) -> dict:
    service = AnomalyService(db, business_id)
    min_amount = args.get("min_amount")
    return service.detect(top_n=int(args.get("limit") or 10),
                          min_amount=float(min_amount) if min_amount is not None else None)


def get_anomalies(db: Session, business_id: int, args: dict) -> dict:
    result = _detect(db, business_id, args)
    if not result.get("available"):
        return _fail("NO_DATA", result.get("reason") or "Anomaly detection is not available yet.")
    flagged = result["anomalies"]
    data = {"count": len(flagged), "model": result.get("model"), "anomalies": flagged}
    if not flagged:
        lines = ["No unusual transactions were detected in the current analysis."]
    else:
        lines = [f"{len(flagged)} unusual transactions flagged (most anomalous first):"]
        for a in flagged:
            ev = a.get("evidence", {})
            ztxt = ""
            if ev.get("amount_z_score") is not None:
                ztxt = f", {ev['amount_z_score']:.1f} std devs above the average spend of {_fmt_myr(ev['business_avg_spend'])}"
            lines.append(f"  {a['date'][:10] if a['date'] else '—'} — {_fmt_myr(a['amount'])} to "
                         f"{a['merchant']} (category: {a['category'] or 'uncategorised'}){ztxt}")
    return _ok("DATA", "anomaly_service", data, lines)


def get_anomaly_details(db: Session, business_id: int, args: dict) -> dict:
    tx_id = args.get("transaction_id")
    if tx_id is None:
        raise ToolError("Provide the transaction_id of the flagged transaction.")
    result = _detect(db, business_id, {"limit": 20})
    if not result.get("available"):
        return _fail("NO_DATA", result.get("reason") or "Anomaly detection is not available yet.")
    match = next((a for a in result["anomalies"] if a.get("transaction_id") == int(tx_id)), None)
    if match is None:
        return _fail("NOT_FOUND",
                     f"Transaction {tx_id} is not currently flagged as anomalous for your business.")
    tx = (
        db.query(models.Transaction)
        .join(models.Account)
        .filter(models.Account.business_id == business_id,
                models.Transaction.id == int(tx_id))
        .first()
    )
    data = dict(match)
    if tx is not None:
        data["transaction"] = _tx_dict(tx)
    lines = [
        f"Transaction {tx_id}: {_fmt_myr(match['amount'])} to {match['merchant']} on "
        f"{match['date'][:10] if match['date'] else 'an unknown date'}.",
        "Reasons: " + "; ".join(match.get("reasons", []) or ["flagged by the anomaly model"]),
    ]
    return _ok("DATA", "anomaly_service", data, lines)


# ---------------------------------------------------------------------------
# DOCUMENT / RAG tools
# ---------------------------------------------------------------------------
def list_documents(db: Session, business_id: int, args: dict) -> dict:
    docs = db.query(models.Document).filter(models.Document.business_id == business_id).all()
    data = {"count": len(docs), "documents": [
        {"document_id": d.id, "filename": d.filename,
         "document_type": d.document_type, "status": d.status}
        for d in docs
    ]}
    if not docs:
        lines = ["No documents have been uploaded yet. You can upload PDF or TXT files on the Documents page."]
    else:
        lines = [f"You have {len(docs)} document(s):"]
        lines.extend(f"  {d.filename} — {d.document_type} ({d.status})" for d in docs)
    return _ok("DATA", "document_service", data, lines)


def search_knowledge(db: Session, business_id: int, args: dict) -> dict:
    query = args.get("query") or args.get("search") or ""
    if not query:
        raise ToolError("No question was provided to search the documents.")
    try:
        result = answer_question(question=str(query), business_id=business_id, top_k=5)
    except RAGError as exc:
        return _fail("TOOL_FAILURE", f"Document search failed: {exc}")
    sources = result.get("sources") or []
    source_meta = [
        {"filename": s.get("filename"), "document_id": s.get("document_id"),
         "source_pages": s.get("source_pages")}
        for s in sources[:5]
    ]
    if not result.get("grounded"):
        return _ok("STATUS", "rag_service",
                   {"grounded": False, "answer": result.get("answer") or "", "sources": source_meta},
                   [result.get("answer") or
                    "I couldn't find sufficient information in your uploaded documents to answer that."])
    data = {"grounded": True, "answer": result["answer"], "sources": source_meta}
    lines = [result["answer"], "", "Evidence:"]
    for s in sources[:5]:
        fname = s.get("filename") or f"document #{s.get('document_id')}"
        pages = s.get("source_pages")
        page_txt = f", page {pages[0]}" if isinstance(pages, list) and pages else ""
        lines.append(f"  — {fname}{page_txt}")
    return _ok("EVIDENCE", "rag_service", data, lines)


# ---------------------------------------------------------------------------
# CARD tools (block only – no unblock exists by design)
# ---------------------------------------------------------------------------
def list_cards(db: Session, business_id: int, args: dict) -> dict:
    cards = db.query(models.Card).filter(models.Card.business_id == business_id).all()
    data = {"count": len(cards), "cards": [
        {"card_id": c.id, "last_four": c.last_four, "status": c.status} for c in cards
    ]}
    if not cards:
        return _ok("DATA", "cards_service", data, ["Your business has no cards issued."])
    lines = [f"Your business has {len(cards)} card(s):"]
    lines.extend(f"  •••• {c.last_four} — {c.status}" for c in cards)
    return _ok("DATA", "cards_service", data, lines)


def get_card_status(db: Session, business_id: int, args: dict) -> dict:
    last_four = str(args.get("last_four") or "")
    query = db.query(models.Card).filter(models.Card.business_id == business_id)
    if last_four:
        query = query.filter(models.Card.last_four == last_four)
    cards = query.all()
    if not cards:
        return _fail("NOT_FOUND",
                     f"No card{' ending in ' + last_four if last_four else ''} was found for your business.")
    data = {"cards": [{"card_id": c.id, "last_four": c.last_four, "status": c.status} for c in cards]}
    lines = [f"Card •••• {c.last_four} is {c.status}." for c in cards]
    return _ok("DATA", "cards_service", data, lines)


def block_card(db: Session, business_id: int, args: dict) -> dict:
    """Executes a card block. Requires an explicit, user-scoped confirmation
    that the policy layer registered before this call. There is no unblock
    counterpart, by design."""
    from .orchestrator import pop_pending_block
    last_four = str(args.get("last_four") or "")
    if not last_four or len(last_four) != 4 or not last_four.isdigit():
        raise ToolError("Please provide the last four digits of the card.")
    card = (
        db.query(models.Card)
        .filter(models.Card.business_id == business_id, models.Card.last_four == last_four)
        .first()
    )
    if not card:
        return _fail("NOT_FOUND", f"No card ending in {last_four} was found for your business.")
    if card.status == "blocked":
        return _ok("STATUS", "cards_service", {"card_id": card.id, "last_four": last_four,
                                               "status": "blocked"},
                   [f"Card •••• {last_four} is already blocked."])
    pending = pop_pending_block(card.id)
    if not pending:
        return _ok("STATUS", "cards_service",
                   {"confirmation_required": True, "last_four": last_four, "status": card.status},
                   [f"Blocking card •••• {last_four} needs your explicit confirmation first. "
                    f"Please confirm to proceed — blocking is immediate and cannot be undone by the assistant."])
    card.status = "blocked"
    db.commit()
    return _ok("STATUS", "cards_service",
               {"card_id": card.id, "last_four": last_four, "status": "blocked"},
               [f"Card •••• {last_four} has been blocked. It can no longer be used for new transactions."])


# ---------------------------------------------------------------------------
# NOTIFICATION tools
# ---------------------------------------------------------------------------
def get_notifications(db: Session, business_id: int, args: dict) -> dict:
    limit = int(args.get("limit") or 10)
    limit = max(1, min(limit, 50))
    notifications = (
        db.query(models.Notification)
        .filter(models.Notification.business_id == business_id)
        .order_by(models.Notification.created_at.desc())
        .limit(limit)
        .all()
    )
    unread = sum(1 for n in notifications if not n.read)
    data = {"count": len(notifications), "unread": unread, "notifications": [
        {"id": n.id, "title": n.title, "severity": n.severity, "read": bool(n.read),
         "message": n.message} for n in notifications
    ]}
    if not notifications:
        lines = ["You have no notifications."]
    else:
        lines = [f"Your notifications ({unread} unread of {len(notifications)} shown):"]
        lines.extend(f"  [{n.severity}] {n.title} — {n.message}" for n in notifications)
    return _ok("DATA", "notifications_service", data, lines)


def mark_notification_read(db: Session, business_id: int, args: dict) -> dict:
    notification_id = args.get("notification_id")
    if notification_id is None and not args.get("all"):
        raise ToolError("Provide notification_id, or pass all=true to mark everything read.")
    query = db.query(models.Notification).filter(models.Notification.business_id == business_id)
    if notification_id is not None:
        query = query.filter(models.Notification.id == int(notification_id))
    updated = query.update({"read": 1})
    db.commit()
    if updated == 0:
        return _fail("NOT_FOUND", "No matching unread notification was found.")
    what = "All notifications" if notification_id is None else f"Notification {notification_id}"
    return _ok("STATUS", "notifications_service", {"updated": updated},
               [f"{what} marked as read."])


# ---------------------------------------------------------------------------
# HELP DESK tools
# ---------------------------------------------------------------------------
def list_helpdesk_tickets(db: Session, business_id: int, args: dict) -> dict:
    limit = int(args.get("limit") or 10)
    tickets = (
        db.query(models.Ticket)
        .filter(models.Ticket.business_id == business_id)
        .order_by(models.Ticket.created_at.desc())
        .limit(max(1, min(limit, 50)))
        .all()
    )
    data = {"count": len(tickets), "tickets": [
        {"ticket_id": t.id, "title": t.title, "status": t.status,
         "priority": t.priority, "category": t.category} for t in tickets
    ]}
    if not tickets:
        lines = ["You have no support tickets. You can create one here or on the Help Desk page."]
    else:
        lines = [f"You have {len(tickets)} ticket(s):"]
        lines.extend(f"  #{t.id} [{t.status}/{t.priority}] {t.title}" for t in tickets)
    return _ok("DATA", "helpdesk_service", data, lines)


def create_helpdesk_ticket(db: Session, business_id: int, args: dict) -> dict:
    description = (args.get("description") or args.get("text") or "").strip()
    if not description:
        raise ToolError("Describe the problem so the support team can help.")
    title = (args.get("title") or description[:60] + ("…" if len(description) > 60 else "")) \
        or "Support request"
    ticket = models.Ticket(
        business_id=business_id,
        title=title,
        description=description,
        status="open",
        priority=str(args.get("priority") or "normal"),
        category=str(args.get("category") or "general"),
    )
    db.add(ticket)
    db.commit()
    data = {"ticket_id": ticket.id, "title": ticket.title, "status": ticket.status}
    return _ok("STATUS", "helpdesk_service", data, [
        f"Support ticket #{ticket.id} created: “{ticket.title}”. Our team will follow up."
    ])


def get_helpdesk_ticket(db: Session, business_id: int, args: dict) -> dict:
    ticket_id = args.get("ticket_id")
    if ticket_id is None:
        raise ToolError("Provide the ticket_id.")
    ticket = (
        db.query(models.Ticket)
        .filter(models.Ticket.business_id == business_id, models.Ticket.id == int(ticket_id))
        .first()
    )
    if not ticket:
        return _fail("NOT_FOUND", f"Ticket {ticket_id} was not found for your business.")
    data = {"ticket_id": ticket.id, "title": ticket.title, "description": ticket.description,
            "status": ticket.status, "priority": ticket.priority, "category": ticket.category,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None}
    lines = [
        f"Ticket #{ticket.id} — {ticket.title}",
        f"Status: {ticket.status} · Priority: {ticket.priority} · Category: {ticket.category}",
        f"Description: {ticket.description}",
    ]
    return _ok("DATA", "helpdesk_service", data, lines)


# ---------------------------------------------------------------------------
# Registry: name -> callable(db, business_id, args) with a Gemini schema.
# There is intentionally NO unblock/unfreeze/unlock/reactivate tool here.
# ---------------------------------------------------------------------------
TOOL_REGISTRY: Dict[str, Callable[[Session, int, dict], dict]] = {
    "get_accounts": get_accounts,
    "get_account_balance": get_account_balance,
    "get_account_details": get_account_details,
    "get_transactions": get_transactions,
    "search_transactions": search_transactions,
    "analyze_spending": analyze_spending,
    "analyze_spending_by_category": analyze_spending_by_category,
    "analyze_merchants": analyze_merchants,
    "analyze_cash_flow": analyze_cash_flow,
    "get_cash_runway": get_cash_runway,
    "get_forecast": get_forecast,
    "get_anomalies": get_anomalies,
    "get_anomaly_details": get_anomaly_details,
    "list_documents": list_documents,
    "search_knowledge": search_knowledge,
    "list_cards": list_cards,
    "get_card_status": get_card_status,
    "block_card": block_card,
    "get_notifications": get_notifications,
    "mark_notification_read": mark_notification_read,
    "list_helpdesk_tickets": list_helpdesk_tickets,
    "create_helpdesk_ticket": create_helpdesk_ticket,
    "get_helpdesk_ticket": get_helpdesk_ticket,
}


def _schema(properties: dict, required: list = None) -> dict:
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


TOOL_DECLARATIONS: List[dict] = [
    {
        "name": "get_accounts",
        "description": "List all bank accounts of the business with their current balances.",
        "parameters": _schema({}),
    },
    {
        "name": "get_account_balance",
        "description": "Get the current balance — the business total, or one account when account_id is given.",
        "parameters": _schema({
            "account_id": {"type": "integer", "description": "Optional account ID to get a single account balance"},
        }),
    },
    {
        "name": "get_account_details",
        "description": "Get details for one account (balance, recent activity), or an overview of all accounts when account_id is omitted.",
        "parameters": _schema({
            "account_id": {"type": "integer", "description": "Optional account ID"},
        }),
    },
    {
        "name": "get_transactions",
        "description": "Get the most recent transactions, with optional filters (type, category, date range, amounts).",
        "parameters": _schema({
            "limit": {"type": "integer", "description": "How many transactions to return (1-50, default 10)"},
            "type": {"type": "string", "enum": ["credit", "debit"], "description": "credit = money in, debit = money out"},
            "category": {"type": "string", "description": "Category name, e.g. Inventory, Rent, Payroll"},
            "start": {"type": "string", "description": "ISO date/datetime lower bound, e.g. 2026-09-01"},
            "end": {"type": "string", "description": "ISO date/datetime exclusive upper bound"},
            "min_amount": {"type": "number", "description": "Minimum transaction amount"},
            "max_amount": {"type": "number", "description": "Maximum transaction amount"},
        }),
    },
    {
        "name": "search_transactions",
        "description": "Search transactions by merchant/description text, category, type, date range or amount range.",
        "parameters": _schema({
            "search": {"type": "string", "description": "Free-text term to match merchant or description"},
            "category": {"type": "string"},
            "type": {"type": "string", "enum": ["credit", "debit"]},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "min_amount": {"type": "number"},
            "max_amount": {"type": "number"},
            "limit": {"type": "integer"},
        }),
    },
    {
        "name": "analyze_spending",
        "description": "Analyze expenses: total spending for a period, optionally focused on one category. Use for questions like 'How much did I spend on inventory this month?'",
        "parameters": _schema({
            "category": {"type": "string", "description": "Optional category to focus on, e.g. Inventory"},
            "start": {"type": "string", "description": "ISO date/datetime of period start"},
            "end": {"type": "string", "description": "ISO date/datetime exclusive period end"},
            "label": {"type": "string", "description": "Optional human label of the period, e.g. 'this month'"},
        }),
    },
    {
        "name": "analyze_spending_by_category",
        "description": "Full spending breakdown by category for a period, with amounts and percentages.",
        "parameters": _schema({
            "start": {"type": "string"},
            "end": {"type": "string"},
            "label": {"type": "string"},
        }),
    },
    {
        "name": "analyze_merchants",
        "description": "Top merchants by total spend for a period. Use for 'Where is my money going?'",
        "parameters": _schema({
            "start": {"type": "string"},
            "end": {"type": "string"},
            "label": {"type": "string"},
        }),
    },
    {
        "name": "analyze_cash_flow",
        "description": "Cash-flow analysis for a period: income, expenses and net. Use for comparing months or explaining expense changes.",
        "parameters": _schema({
            "start": {"type": "string"},
            "end": {"type": "string"},
            "label": {"type": "string"},
        }),
    },
    {
        "name": "get_cash_runway",
        "description": "How many months the current cash balance covers at the average burn rate.",
        "parameters": _schema({}),
    },
    {
        "name": "get_forecast",
        "description": "Machine-learning cash-flow forecast for the next days (1-90). Values are model predictions from the forecasting service.",
        "parameters": _schema({
            "days": {"type": "integer", "description": "Forecast horizon in days (default 30)"},
        }),
    },
    {
        "name": "get_anomalies",
        "description": "Unusual transactions flagged by the anomaly model, with statistical evidence. Use for 'Which transactions look unusual?'",
        "parameters": _schema({
            "limit": {"type": "integer", "description": "Maximum anomalies to return (default 10)"},
            "min_amount": {"type": "number", "description": "Optional minimum amount filter"},
        }),
    },
    {
        "name": "get_anomaly_details",
        "description": "Detailed evidence for one specific flagged transaction.",
        "parameters": _schema({
            "transaction_id": {"type": "integer", "description": "The flagged transaction ID"},
        }, ["transaction_id"]),
    },
    {
        "name": "list_documents",
        "description": "List the documents the business has uploaded (invoices, statements, etc.).",
        "parameters": _schema({}),
    },
    {
        "name": "search_knowledge",
        "description": "Search the business's uploaded documents for evidence and answer from them (RAG). Use for questions about document contents such as payment terms.",
        "parameters": _schema({
            "query": {"type": "string", "description": "The question to answer from the documents"},
        }, ["query"]),
    },
    {
        "name": "get_notifications",
        "description": "Get the business's recent notifications.",
        "parameters": _schema({
            "limit": {"type": "integer"},
        }),
    },
    {
        "name": "mark_notification_read",
        "description": "Mark one notification (or all) as read.",
        "parameters": _schema({
            "notification_id": {"type": "integer", "description": "Notification ID to mark read"},
            "all": {"type": "boolean", "description": "Pass true to mark every notification read"},
        }),
    },
    {
        "name": "list_cards",
        "description": "List the business's cards with their status.",
        "parameters": _schema({}),
    },
    {
        "name": "get_card_status",
        "description": "Get the status of one card by its last four digits.",
        "parameters": _schema({
            "last_four": {"type": "string", "description": "Last four digits of the card"},
        }, ["last_four"]),
    },
    {
        "name": "block_card",
        "description": "Block a card by its last four digits. Requires an explicit user confirmation — if not yet confirmed, reply to the user asking them to confirm and report that confirmation is required. Blocking cannot be undone by the assistant.",
        "parameters": _schema({
            "last_four": {"type": "string", "description": "Last four digits of the card to block"},
        }, ["last_four"]),
    },
    {
        "name": "list_helpdesk_tickets",
        "description": "List the business's support tickets.",
        "parameters": _schema({
            "limit": {"type": "integer"},
        }),
    },
    {
        "name": "create_helpdesk_ticket",
        "description": "Create a support ticket for the business, e.g. to dispute a transaction.",
        "parameters": _schema({
            "title": {"type": "string", "description": "Short ticket title"},
            "description": {"type": "string", "description": "What the ticket is about"},
        }, ["description"]),
    },
    {
        "name": "get_helpdesk_ticket",
        "description": "Get one support ticket by ID.",
        "parameters": _schema({
            "ticket_id": {"type": "integer"},
        }, ["ticket_id"]),
    },
]
