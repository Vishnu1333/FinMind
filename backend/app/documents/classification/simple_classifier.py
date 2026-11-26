# Simple deterministic rule‑based document classifier
"""Document classification based on deterministic keyword rules.

The classifier inspects the extracted document text (and optionally the
filename) and returns a predictable result without any external calls.

Output schema:
{
    "document_type": "invoice" | "bank_statement" | "contract" |
                       "financial_report" | "policy" | "other" | "unknown",
    "confidence": float,  # 0.0 – 1.0 deterministic score
    "matched_rules": List[str]  # keywords that contributed to the decision
}
"""

import re
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Classification taxonomy and associated keyword patterns (case‑insensitive).
# ---------------------------------------------------------------------------
_TAXONOMY: Dict[str, List[str]] = {
    "invoice": [
        r"invoice",
        r"invoice number",
        r"bill to",
        r"subtotal",
        r"tax",
        r"total due",
        r"amount due",
    ],
    "bank_statement": [
        r"bank statement",
        r"account number",
        r"opening balance",
        r"closing balance",
        r"transaction date",
        r"debit",
        r"credit",
        r"balance",
    ],
    "contract": [
        r"agreement",
        r"contract",
        r"parties",
        r"terms and conditions",
        r"effective date",
        r"termination",
    ],
    "financial_report": [
        r"income statement",
        r"balance sheet",
        r"cash flow statement",
        r"revenue",
        r"expenses",
        r"assets",
        r"liabilities",
        r"profit",
    ],
    "policy": [
        r"policy",
        r"guidelines",
        r"eligibility",
        r"procedure",
        r"requirements",
        r"terms",
    ],
}

# Generic keywords that indicate a document but not a known taxonomy –
# used to classify as "other" when no specific category matches.
_OTHER_KEYWORDS = [
    r"agenda",
    r"minutes",
    r"summary",
    r"report",
    r"document",
]

# Filename hints – simple lower‑case substring checks that add weight.
_FILENAME_HINTS: Dict[str, List[str]] = {
    "invoice": ["invoice"],
    "bank_statement": ["bank_statement", "bankstatement", "statement"],
    "contract": ["contract"],
    "financial_report": ["financial_report", "report"],
    "policy": ["policy"],
}


def _find_matches(patterns: List[str], text: str) -> List[str]:
    """Return a list of distinct patterns that appear in *text*.

    Matching is case‑insensitive and uses ``re.search`` for each pattern.
    """
    matches: List[str] = []
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            matches.append(pat)
    return matches


def classify_document(text: str, filename: Optional[str] = None) -> Dict:
    """Classify a document deterministically.

    Parameters
    ----------
    text:
        Extracted document text. Must be a string – empty strings are allowed.
    filename:
        Optional original filename (e.g. ``invoice_2023.pdf``). Simple
        substring checks add a small deterministic bias.

    Returns
    -------
    dict
        ``{"document_type": ..., "confidence": ..., "matched_rules": [...]}``
    """
    lowered = text.lower()
    # -------------------------------------------------------------------
    # Gather keyword matches for each taxonomy entry.
    # -------------------------------------------------------------------
    category_matches: Dict[str, List[str]] = {}
    for cat, keywords in _TAXONOMY.items():
        category_matches[cat] = _find_matches(keywords, lowered)

    # Generic "other" matches (used only if no taxonomy matches).
    other_matches = _find_matches(_OTHER_KEYWORDS, lowered)

    # Filename hints – adds a virtual match without altering confidence too much.
    filename_hints: List[str] = []
    if filename:
        fn_lower = filename.lower()
        for cat, hints in _FILENAME_HINTS.items():
            for hint in hints:
                if hint in fn_lower:
                    filename_hints.append(cat)
                    # Record that a filename hint exists for this category (handled later)
                    # Do not add to category_matches to avoid bias in tie-breaking
                    break

    # Determine best category based on the number of matched keyword patterns.
    best_category = None
    best_count = 0
    for cat, matches in category_matches.items():
        cnt = len(matches)
        if cnt > best_count:
            best_count = cnt
            best_category = cat
        # Tie‑breaker: deterministic order defined by taxonomy dict iteration.

    # -------------------------------------------------------------------
    # Resolve final document type.
    # -------------------------------------------------------------------
    if best_count == 0:
        # No taxonomy keyword matched – check for generic other keywords.
        if other_matches:
            document_type = "other"
            matched = other_matches
        else:
            document_type = "unknown"
            matched = []
        confidence = 0.0
    else:
        document_type = best_category  # type: ignore[arg-type]
        # Start with taxonomy matches for the chosen category
        matched = list(category_matches.get(document_type, []))
        # Include filename hint as a matched rule if applicable
        if document_type and document_type in filename_hints:
            matched.append(f"filename:{document_type}")
        # Confidence as proportion of matches for the chosen category over all
        # matches (including generic "other" matches and filename hints). This yields a value in
        # [0, 1] and is fully deterministic.
        total_matches = sum(len(v) for v in category_matches.values()) + len(other_matches) + len(filename_hints)
        confidence = len(matched) / total_matches if total_matches > 0 else 0.0

    return {
        "document_type": document_type,
        "confidence": round(confidence, 3),
        "matched_rules": matched,
    }

# ---------------------------------------------------------------------------
# Export name for ``backend.app.documents.classification`` package.
# ---------------------------------------------------------------------------
__all__ = ["classify_document"]
