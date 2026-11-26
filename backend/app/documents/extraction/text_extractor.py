# Simple plain‑text extractor
"""Extract raw text from a UTF‑8 bytes payload.
Returns a list containing a single tuple ``(0, text)`` where ``0`` is the page number placeholder.
"""
from typing import List, Tuple

def extract_text_from_plain(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Unsupported encoding for plain text file") from exc
