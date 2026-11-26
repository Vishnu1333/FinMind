# Document extraction service
"""Extraction service for supported document formats.

Supported formats:
- PDF (via PyPDF2) – page‑wise extraction
- Plain‑text ``.txt`` – UTF‑8 decoding

The service validates the file extension, enforces a maximum size limit, and
returns a deterministic structured result.
"""

import logging
from typing import List, Dict, Any

from . import extract_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – adjust as needed. Values are deterministic and small.
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MiB
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

def _validate_filename(filename: str) -> str:
    """Return the lower‑cased extension if supported, otherwise raise.

    The filename is NOT exposed in any result – it is used only for validation.
    """
    lower = filename.lower()
    if lower.endswith('.pdf'):
        return 'pdf'
    if lower.endswith('.txt'):
        return 'txt'
    raise ValueError(f"Unsupported file extension for '{filename}'. Supported: .pdf, .txt")

def _validate_size(content: bytes) -> None:
    """Enforce the maximum file size limit.

    Raises ``ValueError`` if the limit is exceeded.
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum allowed ({MAX_FILE_SIZE_BYTES} bytes)")

def extract_document(filename: str, content: bytes) -> Dict[str, Any]:
    """Validate, extract and return a deterministic result dictionary.

    Parameters
    ----------
    filename:
        Original filename supplied by the caller (used only for extension check).
    content:
        Raw bytes of the uploaded document.

    Returns
    -------
    dict
        ``{"text": str, "pages": List[{"page_number": int, "text": str}],
          "format": "pdf" | "txt"}``
    """
    # 1️⃣ Validation – order matters for deterministic error messages.
    _validate_size(content)
    fmt = _validate_filename(filename)

    # 2️⃣ Extraction – delegate to the unified extractor.
    try:
        page_tuples: List[tuple[int, str]] = extract_text(filename, content)
    except Exception as exc:
        # If the underlying extractor raised an unsupported encoding error, propagate it unchanged
        if isinstance(exc, ValueError) and "Unsupported encoding" in str(exc):
            raise
        logger.error("Extraction failed for %s: %s", filename, exc)
        raise ValueError("Failed to extract document content") from exc

    # 3️⃣ Build structured output.
    pages = [{"page_number": pn, "text": txt} for pn, txt in page_tuples]
    # Concatenate page texts preserving order. Use "\n" as a simple separator.
    full_text = "\n".join(txt for _, txt in page_tuples)

    return {
        "text": full_text,
        "pages": pages,
        "format": fmt,
    }

# Export for ``backend.app.documents.extraction`` package.
__all__ = ["extract_document"]
