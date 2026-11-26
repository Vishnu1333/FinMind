from typing import List, Dict, Any

from . import MAX_CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(
    document_id: int,
    pages: List[Dict[str, Any]],
    *,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Deterministically split a document into character chunks.

    Parameters
    ----------
    document_id: int
        Identifier of the source document.
    pages: List[Dict]
        Extraction output – each entry is ``{"page_number": int, "text": str}``.
        The function concatenates page texts in order while preserving the mapping
        from character offsets to originating page numbers.
    max_chunk_size: int, optional
        Upper bound for characters per chunk (default from config).
    chunk_overlap: int, optional
        Number of characters each chunk repeats from the previous chunk (default
        from config).

    Returns
    -------
    List[Dict]
        Each dict contains:
        - ``document_id``
        - ``chunk_number`` (zero‑based)
        - ``text`` (the chunk string)
        - ``source_pages`` (sorted list of page numbers that contributed any
          character to the chunk).
    """
    # Concatenate page texts preserving order.
    full_text = "".join(p["text"] for p in pages)

    # If the text is empty (including whitespace‑only), we still return a single chunk.
    if full_text == "":
        return [{
            "document_id": document_id,
            "chunk_number": 0,
            "text": full_text,
            "source_pages": [p["page_number"] for p in pages] or [],
        }]

    # Build a mapping from each character offset to its source page number.
    offset_to_page: List[int] = []
    for p in pages:
        txt = p["text"]
        offset_to_page.extend([p["page_number"]] * len(txt))

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_idx = 0
    total_len = len(full_text)
    while start < total_len:
        end = min(start + max_chunk_size, total_len)
        chunk_text = full_text[start:end]
        # Determine which pages contributed characters in this slice, including overlap region for accurate provenance.
        overlap_start = max(0, start - chunk_overlap)
        contributing_pages = set(offset_to_page[overlap_start:end])
        # Ensure deterministic ordering.
        source_pages = sorted(contributing_pages)
        chunks.append({
            "document_id": document_id,
            "chunk_number": chunk_idx,
            "text": chunk_text,
            "source_pages": source_pages,
        })
        if end == total_len:
            break
        start = end - chunk_overlap
        chunk_idx += 1
    return chunks
