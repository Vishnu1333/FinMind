import pytest
from backend.app.documents.chunking.character_chunker import chunk_document

# Helper to generate a list of page dicts
def make_pages(texts):
    return [{"page_number": i + 1, "text": txt} for i, txt in enumerate(texts)]


def test_short_text():
    pages = make_pages(["short text"])
    chunks = chunk_document(document_id=1, pages=pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "short text"
    assert chunks[0]["source_pages"] == [1]
    assert chunks[0]["chunk_number"] == 0


def test_empty_text():
    pages = []
    chunks = chunk_document(document_id=2, pages=pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == ""
    assert chunks[0]["source_pages"] == []
    assert chunks[0]["chunk_number"] == 0


def test_whitespace_only():
    pages = make_pages(["   "])
    chunks = chunk_document(document_id=3, pages=pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "   "
    assert chunks[0]["source_pages"] == [1]


def test_exact_chunk_size():
    text = "a" * 1000
    pages = make_pages([text])
    chunks = chunk_document(document_id=4, pages=pages)
    assert len(chunks) == 1
    assert len(chunks[0]["text"]) == 1000
    assert chunks[0]["source_pages"] == [1]


def test_multiple_chunks_with_overlap():
    # 1500 chars spread over two pages (800 + 700)
    page1 = "b" * 800
    page2 = "c" * 700
    pages = make_pages([page1, page2])
    chunks = chunk_document(document_id=5, pages=pages)
    # Expect two chunks: first 1000, second remaining 700 (overlap 200)
    assert len(chunks) == 2
    first, second = chunks
    assert len(first["text"]) == 1000
    assert len(second["text"]) == 700
    # Overlap check: last 200 chars of first should equal first 200 of second
    assert first["text"][-200:] == second["text"][:200]
    # Source pages should include both pages for the second chunk (since it crosses boundary)
    assert set(second["source_pages"]).issuperset({1, 2})



def test_deterministic_repeat():
    pages = make_pages(["repeat text"])
    first = chunk_document(document_id=6, pages=pages)
    second = chunk_document(document_id=6, pages=pages)
    assert first == second


def test_chunk_numbering_sequential():
    pages = make_pages(["x" * 2500])
    chunks = chunk_document(document_id=7, pages=pages)
    numbers = [c["chunk_number"] for c in chunks]
    assert numbers == list(range(len(chunks)))


def test_txt_document_behavior():
    # Plain‑text documents are represented as a single page with number 0
    pages = [{"page_number": 0, "text": "plain text content"}]
    chunks = chunk_document(document_id=8, pages=pages)
    assert len(chunks) == 1
    assert chunks[0]["source_pages"] == [0]


def test_malicious_content_preserved():
    malicious = "import os; os.system('rm -rf /')"
    pages = make_pages([malicious])
    chunks = chunk_document(document_id=9, pages=pages)
    assert chunks[0]["text"] == malicious


def test_no_duplicate_chunk_numbers():
    pages = make_pages(["y" * 1200])
    chunks = chunk_document(document_id=10, pages=pages)
    numbers = [c["chunk_number"] for c in chunks]
    assert len(numbers) == len(set(numbers))
