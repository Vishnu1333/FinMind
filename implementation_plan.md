# Implementation Plan for BLOCK 2 – Retrieval & Evidence Validation

## Goal Description

Implement the remaining retrieval components and full integration tests for the FinMind application, ensuring all functionality works against a real PostgreSQL database with pgvector support. This includes:
1. Hybrid retriever (vector + lexical merging, scoring, deduplication, deterministic ranking).
2. Retrieval service orchestrating embedding generation, vector and lexical retrieval, hybrid ranking, and evidence validation.
3. Evidence validator (detailed checks of chunk/document consistency and metadata).
4. Comprehensive PostgreSQL‑backed integration tests for vector, lexical, hybrid retrieval, business isolation, filters, and evidence validation.
5. Verify pgvector index existence.
6. Run targeted and full test suites to confirm no regressions.

All code must remain Python 3.9‑compatible, use the existing ORM models, and not modify any already‑verified vector/lexical modules unless a genuine defect is discovered.

---

## User Review Required

> [!IMPORTANT]
> **Business‑level isolation**: The retrieval service must enforce `business_id` filtering **in SQL**, not in post‑processing. Ensure the generated queries include the filter.
>
> **Insufficient evidence handling**: When no chunk meets the `RELEVANCE_THRESHOLD`, the service should return an explicit `insufficient_evidence` flag instead of an empty list.
>
> **Test data isolation**: The integration tests will create temporary data in the real PostgreSQL database. They must be wrapped in a transaction/fixture that rolls back after each test to avoid polluting production data.
>
> **pgvector extension**: The plan assumes the PostgreSQL server has the `vector` extension installed. If not, the tests will fail; you may need to run `CREATE EXTENSION IF NOT EXISTS vector;` manually.

## Open Questions

> [!WARNING]
> 1. **Embedding provider mocking** – Do you prefer using the real OpenAI provider (network calls) with a mock key, or should we patch the provider to return deterministic deterministic vectors for tests?
>
> 2. **Top‑K default** – What default value should the retrieval service use for `top_k` when the caller does not specify it?
>
> 3. **Weight configuration** – The existing `backend/app/documents/retrieval/config.py` defines `VECTOR_WEIGHT`, `LEXICAL_WEIGHT`, and `RELEVANCE_THRESHOLD`. Should we keep those defaults or adjust them for the test environment?
>
> 4. **Test data vectors** – Shall we generate deterministic random vectors (e.g., using `np.random.RandomState(0)`) or use fixed static vectors for reproducibility?

## Proposed Changes

---
### Hybrid Retriever

**File:** `backend/app/documents/retrieval/hybrid_retriever.py`  
**[NEW]** Implements `HybridRetriever` with method:
```python
def retrieve(
    self,
    query: str,
    vector_candidates: List[DocumentChunk],
    lexical_candidates: List[DocumentChunk],
    top_k: int,
) -> List[HybridResult]:
```
- Merges candidates, deduplicates by `DocumentChunk.id`.
- Computes `hybrid_score = VECTOR_WEIGHT * vector_score + LEXICAL_WEIGHT * lexical_score`.
- Applies `RELEVENCE_THRESHOLD` – only returns chunks where `hybrid_score` >= threshold.
- Deterministic ordering: sort by `hybrid_score` (desc), then by `DocumentChunk.id` (asc).
- Returns objects containing original chunk plus `vector_score`, `lexical_score`, `hybrid_score`.

---
### Retrieval Service

**File:** `backend/app/documents/retrieval/retrieval_service.py`  
**[NEW]** Implements `RetrievalService` with method:
```python
def retrieve(
    self,
    query: str,
    business_id: int,
    *,
    document_id: Optional[int] = None,
    document_type: Optional[str] = None,
    source: Optional[str] = None,
    top_k: int = 10,
) -> RetrievalResult:
```
- Validates `business_id` presence.
- Uses `EmbeddingProvider` to obtain query embedding (mocked in tests).
- Calls existing `VectorRetriever` and `LexicalRetriever` with same filters.
- Passes candidates to `HybridRetriever`.
- Runs `EvidenceValidator.validate` on final chunks; if validation fails, raises `EvidenceValidatorError`.
- Returns a dataclass `RetrievalResult` containing `chunks: List[DocumentChunk]` and `insufficient_evidence: bool`.
- All SQL queries include `WHERE document.business_id = :business_id` and optional filter clauses.

---
### Evidence Validator (enhanced)

**File:** `backend/app/documents/retrieval/evidence_validator.py`  
**[MODIFY]** Existing stub (if any) will be replaced with full implementation:
- Checks existence of chunk and document via ORM session.
- Verifies `chunk.document_id == document.id`.
- Verifies `document.business_id == requested_business_id`.
- Ensures `chunk.text` is non‑empty.
- Ensures `metadata_json` is a dict and, if `source_pages` present, it is a list of positive ints.
- Raises `EvidenceValidatorError` with clear messages on failure.

---
### Tests

**Files:**
- `backend/tests/test_retrieval.py`  **[NEW]** – integration tests covering vector, lexical, hybrid retrieval, business isolation, and filter parameters. Uses fixtures to insert deterministic test data (businesses, documents, chunks) and rolls back after each test.
- `backend/tests/test_evidence_validator.py` **[NEW]** – unit tests for each validation rule, including malformed metadata and business mismatch cases.

**Test Data Generation:**
- Helper functions in the test module create two businesses, three documents per business, and multiple chunks.
- Vectors are generated with a deterministic NumPy RNG (seed 42) to ensure repeatable cosine distances.
- Text fields contain distinguishable keywords to trigger lexical search.
- Fixtures use SQLAlchemy session scoped to a transaction that is rolled back (`session.begin_nested()`).

---
### Index Verification

**File:** `backend/tests/test_index_verification.py` **[NEW]** – runs a raw SQL query to assert that `ix_document_chunks_embedding` exists and uses `vector_cosine_ops`.

---
### CI / Test Runner Adjustments

No changes to existing CI scripts are required. After implementation, we will run:
```bash
./backend/venv/bin/pytest -q backend/tests/test_retrieval.py backend/tests/test_evidence_validator.py
./backend/venv/bin/pytest -q
```
and report the exact summary.

---
## Verification Plan

1. **Static analysis** – lint the new modules for syntax errors.
2. **Unit tests** – run the newly added test files; they should pass on a PostgreSQL instance with pgvector installed.
3. **Targeted test run** – capture the pytest output for the two new test files.
4. **Full regression** – run the entire suite and ensure no failures.
5. **Index check** – ensure the pgvector index exists.
6. **Manual sanity check** – optionally use a Python REPL to call `RetrievalService.retrieve` and verify output structure.

---

*Please review the plan, answer the open questions, and approve so we can proceed with implementation.*
