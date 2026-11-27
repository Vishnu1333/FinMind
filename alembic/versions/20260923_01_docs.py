"""Add documents and document_chunks tables – Phase 7 RAG support"""

# Revision identifiers, used by Alembic.
revision = "20260923_01_docs"
down_revision = "6b1e2c9a3f7d"
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
# pgvector provides the Vector type
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = sa.LargeBinary  # fallback when pgvector not available

# Embedding dimension – OpenAI text‑embedding‑3‑small returns 1536 dimensions
EMBEDDING_DIM = 1536

def upgrade() -> None:
    # --------------------------------------------------------------
    # 1️⃣ Install the pgvector extension (PostgreSQL only)
    # --------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --------------------------------------------------------------
    # 2️⃣ documents table – business‑level ownership and audit fields
    # --------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column(
            "business_id",
            sa.Integer,
            sa.ForeignKey("businesses.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("filename", sa.String, nullable=False),
        sa.Column("document_type", sa.String, nullable=False),
        sa.Column("source", sa.String, nullable=True),
        sa.Column(
            "uploaded_at",
            sa.DateTime,
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("content_hash", sa.String, nullable=False),
        sa.Column(
            "status",
            sa.String,
            nullable=False,
            server_default="pending",
        ),
        sa.UniqueConstraint(
            "business_id",
            "content_hash",
            name="uq_documents_business_hash",
        ),
    )

    # --------------------------------------------------------------
    # 3️⃣ document_chunks table – chunks belong to a document, business derived via FK
    # --------------------------------------------------------------
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("chunk_number", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        # Embedding generated after chunk creation → nullable
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.UniqueConstraint(
            "document_id",
            "chunk_number",
            name="uq_document_chunk_number",
        ),
    )

    # --------------------------------------------------------------
    # 4️⃣ Indexes for fast vector similarity search
    # --------------------------------------------------------------
    op.create_index(
        "ix_document_chunks_embedding",
        "document_chunks",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

def downgrade() -> None:
    # --------------------------------------------------------------
    # 1️⃣ Drop vector similarity index
    # --------------------------------------------------------------
    op.drop_index("ix_document_chunks_embedding", table_name="document_chunks")

    # --------------------------------------------------------------
    # 2️⃣ Drop document_chunks table
    # --------------------------------------------------------------
    op.drop_table("document_chunks")

    # --------------------------------------------------------------
    # 3️⃣ Drop documents table
    # --------------------------------------------------------------
    op.drop_table("documents")

    # --------------------------------------------------------------
    # 4️⃣ Keep the pgvector extension – do NOT drop it here
    # --------------------------------------------------------------
    # If you ever need to remove it deliberately, uncomment:
    # op.execute("DROP EXTENSION IF EXISTS vector;")
