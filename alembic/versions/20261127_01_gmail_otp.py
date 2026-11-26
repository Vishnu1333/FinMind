"""Create persistent email OTP challenge storage.

Revision ID: 20261127_01_otp
Revises: 61ce229eab88
Create Date: 2026-11-27
"""
from alembic import op
import sqlalchemy as sa

revision = "20261127_01_otp"
down_revision = "61ce229eab88"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "otp_challenges",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("purpose", sa.String(length=20), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("business_name", sa.String(), nullable=True),
        sa.Column("hashed_password", sa.String(), nullable=True),
        sa.Column("otp_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("resend_after", sa.DateTime(), nullable=False),
        sa.Column("consumed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_otp_challenges_id", "otp_challenges", ["id"], unique=False)
    op.create_index("ix_otp_challenges_email", "otp_challenges", ["email"], unique=False)
    op.create_index("ix_otp_challenges_user_id", "otp_challenges", ["user_id"], unique=False)
    op.create_index(
        "ix_otp_email_purpose_created",
        "otp_challenges",
        ["email", "purpose", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_otp_email_purpose_created", table_name="otp_challenges")
    op.drop_index("ix_otp_challenges_user_id", table_name="otp_challenges")
    op.drop_index("ix_otp_challenges_email", table_name="otp_challenges")
    op.drop_index("ix_otp_challenges_id", table_name="otp_challenges")
    op.drop_table("otp_challenges")
