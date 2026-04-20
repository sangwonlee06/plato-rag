"""Add corpus_entry_id to documents for startup bootstrap tracking.

Revision ID: 002
Revises: 001
Create Date: 2026-04-19
"""

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("corpus_entry_id", sa.String(length=100), nullable=True))
    op.create_index(
        "ix_documents_corpus_entry_id",
        "documents",
        ["corpus_entry_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_documents_corpus_entry_id", table_name="documents")
    op.drop_column("documents", "corpus_entry_id")
