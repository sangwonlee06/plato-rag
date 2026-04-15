"""Initial schema with documents and chunks tables.

Revision ID: 001
Create Date: 2026-04-14
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("author", sa.String(300), nullable=False),
        sa.Column("source_class", sa.String(50), nullable=False, index=True),
        sa.Column("collection", sa.String(100), nullable=False, index=True),
        sa.Column("tradition", sa.String(100)),
        sa.Column("period", sa.String(100)),
        sa.Column("topics", sa.JSON()),
        sa.Column("translation", sa.String(300)),
        sa.Column("edition", sa.String(300)),
        sa.Column("source_url", sa.String(1000)),
        sa.Column("last_verified_at", sa.DateTime(timezone=True)),
        sa.Column("ingested_at", sa.DateTime(timezone=True)),
        sa.Column("parser_version", sa.String(50)),
        sa.Column("raw_hash", sa.String(64), unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("document_id", sa.UUID(), nullable=False, index=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(3072)),
        sa.Column("source_class", sa.String(50), nullable=False, index=True),
        sa.Column("collection", sa.String(100), nullable=False),
        sa.Column("work_title", sa.String(500), nullable=False),
        sa.Column("author", sa.String(300), nullable=False),
        sa.Column("location_system", sa.String(30)),
        sa.Column("location_value", sa.String(100)),
        sa.Column("location_range_end", sa.String(100)),
        sa.Column("section_title", sa.String(500)),
        sa.Column("speaker", sa.String(200)),
        sa.Column("interlocutor", sa.String(200)),
        sa.Column("context_type", sa.String(50)),
        sa.Column("extra_metadata", sa.JSON()),
        sa.Column("chunk_index", sa.Integer(), default=0),
        sa.Column("token_count", sa.Integer(), default=0),
        sa.Column("overlap_tokens", sa.Integer()),
        sa.Column("embedding_model", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("ix_chunks_source_class_collection", "chunks", ["source_class", "collection"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
