"""SQLAlchemy ORM models for documents and chunks.

These map to the domain models but are database-specific. The repository
layer converts between ORM models and domain dataclasses.

trust_tier is NOT stored — it is derived from source_class at query time.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    author: Mapped[str] = mapped_column(String(300), nullable=False)
    source_class: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    collection: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Philosophy metadata
    tradition: Mapped[str | None] = mapped_column(String(100))
    period: Mapped[str | None] = mapped_column(String(100))
    topics: Mapped[list[str] | None] = mapped_column(JSON)

    # Source-specific
    translation: Mapped[str | None] = mapped_column(String(300))
    edition: Mapped[str | None] = mapped_column(String(300))
    source_url: Mapped[str | None] = mapped_column(String(1000))
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Ingestion tracking
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parser_version: Mapped[str | None] = mapped_column(String(50))
    raw_hash: Mapped[str | None] = mapped_column(String(64), unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=lambda: datetime.now(UTC)
    )


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(3072))

    # Source classification (denormalized)
    source_class: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    collection: Mapped[str] = mapped_column(String(100), nullable=False)
    work_title: Mapped[str] = mapped_column(String(500), nullable=False)
    author: Mapped[str] = mapped_column(String(300), nullable=False)

    # Structured location reference (3 columns for queryability)
    location_system: Mapped[str | None] = mapped_column(String(30))
    location_value: Mapped[str | None] = mapped_column(String(100))
    location_range_end: Mapped[str | None] = mapped_column(String(100))

    section_title: Mapped[str | None] = mapped_column(String(500))

    # Dialogue metadata
    speaker: Mapped[str | None] = mapped_column(String(200))
    interlocutor: Mapped[str | None] = mapped_column(String(200))
    context_type: Mapped[str | None] = mapped_column(String(50))

    # Flexible metadata
    extra_metadata: Mapped[dict[str, object] | None] = mapped_column(JSON)

    # Chunking metadata
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    overlap_tokens: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_chunks_source_class_collection", "source_class", "collection"),
    )
