"""Document domain model.

A Document represents a single ingested source — one Platonic dialogue,
one SEP entry, one journal article. Trust tier is NOT stored here;
it is derived from source_class via the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from plato_rag.domain.source import SourceClass


@dataclass
class DocumentMetadata:
    """Full metadata for an ingested document."""

    # Identity
    id: UUID
    title: str
    author: str

    # Source classification
    source_class: SourceClass
    collection: str
    corpus_entry_id: str | None = None

    # Philosophy-specific (optional, varies by collection)
    tradition: str | None = None
    period: str | None = None
    topics: list[str] = field(default_factory=list)

    # Source-specific (optional, varies by collection)
    translation: str | None = None
    edition: str | None = None
    source_url: str | None = None
    last_verified_at: datetime | None = None

    # Ingestion tracking
    ingested_at: datetime | None = None
    parser_version: str | None = None
    raw_hash: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None
