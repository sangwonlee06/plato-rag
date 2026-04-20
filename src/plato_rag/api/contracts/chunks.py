"""Retrieved chunk response models."""

from __future__ import annotations

from pydantic import BaseModel

from plato_rag.api.contracts.common import CompatSourceType, SourceClass, SourceExposure


class ChunkMetadataResponse(BaseModel):
    """Chunk-level metadata that varies by source class."""

    # Dialogue-specific
    speaker: str | None = None
    interlocutor: str | None = None

    # Translation / edition
    translation: str | None = None

    # SEP/IEP-specific
    entry_url: str | None = None
    section_title: str | None = None
    last_updated: str | None = None

    # Context
    context_type: str | None = None
    chunk_index: int | None = None
    token_count: int | None = None


class LocationRefResponse(BaseModel):
    """Structured location reference (for consumers that want to parse it)."""

    system: str
    value: str
    range_end: str | None = None


class RetrievedChunkResponse(BaseModel):
    """A retrieved chunk in the API response."""

    id: str
    text: str

    # Source classification — dual representation
    source_type: CompatSourceType
    source_class: SourceClass
    source_exposure: SourceExposure
    trust_tier: int

    # Citation fields
    work: str
    author: str
    location: str | None = None
    location_ref: LocationRefResponse | None = None
    collection: str | None = None

    chunk_metadata: ChunkMetadataResponse | None = None
    similarity_score: float | None = None
