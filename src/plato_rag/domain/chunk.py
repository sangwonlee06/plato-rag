"""Chunk domain model.

A Chunk is the atomic unit of retrieval. Source classification is
denormalized from its parent Document for query performance.
Trust tier is NOT stored — it is derived from source_class via
the registry at query time.

Location references use the structured LocationRef type instead of
opaque strings, enabling correct citation formatting and reliable
matching in the CitationExtractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from plato_rag.domain.location import LocationRef
from plato_rag.domain.source import SourceClass


@dataclass
class ChunkData:
    """A chunk of text with full metadata for retrieval and citation."""

    id: UUID
    document_id: UUID
    text: str

    # Source classification (denormalized from Document)
    source_class: SourceClass
    collection: str
    work_title: str
    author: str

    # Location within source (structured for citation quality)
    location_ref: LocationRef | None = None
    section_title: str | None = None

    # Dialogue metadata (Platonic dialogues and similar sources)
    # These are first-class fields because dialogues are a major genre
    # for the initial corpus. Other genre-specific metadata can use
    # the extra_metadata dict.
    speaker: str | None = None
    interlocutor: str | None = None

    # Context classification
    context_type: str | None = None     # "argument", "definition", "example", "commentary"

    # Flexible metadata for collection-specific fields not covered above
    # e.g., {"entry_url": "...", "last_updated": "..."} for SEP
    extra_metadata: dict[str, str] | None = None

    # Chunking metadata
    chunk_index: int = 0
    token_count: int = 0
    overlap_tokens: int | None = None
    embedding_model: str | None = None

    created_at: datetime | None = None

    @property
    def location_display(self) -> str | None:
        """Format the location reference as a display string."""
        if self.location_ref is None:
            return None
        return self.location_ref.display()


@dataclass
class ScoredChunk:
    """A chunk with retrieval and reranking scores."""

    chunk: ChunkData
    similarity_score: float
    boosted_score: float | None = None

    @property
    def effective_score(self) -> float:
        return self.boosted_score if self.boosted_score is not None else self.similarity_score
