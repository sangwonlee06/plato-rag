"""Ingestion protocols and intermediate data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationRef


@dataclass
class ParsedSection:
    """A section extracted by a parser."""

    title: str | None
    text: str
    location_ref: LocationRef | None = None
    speaker: str | None = None
    interlocutor: str | None = None
    level: int = 0
    subsections: list[ParsedSection] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Output of a parser: structured text with extracted metadata."""

    metadata: DocumentMetadata
    sections: list[ParsedSection]
    raw_text: str


@dataclass(frozen=True)
class ChunkConfig:
    """Parameters governing chunking behavior."""

    max_chunk_tokens: int = 512
    overlap_tokens: int = 64
    preserve_section_boundaries: bool = True
    min_chunk_tokens: int = 50


@dataclass
class RawChunk:
    """A chunk produced by a chunker, before embedding."""

    text: str
    location_ref: LocationRef | None = None
    section_title: str | None = None
    speaker: str | None = None
    interlocutor: str | None = None
    context_type: str | None = None
    chunk_index: int = 0
    token_count: int = 0
    overlap_tokens: int | None = None


class Parser(Protocol):
    """Parses raw source content into structured sections."""

    def parse(self, raw_content: str, metadata: DocumentMetadata) -> ParsedDocument: ...


class Chunker(Protocol):
    """Splits a parsed document into chunks for embedding."""

    def chunk(self, document: ParsedDocument, config: ChunkConfig) -> list[RawChunk]: ...
