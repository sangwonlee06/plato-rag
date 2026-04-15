"""Generation protocols: LLM and citation extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from plato_rag.domain.chunk import ChunkData
from plato_rag.domain.source import SourceClass


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass
class ExtractedCitation:
    """A citation parsed from LLM output and matched against retrieval."""

    work: str
    location: str | None = None
    excerpt: str | None = None
    matched_chunk_id: UUID | None = None
    is_grounded: bool = False
    source_class: SourceClass | None = None
    author: str | None = None
    access_url: str | None = None
    translation: str | None = None


class LLM(Protocol):
    async def generate(self, messages: list[LLMMessage]) -> str: ...
    def model_name(self) -> str: ...


class CitationExtractor(Protocol):
    def extract(
        self, generated_text: str, retrieved_chunks: list[ChunkData],
    ) -> list[ExtractedCitation]: ...
