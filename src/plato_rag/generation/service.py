"""Generation service — orchestrates prompt construction, LLM call, and citation extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.generation.citation_extractor import BasicCitationExtractor
from plato_rag.generation.prompts.philosophy import build_query_messages
from plato_rag.protocols.generation import LLM, CitationExtractor, ExtractedCitation

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    citations: list[ExtractedCitation]
    ungrounded_citations: list[str]


class GenerationService:
    def __init__(
        self,
        llm: LLM,
        extractor: CitationExtractor | None = None,
    ) -> None:
        self._llm = llm
        self._extractor: CitationExtractor = extractor or BasicCitationExtractor()

    async def generate(
        self,
        question: str,
        chunks: list[ScoredChunk],
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> GenerationResult:
        messages = build_query_messages(question, chunks, conversation_history)

        answer = await self._llm.generate(messages)

        # Extract and verify citations
        chunk_data = [sc.chunk for sc in chunks]
        citations = self._extractor.extract(answer, chunk_data)

        grounded = [c for c in citations if c.is_grounded]
        ungrounded = [f"{c.work} {c.location or ''}" for c in citations if not c.is_grounded]

        if ungrounded:
            logger.warning("Ungrounded citations detected: %s", ungrounded)

        return GenerationResult(
            answer=answer,
            citations=grounded,
            ungrounded_citations=ungrounded,
        )
