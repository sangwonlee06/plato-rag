"""Generation service — orchestrates prompt construction, LLM call, and citation extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.generation.citation_extractor import BasicCitationExtractor
from plato_rag.generation.prompts.philosophy import build_query_messages
from plato_rag.generation.structured_output import (
    StructuredOutputParseError,
    parse_structured_generation,
)
from plato_rag.protocols.generation import (
    LLM,
    CitationExtractor,
    ExtractedCitation,
    StructuredClaim,
)

logger = logging.getLogger(__name__)


class GenerationServiceError(RuntimeError):
    """Raised when answer generation cannot complete operationally."""


@dataclass
class GenerationResult:
    answer: str
    citations: list[ExtractedCitation]
    ungrounded_citations: list[str]
    unsupported_claims: list[str]


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

        try:
            raw_output = await self._llm.generate(messages)
        except Exception as exc:
            raise GenerationServiceError("Failed to generate answer from LLM") from exc
        answer, claims = self._parse_or_fallback(raw_output)

        # Extract and verify citations
        chunk_data = [sc.chunk for sc in chunks]
        citations = self._extractor.extract(
            raw_output if not claims else answer,
            chunk_data,
            question=question,
            claims=claims or None,
        )

        grounded = [c for c in citations if c.is_grounded]
        ungrounded = [f"{c.work} {c.location or ''}" for c in citations if not c.is_grounded]
        unsupported_claims = _unsupported_claims(claims, grounded)

        if ungrounded:
            logger.warning("Ungrounded citations detected: %s", ungrounded)
        if unsupported_claims:
            logger.warning("Unsupported claims detected: %s", unsupported_claims)

        return GenerationResult(
            answer=answer,
            citations=grounded,
            ungrounded_citations=ungrounded,
            unsupported_claims=unsupported_claims,
        )

    def _parse_or_fallback(self, raw_output: str) -> tuple[str, list[StructuredClaim]]:
        try:
            return parse_structured_generation(raw_output)
        except StructuredOutputParseError as exc:
            logger.warning("Structured generation parse failed; falling back: %s", exc)
            return raw_output, []


def _unsupported_claims(
    claims: list[StructuredClaim],
    grounded_citations: list[ExtractedCitation],
) -> list[str]:
    if not claims:
        return []

    grounded_claims = {
        citation.claim_text.strip()
        for citation in grounded_citations
        if citation.claim_text is not None and citation.claim_text.strip()
    }

    unsupported: list[str] = []
    for claim in claims:
        if not claim.citations:
            unsupported.append(claim.claim)
            continue
        if claim.claim not in grounded_claims:
            unsupported.append(claim.claim)
    return unsupported
