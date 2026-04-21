from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from plato_rag.domain.chunk import ChunkData, ScoredChunk
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import SourceClass
from plato_rag.generation.service import GenerationService, GenerationServiceError
from plato_rag.protocols.generation import ExtractedCitation, StructuredClaim


class _FakeLLM:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def generate(self, messages: list[object]) -> str:
        del messages
        return self._response_text

    def model_name(self) -> str:
        return "fake-llm"


class _FailingLLM:
    async def generate(self, messages: list[object]) -> str:
        del messages
        raise TimeoutError("llm timeout")

    def model_name(self) -> str:
        return "failing-llm"


@dataclass
class _RecordingExtractor:
    claims_seen: list[StructuredClaim] | None = None
    generated_text_seen: str | None = None

    def extract(
        self,
        generated_text: str,
        retrieved_chunks: list[ChunkData],
        *,
        question: str | None = None,
        claims: list[StructuredClaim] | None = None,
    ) -> list[ExtractedCitation]:
        del retrieved_chunks, question
        self.generated_text_seen = generated_text
        self.claims_seen = claims
        if claims:
            return [
                ExtractedCitation(
                    work="Epistemology",
                    location="\u00a72",
                    claim_text=claims[0].claim,
                    is_grounded=True,
                    source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
                    collection="iep",
                    author="David A. Truncellito",
                )
            ]
        return []


def _scored_chunk() -> ScoredChunk:
    return ScoredChunk(
        chunk=ChunkData(
            id=uuid4(),
            document_id=uuid4(),
            text="Knowledge is often analyzed as justified true belief.",
            source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
            collection="iep",
            work_title="Epistemology",
            author="David A. Truncellito",
            location_ref=LocationRef(system=LocationSystem.SECTION, value="2"),
        ),
        similarity_score=0.9,
    )


@pytest.mark.asyncio
async def test_generation_service_uses_structured_json_claims() -> None:
    extractor = _RecordingExtractor()
    service = GenerationService(
        llm=_FakeLLM(
            """
            {
              "answer": "Knowledge is often analyzed as justified true belief.",
              "claims": [
                {
                  "claim": "Knowledge is often analyzed as justified true belief.",
                  "citations": [
                    {
                      "work": "Epistemology",
                      "author": "David A. Truncellito",
                      "collection": "iep",
                      "location": "§2"
                    }
                  ]
                }
              ]
            }
            """
        ),
        extractor=extractor,
    )

    result = await service.generate("What is knowledge?", [_scored_chunk()])

    assert result.answer == "Knowledge is often analyzed as justified true belief."
    assert extractor.claims_seen is not None
    assert extractor.claims_seen[0].claim == "Knowledge is often analyzed as justified true belief."
    assert result.unsupported_claims == []


@pytest.mark.asyncio
async def test_generation_service_uses_bracketed_fallback_claims_when_json_parse_fails() -> None:
    extractor = _RecordingExtractor()
    service = GenerationService(
        llm=_FakeLLM(
            "Knowledge is often analyzed as justified true belief "
            "[David A. Truncellito, IEP §2]."
        ),
        extractor=extractor,
    )

    result = await service.generate("What is knowledge?", [_scored_chunk()])

    assert result.answer == "Knowledge is often analyzed as justified true belief."
    assert extractor.claims_seen is not None
    assert extractor.claims_seen[0].claim == "Knowledge is often analyzed as justified true belief."
    assert extractor.claims_seen[0].citations[0].work == "IEP"
    assert result.unsupported_claims == []


@pytest.mark.asyncio
async def test_generation_service_returns_plaintext_when_fallback_finds_no_claims() -> None:
    extractor = _RecordingExtractor()
    service = GenerationService(
        llm=_FakeLLM("This is not JSON."),
        extractor=extractor,
    )

    result = await service.generate("What is knowledge?", [_scored_chunk()])

    assert result.answer == "This is not JSON."
    assert extractor.claims_seen is None
    assert result.unsupported_claims == []


@pytest.mark.asyncio
async def test_generation_service_wraps_llm_failures() -> None:
    service = GenerationService(llm=_FailingLLM())

    with pytest.raises(GenerationServiceError, match="generate answer"):
        await service.generate("What is knowledge?", [_scored_chunk()])
