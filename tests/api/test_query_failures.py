from __future__ import annotations

import pytest
from fastapi import HTTPException

from plato_rag.api.contracts import ChatMode, InterpretationLevel, QueryRequest
from plato_rag.api.v1.query import query
from plato_rag.config import Settings
from plato_rag.generation.service import GenerationServiceError
from plato_rag.retrieval.service import RetrievalServiceError


class _FailingRetrievalService:
    async def retrieve(self, **_: object) -> object:
        raise RetrievalServiceError("vector search unavailable")


class _SuccessfulRetrievalService:
    async def retrieve(self, **_: object) -> object:
        from plato_rag.retrieval.service import GroundingAssessment, RetrievalResult

        return RetrievalResult(
            chunks=[],
            grounding=GroundingAssessment(
                interpretation_level=InterpretationLevel.LOW_CONFIDENCE,
                confidence_summary="Insufficient source material retrieved.",
                limitations=None,
                source_counts={},
                grounding_notes=[],
                total_searched=0,
            ),
        )


class _FailingGenerationService:
    async def generate(self, **_: object) -> object:
        raise GenerationServiceError("llm unavailable")


@pytest.mark.asyncio
async def test_query_returns_503_when_retrieval_fails() -> None:
    request = QueryRequest(
        question="What is recollection?",
        mode=ChatMode.PLATO,
        options={"allowed_collections": ["platonic_dialogues"]},
    )

    with pytest.raises(HTTPException) as exc_info:
        await query(
            request,
            settings=Settings(),
            retrieval_service=_FailingRetrievalService(),
            generation_service=_FailingGenerationService(),
        )

    assert exc_info.value.status_code == 503
    assert "retrieval backend unavailable" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_query_returns_503_when_generation_fails() -> None:
    request = QueryRequest(
        question="What is recollection?",
        mode=ChatMode.PLATO,
        options={"allowed_collections": ["platonic_dialogues"]},
    )

    with pytest.raises(HTTPException) as exc_info:
        await query(
            request,
            settings=Settings(),
            retrieval_service=_SuccessfulRetrievalService(),
            generation_service=_FailingGenerationService(),
        )

    assert exc_info.value.status_code == 503
    assert "generation backend unavailable" in str(exc_info.value.detail)
