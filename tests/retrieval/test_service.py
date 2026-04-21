from __future__ import annotations

import asyncio
from dataclasses import replace

from plato_rag.domain.chunk import ChunkData, ScoredChunk
from plato_rag.domain.source import SourceClass
from plato_rag.retrieval.policy import PLATO_RETRIEVAL_POLICY
from plato_rag.retrieval.service import RetrievalService, RetrievalServiceError


class _FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def model_name(self) -> str:
        return "fake-embedder"

    def dimensions(self) -> int:
        return 3


class _FailingEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise TimeoutError("embedding timeout")

    def model_name(self) -> str:
        return "failing-embedder"

    def dimensions(self) -> int:
        return 3


class _StageAwareVectorStore:
    def __init__(self, responses: dict[SourceClass, list[ScoredChunk]]) -> None:
        self._responses = responses
        self.calls: list[list[SourceClass]] = []

    async def search(
        self,
        query_vector: list[float],
        filters: object | None = None,
        limit: int = 10,
    ) -> list[ScoredChunk]:
        del query_vector, limit
        source_classes = list(getattr(filters, "source_classes", []) or [])
        self.calls.append(source_classes)
        if not source_classes:
            return []

        source_class = source_classes[0]
        return list(self._responses.get(source_class, []))


def _scored(chunk: ChunkData, similarity_score: float) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, similarity_score=similarity_score)


def test_retrieval_searches_reference_stage_before_early_exit(
    primary_chunk: ChunkData,
    sep_chunk: ChunkData,
) -> None:
    primary_results = [_scored(replace(primary_chunk), 0.92) for _ in range(5)]
    reference_results = [_scored(replace(sep_chunk, collection="iep"), 0.61)]
    store = _StageAwareVectorStore(
        {
            SourceClass.PRIMARY_TEXT: primary_results,
            SourceClass.REFERENCE_ENCYCLOPEDIA: reference_results,
        }
    )
    service = RetrievalService(vector_store=store, embedder=_FakeEmbedder())

    result = asyncio.run(service.retrieve("What is justice?", PLATO_RETRIEVAL_POLICY))

    assert store.calls == [
        [SourceClass.PRIMARY_TEXT],
        [SourceClass.REFERENCE_ENCYCLOPEDIA],
    ]
    assert any(
        scored_chunk.chunk.source_class == SourceClass.REFERENCE_ENCYCLOPEDIA
        for scored_chunk in result.chunks
    )


def test_retrieval_continues_to_later_stages_when_reference_quota_unmet(
    primary_chunk: ChunkData,
    peer_reviewed_chunk: ChunkData,
) -> None:
    primary_results = [_scored(replace(primary_chunk), 0.91) for _ in range(5)]
    peer_results = [_scored(replace(peer_reviewed_chunk), 0.58)]
    store = _StageAwareVectorStore(
        {
            SourceClass.PRIMARY_TEXT: primary_results,
            SourceClass.REFERENCE_ENCYCLOPEDIA: [],
            SourceClass.PEER_REVIEWED: peer_results,
        }
    )
    service = RetrievalService(vector_store=store, embedder=_FakeEmbedder())

    asyncio.run(service.retrieve("What is knowledge?", PLATO_RETRIEVAL_POLICY))

    assert store.calls == [
        [SourceClass.PRIMARY_TEXT],
        [SourceClass.REFERENCE_ENCYCLOPEDIA],
        [SourceClass.PEER_REVIEWED],
    ]


def test_retrieval_wraps_embedding_failures(primary_chunk: ChunkData) -> None:
    del primary_chunk
    store = _StageAwareVectorStore({})
    service = RetrievalService(vector_store=store, embedder=_FailingEmbedder())

    try:
        asyncio.run(service.retrieve("What is justice?", PLATO_RETRIEVAL_POLICY))
    except RetrievalServiceError as exc:
        assert "embed retrieval query" in str(exc)
    else:
        raise AssertionError("Expected RetrievalServiceError")
