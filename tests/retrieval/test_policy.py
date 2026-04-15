"""Tests for retrieval policy behavior."""

from plato_rag.domain.chunk import ChunkData, ScoredChunk
from plato_rag.domain.source import SourceClass
from plato_rag.retrieval.policy import PLATO_RETRIEVAL_POLICY, RetrievalPolicy
from plato_rag.retrieval.reranker.source_priority import SourcePriorityReranker


class TestPlatoPolicy:
    def test_max_chunks(self) -> None:
        assert PLATO_RETRIEVAL_POLICY.max_chunks == 5

    def test_primary_quota_exists(self) -> None:
        q = PLATO_RETRIEVAL_POLICY.quota_for(SourceClass.PRIMARY_TEXT)
        assert q is not None
        assert q.min_chunks == 1

    def test_reference_quota_exists(self) -> None:
        q = PLATO_RETRIEVAL_POLICY.quota_for(SourceClass.REFERENCE_ENCYCLOPEDIA)
        assert q is not None
        assert q.min_chunks == 1

    def test_no_quota_for_bibliography(self) -> None:
        assert PLATO_RETRIEVAL_POLICY.quota_for(SourceClass.CURATED_BIBLIOGRAPHY) is None

    def test_primary_boost_highest(self) -> None:
        policy = PLATO_RETRIEVAL_POLICY
        assert policy.boost_for_tier(1) > policy.boost_for_tier(2) > policy.boost_for_tier(3)

    def test_bibliography_penalized(self) -> None:
        assert PLATO_RETRIEVAL_POLICY.boost_for_tier(4) < 1.0

    def test_unknown_tier_returns_neutral(self) -> None:
        assert PLATO_RETRIEVAL_POLICY.boost_for_tier(99) == 1.0

    def test_three_search_stages(self) -> None:
        stages = PLATO_RETRIEVAL_POLICY.search_stages
        assert len(stages) == 3
        assert SourceClass.PRIMARY_TEXT in stages[0].source_classes
        assert SourceClass.REFERENCE_ENCYCLOPEDIA in stages[1].source_classes
        assert SourceClass.PEER_REVIEWED in stages[2].source_classes

    def test_grounding_requires_primary_for_direct(self) -> None:
        assert PLATO_RETRIEVAL_POLICY.grounding.direct_min_primary >= 1


class TestSourcePriorityReranker:
    def test_primary_outranks_secondary_at_equal_similarity(
        self, primary_chunk: ChunkData, sep_chunk: ChunkData
    ) -> None:
        reranker = SourcePriorityReranker()
        chunks = [
            ScoredChunk(chunk=sep_chunk, similarity_score=0.85),
            ScoredChunk(chunk=primary_chunk, similarity_score=0.85),
        ]
        result = reranker.rerank(chunks, "test query", PLATO_RETRIEVAL_POLICY)
        # Primary (tier 1, 1.3x boost) should outrank reference (tier 2, 1.15x boost)
        assert result[0].chunk.source_class == SourceClass.PRIMARY_TEXT
        assert result[1].chunk.source_class == SourceClass.REFERENCE_ENCYCLOPEDIA

    def test_high_similarity_secondary_can_outrank_low_primary(
        self, primary_chunk: ChunkData, sep_chunk: ChunkData
    ) -> None:
        reranker = SourcePriorityReranker()
        chunks = [
            ScoredChunk(chunk=primary_chunk, similarity_score=0.50),
            ScoredChunk(chunk=sep_chunk, similarity_score=0.90),
        ]
        result = reranker.rerank(chunks, "test query", PLATO_RETRIEVAL_POLICY)
        # SEP at 0.90 * 1.15 = 1.035 should outrank primary at 0.50 * 1.30 = 0.65
        assert result[0].chunk.source_class == SourceClass.REFERENCE_ENCYCLOPEDIA

    def test_boosted_scores_are_set(
        self, primary_chunk: ChunkData
    ) -> None:
        reranker = SourcePriorityReranker()
        chunks = [ScoredChunk(chunk=primary_chunk, similarity_score=0.80)]
        result = reranker.rerank(chunks, "test", PLATO_RETRIEVAL_POLICY)
        assert result[0].boosted_score is not None
        assert abs(result[0].boosted_score - 0.80 * 1.30) < 0.001
