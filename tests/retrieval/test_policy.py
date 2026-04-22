"""Tests for retrieval policy behavior."""

from dataclasses import replace

from plato_rag.domain.chunk import ChunkData, ScoredChunk
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import SourceClass
from plato_rag.retrieval.policy import PLATO_RETRIEVAL_POLICY
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

    def test_iep_orientation_boost_is_below_primary_default_priority(self) -> None:
        policy = PLATO_RETRIEVAL_POLICY
        assert policy.boost_for_collection_query("iep", "orientation") > 1.0
        assert policy.boost_for_tier(1) > (
            policy.boost_for_tier(2) * policy.boost_for_collection_query("iep", "orientation")
        )

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

    def test_boosted_scores_are_set(self, primary_chunk: ChunkData) -> None:
        reranker = SourcePriorityReranker()
        chunks = [ScoredChunk(chunk=primary_chunk, similarity_score=0.80)]
        result = reranker.rerank(chunks, "test", PLATO_RETRIEVAL_POLICY)
        assert result[0].boosted_score is not None
        assert abs(result[0].boosted_score - 0.80 * 1.30) < 0.001

    def test_orientation_query_boosts_iep_but_does_not_equalize_it_with_primary(
        self,
        primary_chunk: ChunkData,
        sep_chunk: ChunkData,
    ) -> None:
        reranker = SourcePriorityReranker()
        iep_chunk = replace(
            sep_chunk,
            collection="iep",
            work_title="Epistemology",
            author="David A. Truncellito",
            text="Epistemology studies knowledge, justification, and belief.",
            extra_metadata={
                "tradition": "cross_tradition",
                "period": "historical_and_contemporary",
                "topics": ["epistemology", "knowledge"],
            },
        )
        primary = replace(
            primary_chunk,
            text="Socrates speaks about recollection and virtue.",
            extra_metadata={
                "tradition": "ancient",
                "period": "classical_greek",
                "topics": ["epistemology", "ethics"],
            },
        )
        chunks = [
            ScoredChunk(chunk=iep_chunk, similarity_score=0.85),
            ScoredChunk(chunk=primary, similarity_score=0.85),
        ]

        result = reranker.rerank(
            chunks,
            "What is epistemology?",
            PLATO_RETRIEVAL_POLICY,
        )

        iep_result = next(sc for sc in result if sc.chunk.collection == "iep")
        assert iep_result.boosted_score is not None
        assert iep_result.boosted_score > 0.85 * 1.15
        assert result[0].chunk.collection == "iep"

    def test_exegetical_query_does_not_get_orientation_iep_boost(
        self,
        primary_chunk: ChunkData,
        sep_chunk: ChunkData,
    ) -> None:
        reranker = SourcePriorityReranker()
        iep_chunk = replace(
            sep_chunk,
            collection="iep",
            work_title="Epistemology",
            author="David A. Truncellito",
            text="Epistemology studies knowledge, justification, and belief.",
            extra_metadata={
                "tradition": "cross_tradition",
                "period": "historical_and_contemporary",
                "topics": ["epistemology", "knowledge"],
            },
        )
        plato_chunk = replace(
            primary_chunk,
            extra_metadata={
                "tradition": "ancient",
                "period": "classical_greek",
                "topics": ["epistemology", "ethics"],
            },
        )
        chunks = [
            ScoredChunk(chunk=iep_chunk, similarity_score=0.85),
            ScoredChunk(chunk=plato_chunk, similarity_score=0.85),
        ]

        orientation_result = reranker.rerank(
            chunks,
            "What is epistemology?",
            PLATO_RETRIEVAL_POLICY,
        )
        exegetical_result = reranker.rerank(
            chunks,
            "What does Plato mean by recollection in the Meno?",
            PLATO_RETRIEVAL_POLICY,
        )

        orientation_iep = next(sc for sc in orientation_result if sc.chunk.collection == "iep")
        exegetical_iep = next(sc for sc in exegetical_result if sc.chunk.collection == "iep")
        assert orientation_iep.boosted_score is not None
        assert exegetical_iep.boosted_score is not None
        assert orientation_iep.boosted_score > exegetical_iep.boosted_score

    def test_topic_aligned_reference_can_beat_ancient_primary_for_general_query(
        self,
        primary_chunk: ChunkData,
    ) -> None:
        reranker = SourcePriorityReranker()
        ancient_primary = replace(
            primary_chunk,
            text="Socrates discusses recollection and the immortality of the soul.",
            extra_metadata={
                "tradition": "ancient",
                "period": "classical_greek",
                "topics": ["epistemology", "ethics"],
            },
        )
        mind_reference = ChunkData(
            id=primary_chunk.id,
            document_id=primary_chunk.document_id,
            text="Consciousness concerns phenomenal awareness and subjective experience.",
            source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
            collection="iep",
            work_title="Consciousness",
            author="Rocco J. Gennaro",
            location_ref=LocationRef(system=LocationSystem.SECTION, value="1"),
            section_title="Introduction",
            extra_metadata={
                "tradition": "cross_tradition",
                "period": "historical_and_contemporary",
                "topics": ["philosophy_of_mind", "consciousness"],
            },
        )
        chunks = [
            ScoredChunk(chunk=ancient_primary, similarity_score=0.84),
            ScoredChunk(chunk=mind_reference, similarity_score=0.80),
        ]

        result = reranker.rerank(
            chunks,
            "What is consciousness in philosophy of mind?",
            PLATO_RETRIEVAL_POLICY,
        )

        assert result[0].chunk.work_title == "Consciousness"

    def test_explicit_plato_query_preserves_primary_priority(
        self,
        primary_chunk: ChunkData,
        sep_chunk: ChunkData,
    ) -> None:
        reranker = SourcePriorityReranker()
        plato_chunk = replace(
            primary_chunk,
            extra_metadata={
                "tradition": "ancient",
                "period": "classical_greek",
                "topics": ["epistemology", "ethics"],
            },
        )
        modern_reference = replace(
            sep_chunk,
            collection="iep",
            work_title="Epistemology",
            author="David A. Truncellito",
            extra_metadata={
                "tradition": "cross_tradition",
                "period": "historical_and_contemporary",
                "topics": ["epistemology", "knowledge"],
            },
        )
        chunks = [
            ScoredChunk(chunk=modern_reference, similarity_score=0.87),
            ScoredChunk(chunk=plato_chunk, similarity_score=0.80),
        ]

        result = reranker.rerank(
            chunks,
            "How does Plato describe recollection in the Meno?",
            PLATO_RETRIEVAL_POLICY,
        )

        assert result[0].chunk.work_title == "Meno"
