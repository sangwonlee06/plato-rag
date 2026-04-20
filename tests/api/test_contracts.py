"""Tests for API contract serialization."""

from plato_rag.api.contracts import (
    ChatMode,
    CompatSourceType,
    ConversationRole,
    ConversationTurn,
    GroundingResponse,
    InterpretationLevel,
    LocationRefResponse,
    QueryRequest,
    QueryResponse,
    SourceClass,
    SourceCoverageResponse,
    SourceExposure,
    compat_source_type_for,
)


class TestCompatSourceType:
    def test_primary_maps_to_primary(self) -> None:
        assert compat_source_type_for(SourceClass.PRIMARY_TEXT) == CompatSourceType.PRIMARY

    def test_reference_maps_to_secondary(self) -> None:
        result = compat_source_type_for(SourceClass.REFERENCE_ENCYCLOPEDIA)
        assert result == CompatSourceType.SECONDARY

    def test_peer_reviewed_maps_to_secondary(self) -> None:
        assert compat_source_type_for(SourceClass.PEER_REVIEWED) == CompatSourceType.SECONDARY


class TestQueryRequest:
    def test_serialization(self) -> None:
        req = QueryRequest(
            question="What is recollection?",
            mode=ChatMode.PLATO,
            conversation_history=[
                ConversationTurn(role=ConversationRole.USER, content="Hi"),
            ],
            options={"allowed_collections": ["platonic_dialogues"]},
        )
        data = req.model_dump()
        assert data["question"] == "What is recollection?"
        assert data["mode"] == "PLATO"
        assert len(data["conversation_history"]) == 1
        assert data["options"]["allowed_collections"] == ["platonic_dialogues"]

    def test_frege_mode_serializes(self) -> None:
        req = QueryRequest(question="What is modus ponens?", mode=ChatMode.FREGE)
        assert req.mode == ChatMode.FREGE


class TestQueryResponse:
    def test_minimal_response(self) -> None:
        resp = QueryResponse(
            answer="Recollection is...",
            retrieved_chunks=[],
            citations=[],
            grounding=GroundingResponse(
                interpretation_level=InterpretationLevel.LOW_CONFIDENCE,
                confidence_summary="No sources found.",
                source_coverage=SourceCoverageResponse(),
            ),
            request_id="req_test",
        )
        data = resp.model_dump()
        assert data["api_version"] == "1.0.0"
        assert data["grounding"]["interpretation_level"] == "LOW_CONFIDENCE"

    def test_chunk_and_citation_allow_source_exposure(self) -> None:
        resp = QueryResponse(
            answer="Recollection is...",
            retrieved_chunks=[
                {
                    "id": "chunk-1",
                    "text": "The soul is immortal.",
                    "source_type": CompatSourceType.PRIMARY,
                    "source_class": SourceClass.PRIMARY_TEXT,
                    "source_exposure": SourceExposure.PUBLIC,
                    "trust_tier": 1,
                    "work": "Meno",
                    "author": "Plato",
                    "collection": "platonic_dialogues",
                },
            ],
            citations=[
                {
                    "work": "Meno",
                    "author": "Plato",
                    "source_type": CompatSourceType.PRIMARY,
                    "source_class": SourceClass.PRIMARY_TEXT,
                    "collection": "platonic_dialogues",
                    "source_exposure": SourceExposure.PUBLIC,
                    "trust_tier": 1,
                },
            ],
            grounding=GroundingResponse(
                interpretation_level=InterpretationLevel.DIRECT,
                confidence_summary="Grounded in primary text.",
                source_coverage=SourceCoverageResponse(),
            ),
            request_id="req_test",
        )
        data = resp.model_dump()
        assert data["retrieved_chunks"][0]["source_exposure"] == "PUBLIC"
        assert data["citations"][0]["collection"] == "platonic_dialogues"


class TestSourceCoverage:
    def test_counts_by_class(self) -> None:
        cov = SourceCoverageResponse(
            counts_by_class={
                SourceClass.PRIMARY_TEXT: 2,
                SourceClass.REFERENCE_ENCYCLOPEDIA: 1,
            },
            total_chunks_returned=3,
        )
        assert cov.primary_count == 2
        assert cov.reference_count == 1

    def test_empty_coverage(self) -> None:
        cov = SourceCoverageResponse()
        assert cov.primary_count == 0
        assert cov.reference_count == 0


class TestLocationRefResponse:
    def test_serialization(self) -> None:
        ref = LocationRefResponse(system="stephanus", value="86b", range_end="86d")
        data = ref.model_dump()
        assert data == {"system": "stephanus", "value": "86b", "range_end": "86d"}
