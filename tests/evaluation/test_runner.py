from __future__ import annotations

from plato_rag.api.contracts import (
    CitationResponse,
    CompatSourceType,
    DebugResponse,
    GroundingResponse,
    InterpretationLevel,
    QueryResponse,
    RetrievedChunkResponse,
    SourceClass,
    SourceCoverageResponse,
)
from plato_rag.domain.source import SourceExposure
from plato_rag.evaluation import EvaluationCase, evaluate_case_response


def _response(
    *,
    answer: str,
    works: list[str],
    citations: list[CitationResponse],
    ungrounded: list[str] | None = None,
    unsupported_claims: list[str] | None = None,
) -> QueryResponse:
    retrieved_chunks = [
        RetrievedChunkResponse(
            id=f"chunk-{index}",
            text=f"{work} supporting text",
            source_type=CompatSourceType.SECONDARY,
            source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
            source_exposure=SourceExposure.PUBLIC,
            trust_tier=2,
            work=work,
            author="Test Author",
            collection="iep",
        )
        for index, work in enumerate(works, start=1)
    ]
    return QueryResponse(
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        citations=citations,
        grounding=GroundingResponse(
            interpretation_level=InterpretationLevel.DIRECT,
            confidence_summary="Grounded in retrieved sources.",
            source_coverage=SourceCoverageResponse(
                counts_by_class={SourceClass.REFERENCE_ENCYCLOPEDIA: len(retrieved_chunks)},
                total_chunks_searched=20,
                total_chunks_returned=len(retrieved_chunks),
            ),
        ),
        debug=DebugResponse(
            ungrounded_citations=ungrounded or [],
            unsupported_claims=unsupported_claims or [],
        ),
        request_id="req_test",
    )


def test_evaluate_case_response_passes_for_matching_response() -> None:
    case = EvaluationCase.model_validate(
        {
            "id": "knowledge_case",
            "question": "What is knowledge?",
            "expectations": {
                "answer_must_contain": ["knowledge"],
                "required_retrieved_works": ["Epistemology"],
                "required_citations": [{"work": "Epistemology", "collection": "iep"}],
                "min_citations": 1,
                "max_ungrounded_citations": 0,
                "max_unsupported_claims": 0,
            },
        }
    )
    response = _response(
        answer="Knowledge is discussed in epistemology.",
        works=["Epistemology"],
        citations=[
            CitationResponse(
                work="Epistemology",
                author="David A. Truncellito",
                location="§2",
                source_type=CompatSourceType.SECONDARY,
                source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
                collection="iep",
                source_exposure=SourceExposure.PUBLIC,
                trust_tier=2,
            )
        ],
    )

    result = evaluate_case_response(case, response)

    assert result.passed is True
    assert result.failures == []


def test_evaluate_case_response_reports_missing_required_citation() -> None:
    case = EvaluationCase.model_validate(
        {
            "id": "meaning_case",
            "question": "What is meaning?",
            "expectations": {
                "required_citations": [{"work": "Meaning and Communication", "collection": "iep"}],
            },
        }
    )
    response = _response(
        answer="Meaning and communication are related.",
        works=["Philosophy of Language"],
        citations=[
            CitationResponse(
                work="Philosophy of Language",
                author="Michael P. Wolf",
                source_type=CompatSourceType.SECONDARY,
                source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
                collection="iep",
                source_exposure=SourceExposure.PUBLIC,
                trust_tier=2,
            )
        ],
    )

    result = evaluate_case_response(case, response)

    assert result.passed is False
    assert (
        "missing required citation: work=Meaning and Communication, collection=iep"
        in result.failures
    )


def test_evaluate_case_response_supports_any_of_work_expectations() -> None:
    case = EvaluationCase.model_validate(
        {
            "id": "jtb_case",
            "question": "What is the justified true belief account?",
            "expectations": {
                "required_retrieved_works_any_of": ["Epistemology", "Knowledge"],
                "required_citations_any_of": [
                    {"work": "Epistemology", "collection": "iep"},
                    {"work": "Knowledge", "collection": "iep"},
                ],
            },
        }
    )
    response = _response(
        answer="The justified true belief account analyzes knowledge.",
        works=["Knowledge"],
        citations=[
            CitationResponse(
                work="Knowledge",
                author="Stephen Hetherington",
                source_type=CompatSourceType.SECONDARY,
                source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
                collection="iep",
                source_exposure=SourceExposure.PUBLIC,
                trust_tier=2,
            )
        ],
    )

    result = evaluate_case_response(case, response)

    assert result.passed is True
