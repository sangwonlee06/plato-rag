"""Evaluation scoring helpers."""

from __future__ import annotations

import re

from plato_rag.api.contracts import CitationResponse, QueryResponse
from plato_rag.evaluation.dataset import (
    CitationExpectation,
    EvaluationCase,
    EvaluationCaseResult,
)


def evaluate_case_response(
    case: EvaluationCase,
    response: QueryResponse,
) -> EvaluationCaseResult:
    """Score one API response against a curated evaluation case."""

    failures: list[str] = []
    expectations = case.expectations

    normalized_answer = _normalize_text(response.answer)
    for phrase in expectations.answer_must_contain:
        if _normalize_text(phrase) not in normalized_answer:
            failures.append(f"answer missing expected phrase: {phrase!r}")

    for phrase in expectations.answer_must_not_contain:
        if _normalize_text(phrase) in normalized_answer:
            failures.append(f"answer contains forbidden phrase: {phrase!r}")

    retrieved_chunks = response.retrieved_chunks
    citations = response.citations

    for work in expectations.required_retrieved_works:
        if not any(_text_equals(chunk.work, work) for chunk in retrieved_chunks):
            failures.append(f"missing required retrieved work: {work}")

    if expectations.required_retrieved_works_any_of and not any(
        _text_equals(chunk.work, work)
        for chunk in retrieved_chunks
        for work in expectations.required_retrieved_works_any_of
    ):
        failures.append(
            "missing any-of retrieved works: "
            + ", ".join(expectations.required_retrieved_works_any_of)
        )

    for work in expectations.forbidden_retrieved_works:
        if any(_text_equals(chunk.work, work) for chunk in retrieved_chunks):
            failures.append(f"retrieved forbidden work: {work}")

    for collection in expectations.required_retrieved_collections:
        if not any(_text_equals(chunk.collection, collection) for chunk in retrieved_chunks):
            failures.append(f"missing required retrieved collection: {collection}")

    for collection in expectations.forbidden_retrieved_collections:
        if any(_text_equals(chunk.collection, collection) for chunk in retrieved_chunks):
            failures.append(f"retrieved forbidden collection: {collection}")

    if len(citations) < expectations.min_citations:
        failures.append(
            f"citation count {len(citations)} below minimum {expectations.min_citations}"
        )

    for citation_expectation in expectations.required_citations:
        if not any(_citation_matches(citation_expectation, citation) for citation in citations):
            failures.append(
                "missing required citation: "
                + _render_citation_expectation(citation_expectation)
            )

    if expectations.required_citations_any_of and not any(
        _citation_matches(citation_expectation, citation)
        for citation_expectation in expectations.required_citations_any_of
        for citation in citations
    ):
        failures.append(
            "missing any-of citation expectations: "
            + ", ".join(
                _render_citation_expectation(expectation)
                for expectation in expectations.required_citations_any_of
            )
        )

    if (
        expectations.allowed_interpretation_levels
        and response.grounding.interpretation_level
        not in expectations.allowed_interpretation_levels
    ):
        failures.append(
            "unexpected interpretation level: "
            + response.grounding.interpretation_level.value
        )

    ungrounded_citation_count = len(response.debug.ungrounded_citations) if response.debug else 0
    if ungrounded_citation_count > expectations.max_ungrounded_citations:
        failures.append(
            "ungrounded citation count "
            f"{ungrounded_citation_count} exceeds {expectations.max_ungrounded_citations}"
        )

    unsupported_claim_count = len(response.debug.unsupported_claims) if response.debug else 0
    if unsupported_claim_count > expectations.max_unsupported_claims:
        failures.append(
            "unsupported claim count "
            f"{unsupported_claim_count} exceeds {expectations.max_unsupported_claims}"
        )

    return EvaluationCaseResult(
        case_id=case.id,
        passed=not failures,
        failures=failures,
        citation_count=len(citations),
        retrieved_chunk_count=len(retrieved_chunks),
        ungrounded_citation_count=ungrounded_citation_count,
        unsupported_claim_count=unsupported_claim_count,
    )


def _citation_matches(
    expectation: CitationExpectation,
    citation: CitationResponse,
) -> bool:
    if expectation.work is not None and not _text_equals(citation.work, expectation.work):
        return False
    if expectation.author is not None and not _text_contains(citation.author, expectation.author):
        return False
    if expectation.collection is not None and not _text_equals(
        citation.collection,
        expectation.collection,
    ):
        return False
    if expectation.location is not None and not _location_equals(
        citation.location,
        expectation.location,
    ):
        return False
    return (
        expectation.source_class is None
        or citation.source_class == expectation.source_class
    )


def _render_citation_expectation(expectation: CitationExpectation) -> str:
    parts: list[str] = []
    if expectation.work is not None:
        parts.append(f"work={expectation.work}")
    if expectation.author is not None:
        parts.append(f"author={expectation.author}")
    if expectation.collection is not None:
        parts.append(f"collection={expectation.collection}")
    if expectation.location is not None:
        parts.append(f"location={expectation.location}")
    if expectation.source_class is not None:
        parts.append(f"source_class={expectation.source_class.value}")
    return ", ".join(parts)


def _text_equals(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    return _normalize_text(actual) == _normalize_text(expected)


def _text_contains(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    return _normalize_text(expected) in _normalize_text(actual)


def _location_equals(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    return _normalize_location(actual) == _normalize_location(expected)


def _normalize_text(value: str) -> str:
    normalized = value.casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_location(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold().replace("§", ""))
