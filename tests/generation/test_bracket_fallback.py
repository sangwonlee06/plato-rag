from __future__ import annotations

from plato_rag.generation.bracket_fallback import (
    parse_bracketed_claims,
    parse_bracketed_generation,
)


def test_parse_bracketed_generation_reconstructs_claims_from_prose() -> None:
    answer, claims = parse_bracketed_generation(
        "Knowledge is often analyzed as justified true belief "
        "[David A. Truncellito, Epistemology, IEP §2]. "
        "Skepticism challenges whether knowledge is possible [IEP §5]."
    )

    assert answer == (
        "Knowledge is often analyzed as justified true belief. "
        "Skepticism challenges whether knowledge is possible."
    )
    assert len(claims) == 2

    first_claim = claims[0]
    assert first_claim.claim == "Knowledge is often analyzed as justified true belief."
    assert len(first_claim.citations) == 1
    assert first_claim.citations[0].work == "Epistemology"
    assert first_claim.citations[0].author == "David A. Truncellito"
    assert first_claim.citations[0].collection == "iep"
    assert first_claim.citations[0].location == "\u00a72"

    second_claim = claims[1]
    assert second_claim.claim == "Skepticism challenges whether knowledge is possible."
    assert len(second_claim.citations) == 1
    assert second_claim.citations[0].work == "IEP"
    assert second_claim.citations[0].collection == "iep"
    assert second_claim.citations[0].location == "\u00a75"


def test_parse_bracketed_generation_keeps_uncited_claims_when_supported_claims_exist() -> None:
    answer, claims = parse_bracketed_generation(
        "Knowledge is often analyzed as justified true belief "
        "[David A. Truncellito, IEP §2]. "
        "Some answers remain underdetermined by the retrieved evidence."
    )

    assert answer == (
        "Knowledge is often analyzed as justified true belief. "
        "Some answers remain underdetermined by the retrieved evidence."
    )
    assert len(claims) == 2
    assert claims[0].citations
    assert claims[1].claim == "Some answers remain underdetermined by the retrieved evidence."
    assert claims[1].citations == []


def test_parse_bracketed_generation_returns_no_claims_without_brackets() -> None:
    answer, claims = parse_bracketed_generation("This is not JSON.")

    assert answer == "This is not JSON."
    assert claims == []


def test_parse_bracketed_claims_splits_semicolon_citation_groups() -> None:
    claims = parse_bracketed_claims(
        "Both sources frame the issue in terms of knowledge and inquiry "
        "[Meno 86b; David A. Truncellito, IEP §1]."
    )

    assert len(claims) == 1
    assert claims[0].claim == "Both sources frame the issue in terms of knowledge and inquiry."
    assert len(claims[0].citations) == 2
    assert claims[0].citations[0].work == "Meno"
    assert claims[0].citations[0].location == "86b"
    assert claims[0].citations[1].work == "IEP"
    assert claims[0].citations[1].author == "David A. Truncellito"
