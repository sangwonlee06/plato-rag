"""Structured generation parsing for claim-level answer grounding."""

from __future__ import annotations

import json
from json import JSONDecodeError

from plato_rag.protocols.generation import StructuredCitation, StructuredClaim


class StructuredOutputParseError(ValueError):
    """Raised when the LLM response cannot be parsed as the expected JSON envelope."""


def parse_structured_generation(
    raw_text: str,
) -> tuple[str, list[StructuredClaim]]:
    data = _extract_json_object(raw_text)
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise StructuredOutputParseError("Structured generation is missing a non-empty 'answer'")

    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        raise StructuredOutputParseError("Structured generation field 'claims' must be a list")

    claims: list[StructuredClaim] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            raise StructuredOutputParseError("Each structured claim must be an object")

        claim_text = raw_claim.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise StructuredOutputParseError(
                "Each structured claim must include a non-empty 'claim'"
            )

        raw_citations = raw_claim.get("citations", [])
        if not isinstance(raw_citations, list):
            raise StructuredOutputParseError(
                f"Structured claim {claim_text!r} has a non-list 'citations' field"
            )

        citations: list[StructuredCitation] = []
        for raw_citation in raw_citations:
            if not isinstance(raw_citation, dict):
                raise StructuredOutputParseError(
                    f"Structured claim {claim_text!r} contains a non-object citation"
                )

            work = raw_citation.get("work")
            collection = raw_citation.get("collection")
            if not isinstance(work, str) or not work.strip():
                if isinstance(collection, str) and collection.strip():
                    work = collection.strip().upper()
                else:
                    raise StructuredOutputParseError(
                        f"Structured claim {claim_text!r} contains a citation without work"
                    )

            location = raw_citation.get("location")
            author = raw_citation.get("author")
            citations.append(
                StructuredCitation(
                    work=work.strip(),
                    location=(
                        location.strip() if isinstance(location, str) and location.strip() else None
                    ),
                    author=(author.strip() if isinstance(author, str) and author.strip() else None),
                    collection=(
                        collection.strip().lower()
                        if isinstance(collection, str) and collection.strip()
                        else None
                    ),
                )
            )

        claims.append(StructuredClaim(claim=claim_text.strip(), citations=citations))

    return answer.strip(), claims


def _extract_json_object(raw_text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    text = raw_text.strip()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise StructuredOutputParseError("LLM response did not contain a valid JSON object")
