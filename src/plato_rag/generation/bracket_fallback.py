"""Fallback parsing for prose outputs that contain inline bracket citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from plato_rag.protocols.generation import StructuredCitation, StructuredClaim

_AUTHOR_WORK_COLLECTION_LOCATION_PATTERN = re.compile(
    r"^(?P<author>.+?),\s*(?P<work>.+?),\s*(?P<collection>SEP|IEP)\s*,?\s*(?P<location>.+)$",
    re.IGNORECASE,
)
_AUTHOR_COLLECTION_LOCATION_PATTERN = re.compile(
    r"^(?P<author>.+?),\s*(?P<collection>SEP|IEP)\s*,?\s*(?P<location>.+)$",
    re.IGNORECASE,
)
_WORK_COLLECTION_LOCATION_PATTERN = re.compile(
    r"^(?P<work>.+?),\s*(?P<collection>SEP|IEP)\s*,?\s*(?P<location>.+)$",
    re.IGNORECASE,
)
_COLLECTION_LOCATION_PATTERN = re.compile(
    r"^(?P<collection>SEP|IEP)\s*,?\s*(?P<location>.+)$",
    re.IGNORECASE,
)
_LOCATION_START_PATTERN = re.compile(
    r"^(?:\u00a7|section\s+|sec\.?\s+|p(?:age)?\.?\s+|chapter\s+|chap\.?\s+|\d)",
    re.IGNORECASE,
)
_MULTISPACE_PATTERN = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r"\s+([,.;:?!])")


@dataclass(frozen=True)
class _Segment:
    kind: str
    text: str


@dataclass(frozen=True)
class _ClaimCandidate:
    claim: str
    citations: list[StructuredCitation]


def parse_bracketed_generation(raw_text: str) -> tuple[str, list[StructuredClaim]]:
    """Parse prose output with inline bracket citations into answer text and claims."""

    segments = _scan_segments(raw_text)
    answer = _render_answer_text(segments)
    candidates = _collect_claim_candidates(segments)

    if not any(candidate.citations for candidate in candidates):
        return answer, []

    claims = [
        StructuredClaim(claim=candidate.claim, citations=candidate.citations)
        for candidate in candidates
        if candidate.claim
    ]
    return answer, claims


def parse_bracketed_claims(raw_text: str) -> list[StructuredClaim]:
    """Extract claim-level citations from prose output with bracket markers."""

    _, claims = parse_bracketed_generation(raw_text)
    return claims


def _scan_segments(text: str) -> list[_Segment]:
    segments: list[_Segment] = []
    text_buffer: list[str] = []
    citation_buffer: list[str] = []
    inside_citation = False

    for character in text:
        if inside_citation:
            if character == "]":
                citation_text = "".join(citation_buffer).strip()
                if citation_text:
                    segments.append(_Segment(kind="citation", text=citation_text))
                citation_buffer = []
                inside_citation = False
                continue
            citation_buffer.append(character)
            continue

        if character == "[":
            if text_buffer:
                segments.append(_Segment(kind="text", text="".join(text_buffer)))
                text_buffer = []
            inside_citation = True
            continue

        text_buffer.append(character)

    if inside_citation:
        text_buffer.extend(["["] + citation_buffer)

    if text_buffer:
        segments.append(_Segment(kind="text", text="".join(text_buffer)))

    return segments


def _collect_claim_candidates(segments: list[_Segment]) -> list[_ClaimCandidate]:
    candidates: list[_ClaimCandidate] = []
    text_buffer: list[str] = []
    citations: list[StructuredCitation] = []

    for segment in segments:
        if segment.kind == "citation":
            citations.extend(_parse_citation_group(segment.text))
            continue

        for character in segment.text:
            text_buffer.append(character)
            if character in ".?!\n":
                _flush_candidate(candidates, text_buffer, citations)

    _flush_candidate(candidates, text_buffer, citations)

    if any(candidate.citations for candidate in candidates):
        return [
            candidate
            for candidate in candidates
            if candidate.citations or _looks_like_substantive_claim(candidate.claim)
        ]

    return []


def _flush_candidate(
    candidates: list[_ClaimCandidate],
    text_buffer: list[str],
    citations: list[StructuredCitation],
) -> None:
    claim_text = _clean_claim_text("".join(text_buffer))
    if claim_text:
        candidates.append(_ClaimCandidate(claim=claim_text, citations=list(citations)))
    text_buffer.clear()
    citations.clear()


def _render_answer_text(segments: list[_Segment]) -> str:
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for segment in segments:
        if segment.kind != "text":
            continue
        current_lines.append(segment.text)

    raw_answer = "".join(current_lines)
    for paragraph in re.split(r"\n\s*\n", raw_answer):
        cleaned = _clean_text_block(paragraph)
        if cleaned:
            paragraphs.append(cleaned)

    return "\n\n".join(paragraphs).strip()


def _parse_citation_group(content: str) -> list[StructuredCitation]:
    citations: list[StructuredCitation] = []
    seen: set[tuple[str, str | None, str | None, str | None]] = set()

    for part in [part.strip() for part in content.split(";") if part.strip()]:
        citation = _parse_citation_part(part)
        key = (citation.work, citation.location, citation.author, citation.collection)
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)

    return citations


def _parse_citation_part(part: str) -> StructuredCitation:
    normalized = _clean_text_block(part)

    author_work_match = _AUTHOR_WORK_COLLECTION_LOCATION_PATTERN.match(normalized)
    if author_work_match is not None:
        return StructuredCitation(
            work=author_work_match.group("work").strip(),
            author=author_work_match.group("author").strip(),
            collection=author_work_match.group("collection").strip().lower(),
            location=author_work_match.group("location").strip(),
        )

    author_match = _AUTHOR_COLLECTION_LOCATION_PATTERN.match(normalized)
    if author_match is not None:
        collection = author_match.group("collection").strip()
        return StructuredCitation(
            work=collection.upper(),
            author=author_match.group("author").strip(),
            collection=collection.lower(),
            location=author_match.group("location").strip(),
        )

    work_match = _WORK_COLLECTION_LOCATION_PATTERN.match(normalized)
    if work_match is not None:
        return StructuredCitation(
            work=work_match.group("work").strip(),
            collection=work_match.group("collection").strip().lower(),
            location=work_match.group("location").strip(),
        )

    collection_match = _COLLECTION_LOCATION_PATTERN.match(normalized)
    if collection_match is not None:
        collection = collection_match.group("collection").strip()
        return StructuredCitation(
            work=collection.upper(),
            collection=collection.lower(),
            location=collection_match.group("location").strip(),
        )

    work_and_location = _split_work_and_location(normalized)
    if work_and_location is not None:
        work, location = work_and_location
        return StructuredCitation(work=work, location=location)

    return StructuredCitation(work=normalized)


def _split_work_and_location(content: str) -> tuple[str, str] | None:
    tokens = content.split()
    for index in range(1, len(tokens)):
        candidate_location = " ".join(tokens[index:])
        if not _looks_like_location(candidate_location):
            continue
        work = " ".join(tokens[:index]).strip(" ,")
        if not work:
            continue
        return work, candidate_location.strip()
    return None


def _looks_like_location(value: str) -> bool:
    if not _LOCATION_START_PATTERN.match(value):
        return False

    normalized = value.strip().lower()
    normalized = normalized.replace("\u2013", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.lstrip("\u00a7").strip()
    normalized = re.sub(r"^(?:section|sec\.?|chapter|chap\.?)\s+", "", normalized).strip()
    normalized = re.sub(r"^p(?:age)?\.?\s*", "", normalized).strip()
    return bool(re.fullmatch(r"[\da-z.]+(?:\s*-\s*[\da-z.]+)?", normalized))


def _clean_claim_text(text: str) -> str:
    cleaned = _clean_text_block(text)
    cleaned = cleaned.strip(" ;,:")
    return cleaned


def _clean_text_block(text: str) -> str:
    cleaned = _MULTISPACE_PATTERN.sub(" ", text.strip())
    cleaned = _SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r"\1", cleaned)
    return cleaned


def _looks_like_substantive_claim(text: str) -> bool:
    return len(re.findall(r"\b\w+\b", text)) >= 4
