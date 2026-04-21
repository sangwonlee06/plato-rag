"""Citation extraction and verification.

This implementation is still intentionally conservative: it verifies that a
citation can be matched to retrieved evidence, but it does not claim semantic
entailment between the cited chunk and the answer sentence.

Compared with the earlier MVP extractor, this version improves on four points:
- structured parsing for encyclopedia citations and semicolon-separated groups
- normalized title and author matching instead of raw substring checks
- location-aware range matching via ``LocationRef``
- ambiguity handling so weak work-only citations do not ground arbitrarily
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from plato_rag.domain.chunk import ChunkData
from plato_rag.domain.source import collection_exposure
from plato_rag.protocols.generation import ExtractedCitation

CITATION_PATTERN = re.compile(r"\[(?P<content>[^\]]+)\]")
ENCYCLOPEDIA_CITATION_PATTERN = re.compile(
    r"^(?P<author>.+?),\s*(?P<collection>SEP|IEP)\s*(?P<location>.+)$",
    re.IGNORECASE,
)
COLLECTION_ONLY_CITATION_PATTERN = re.compile(
    r"^(?P<collection>SEP|IEP)\s*(?P<location>.+)$",
    re.IGNORECASE,
)
RANGE_SEPARATOR_PATTERN = re.compile(r"\s*[\u2013-]\s*")
LOCATION_START_PATTERN = re.compile(
    r"^(?:\u00a7|section\s+|sec\.?\s+|p(?:age)?\.?\s+|chapter\s+|chap\.?\s+|\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ParsedCitation:
    raw: str
    work: str
    location: str | None = None
    collection_hint: str | None = None
    author_hint: str | None = None


@dataclass(frozen=True)
class _CandidateMatch:
    score: int
    chunk: ChunkData


class BasicCitationExtractor:
    """Extract bracketed citations from generated text and match them conservatively."""

    def extract(
        self,
        generated_text: str,
        retrieved_chunks: list[ChunkData],
    ) -> list[ExtractedCitation]:
        parsed_citations = self._parse_citations(generated_text)
        verified: list[ExtractedCitation] = []

        for citation in parsed_citations:
            match = self._match_to_chunk(citation, retrieved_chunks)
            if match is None:
                verified.append(
                    ExtractedCitation(
                        work=citation.work,
                        location=citation.location,
                        is_grounded=False,
                    )
                )
                continue

            verified.append(
                ExtractedCitation(
                    work=match.work_title,
                    location=citation.location or match.location_display,
                    excerpt=match.text[:200] if match.text else None,
                    matched_chunk_id=match.id,
                    is_grounded=True,
                    source_class=match.source_class,
                    collection=match.collection,
                    source_exposure=collection_exposure(match.collection),
                    author=match.author,
                    access_url=(
                        match.extra_metadata.get("entry_url") if match.extra_metadata else None
                    ),
                )
            )

        return verified

    def _parse_citations(self, text: str) -> list[_ParsedCitation]:
        """Extract citations from bracketed markers.

        Supported shapes:
        - ``[Meno 86b]``
        - ``[Meno 82b-85b]``
        - ``[Brickhouse and Smith, IEP §2]``
        - ``[IEP §2.1]``
        - ``[Meno 86b; Brickhouse and Smith, IEP §2]``
        """
        results: list[_ParsedCitation] = []
        seen: set[str] = set()

        for match in CITATION_PATTERN.finditer(text):
            bracket_content = match.group("content").strip()
            if not bracket_content:
                continue

            for part in self._split_citation_group(bracket_content):
                if part in seen:
                    continue
                seen.add(part)

                encyclopedia_match = ENCYCLOPEDIA_CITATION_PATTERN.match(part)
                if encyclopedia_match is not None:
                    collection = encyclopedia_match.group("collection").lower()
                    results.append(
                        _ParsedCitation(
                            raw=part,
                            work=encyclopedia_match.group("collection").upper(),
                            location=encyclopedia_match.group("location").strip(),
                            collection_hint=collection,
                            author_hint=encyclopedia_match.group("author").strip(),
                        )
                    )
                    continue

                collection_match = COLLECTION_ONLY_CITATION_PATTERN.match(part)
                if collection_match is not None:
                    collection = collection_match.group("collection").lower()
                    results.append(
                        _ParsedCitation(
                            raw=part,
                            work=collection_match.group("collection").upper(),
                            location=collection_match.group("location").strip(),
                            collection_hint=collection,
                        )
                    )
                    continue

                work_and_location = self._split_work_and_location(part)
                if work_and_location is not None:
                    work, location = work_and_location
                    results.append(_ParsedCitation(raw=part, work=work, location=location))
                    continue

                results.append(_ParsedCitation(raw=part, work=part))

        return results

    def _split_citation_group(self, content: str) -> list[str]:
        return [part.strip() for part in re.split(r"\s*;\s*", content) if part.strip()]

    def _split_work_and_location(self, content: str) -> tuple[str, str] | None:
        tokens = content.split()
        for index in range(1, len(tokens)):
            candidate_location = " ".join(tokens[index:])
            if not self._looks_like_location(candidate_location):
                continue
            work = " ".join(tokens[:index]).strip()
            if not work:
                continue
            return work, candidate_location.strip()
        return None

    def _looks_like_location(self, value: str) -> bool:
        if not LOCATION_START_PATTERN.match(value):
            return False

        normalized = value.strip().lower()
        normalized = normalized.replace("\u2013", "-")
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.lstrip("\u00a7").strip()
        normalized = re.sub(r"^(?:section|sec\.?|chapter|chap\.?)\s+", "", normalized).strip()
        normalized = re.sub(r"^p(?:age)?\.?\s*", "", normalized).strip()
        return bool(re.fullmatch(r"[\da-z.]+(?:\s*-\s*[\da-z.]+)?", normalized))

    def _match_to_chunk(
        self,
        citation: _ParsedCitation,
        chunks: list[ChunkData],
    ) -> ChunkData | None:
        candidates: list[_CandidateMatch] = []
        for chunk in chunks:
            score = self._score_chunk_match(citation, chunk)
            if score > 0:
                candidates.append(_CandidateMatch(score=score, chunk=chunk))

        if not candidates:
            return None

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        best = candidates[0]

        if citation.location is None:
            if len(candidates) > 1:
                return None
            return best.chunk

        if len(candidates) > 1 and candidates[1].score == best.score:
            return None

        return best.chunk

    def _score_chunk_match(self, citation: _ParsedCitation, chunk: ChunkData) -> int:
        score = 0

        if citation.collection_hint is not None:
            if chunk.collection != citation.collection_hint:
                return 0
            score += 40

        if citation.author_hint is not None:
            author_score = _name_match_score(citation.author_hint, chunk.author)
            if author_score == 0:
                return 0
            score += author_score

        title_score = _title_match_score(citation.work, chunk.work_title)
        if citation.collection_hint is None and title_score == 0:
            return 0
        score += title_score

        if citation.location is not None:
            if chunk.location_ref is None:
                return 0
            if chunk.location_ref.matches_value(citation.location):
                score += 150
            elif chunk.location_ref.overlaps_raw_value(citation.location):
                score += 125 + self._range_anchor_bonus(citation.location, chunk)
            else:
                return 0

        return score

    def _range_anchor_bonus(self, location: str, chunk: ChunkData) -> int:
        if chunk.location_ref is None:
            return 0

        start, end = _split_location_range(location)
        if start is not None and chunk.location_ref.matches_value(start):
            return 10
        if end is not None and chunk.location_ref.matches_value(end):
            return 5
        return 0


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = normalized.casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _name_match_score(expected: str, actual: str) -> int:
    normalized_expected = _normalize_name(expected)
    normalized_actual = _normalize_name(actual)
    if not normalized_expected or not normalized_actual:
        return 0
    if normalized_expected == normalized_actual:
        return 90

    expected_tokens = set(normalized_expected.split())
    actual_tokens = set(normalized_actual.split())
    if expected_tokens and expected_tokens.issubset(actual_tokens):
        return 70
    if actual_tokens and actual_tokens.issubset(expected_tokens):
        return 60
    return 0


def _title_match_score(expected: str, actual: str) -> int:
    normalized_expected = _normalize_title(expected)
    normalized_actual = _normalize_title(actual)
    if not normalized_expected or not normalized_actual:
        return 0
    if normalized_expected == normalized_actual:
        return 100
    if normalized_expected in normalized_actual:
        return 85
    if normalized_actual in normalized_expected:
        return 75

    expected_tokens = set(normalized_expected.split())
    actual_tokens = set(normalized_actual.split())
    if len(expected_tokens) >= 2 and expected_tokens.issubset(actual_tokens):
        return 70
    return 0


def _normalize_title(value: str) -> str:
    normalized = _normalize_name(value)
    normalized = re.sub(r"\bbook\s+[ivxlcdm]+\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _split_location_range(value: str) -> tuple[str | None, str | None]:
    parts = RANGE_SEPARATOR_PATTERN.split(value.strip(), maxsplit=1)
    if not parts or not parts[0]:
        return None, None
    if len(parts) == 1:
        return parts[0].strip(), None
    return parts[0].strip(), parts[1].strip()
