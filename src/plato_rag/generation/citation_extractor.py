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
from plato_rag.domain.philosophy_profile import (
    PhilosophyProfile,
    is_explicit_ancient_query,
    profile_chunk,
    profile_text,
    significant_tokens,
)
from plato_rag.domain.source import collection_exposure
from plato_rag.generation.bracket_fallback import parse_bracketed_claims
from plato_rag.protocols.generation import ExtractedCitation, StructuredClaim

RANGE_SEPARATOR_PATTERN = re.compile(r"\s*[\u2013-]\s*")


@dataclass(frozen=True)
class _ParsedCitation:
    raw: str
    work: str
    location: str | None = None
    collection_hint: str | None = None
    author_hint: str | None = None
    claim_text: str | None = None


@dataclass(frozen=True)
class _CandidateMatch:
    score: float
    chunk: ChunkData


class BasicCitationExtractor:
    """Match structured citations, or bracket-fallback claims, to retrieved chunks."""

    def extract(
        self,
        generated_text: str,
        retrieved_chunks: list[ChunkData],
        *,
        question: str | None = None,
        claims: list[StructuredClaim] | None = None,
    ) -> list[ExtractedCitation]:
        claim_set = claims if claims is not None else parse_bracketed_claims(generated_text)
        parsed_citations = self._parse_structured_claims(claim_set)
        question_profile = profile_text(question or "")
        verified: list[ExtractedCitation] = []

        for citation in parsed_citations:
            match = self._match_to_chunk(
                citation,
                retrieved_chunks,
                question_profile=question_profile,
            )
            if match is None:
                verified.append(
                    ExtractedCitation(
                        work=citation.work,
                        location=citation.location,
                        claim_text=citation.claim_text,
                        is_grounded=False,
                    )
                )
                continue

            verified.append(
                ExtractedCitation(
                    work=match.work_title,
                    location=citation.location or match.location_display,
                    claim_text=citation.claim_text,
                    excerpt=match.text[:200] if match.text else None,
                    matched_chunk_id=match.id,
                    is_grounded=True,
                    match_score=self._score_chunk_match(
                        citation,
                        match,
                        question_profile=question_profile,
                    ),
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

    def _parse_structured_claims(self, claims: list[StructuredClaim]) -> list[_ParsedCitation]:
        results: list[_ParsedCitation] = []
        for claim in claims:
            for citation in claim.citations:
                results.append(
                    _ParsedCitation(
                        raw=citation.work,
                        work=citation.work,
                        location=citation.location,
                        collection_hint=citation.collection,
                        author_hint=citation.author,
                        claim_text=claim.claim,
                    )
                )
        return results

    def _match_to_chunk(
        self,
        citation: _ParsedCitation,
        chunks: list[ChunkData],
        *,
        question_profile: PhilosophyProfile,
    ) -> ChunkData | None:
        candidates: list[_CandidateMatch] = []
        for chunk in chunks:
            score = self._score_chunk_match(
                citation,
                chunk,
                question_profile=question_profile,
            )
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

    def _score_chunk_match(
        self,
        citation: _ParsedCitation,
        chunk: ChunkData,
        *,
        question_profile: PhilosophyProfile,
    ) -> float:
        score = 0.0

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

        claim_support = self._claim_support_score(
            citation.claim_text,
            chunk,
            question_profile=question_profile,
        )
        score += claim_support

        if citation.claim_text and claim_support < 10 and citation.location is None:
            return 0
        if citation.claim_text and claim_support < 6 and citation.collection_hint is not None:
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

    def _claim_support_score(
        self,
        claim_text: str | None,
        chunk: ChunkData,
        *,
        question_profile: PhilosophyProfile,
    ) -> float:
        if not claim_text:
            return 0.0

        claim_profile = profile_text(claim_text)
        chunk_profile = profile_chunk(chunk)
        claim_tokens = significant_tokens(claim_text)
        chunk_tokens = significant_tokens(
            " ".join(
                part
                for part in (
                    chunk.work_title,
                    chunk.author,
                    chunk.section_title or "",
                    chunk.speaker or "",
                    chunk.interlocutor or "",
                    chunk.context_type or "",
                    chunk.text,
                )
                if part
            )
        )

        overlap = len(claim_tokens & chunk_tokens)
        score = min(overlap * 4.0, 28.0)

        score += 10.0 * len(claim_profile.topics & chunk_profile.topics)
        score += 8.0 * len(claim_profile.traditions & chunk_profile.traditions)
        score += 6.0 * len(claim_profile.periods & chunk_profile.periods)

        score += 4.0 * len(question_profile.topics & chunk_profile.topics)
        score += 3.0 * len(question_profile.traditions & chunk_profile.traditions)
        score += 2.0 * len(question_profile.periods & chunk_profile.periods)

        if (
            chunk.collection == "platonic_dialogues"
            and "ancient" in chunk_profile.traditions
            and not is_explicit_ancient_query(question_profile)
            and (
                (question_profile.topics and not (question_profile.topics & chunk_profile.topics))
                or _is_general_philosophy_question(question_profile)
            )
        ):
            score -= 10.0

        if claim_profile.philosophers:
            if claim_profile.philosophers & chunk_profile.philosophers:
                score += 8.0
            else:
                score -= 6.0

        if (
            claim_profile.topics
            and chunk_profile.topics
            and not (claim_profile.topics & chunk_profile.topics)
        ):
            score -= 8.0

        score += _metadata_phrase_bonus(claim_text, chunk)

        return score


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

    expected_tokens = normalized_expected.split()
    actual_tokens = normalized_actual.split()
    if _compatible_token_sequence(expected_tokens, actual_tokens):
        return 75

    expected_token_set = set(expected_tokens)
    actual_token_set = set(actual_tokens)
    if expected_token_set and expected_token_set.issubset(actual_token_set):
        return 70

    if _compatible_token_sequence(actual_tokens, expected_tokens):
        return 62

    if actual_token_set and actual_token_set.issubset(expected_token_set):
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


def _metadata_phrase_bonus(claim_text: str, chunk: ChunkData) -> float:
    score = 0.0
    score += _phrase_match_bonus(claim_text, chunk.author, weight=7.0)
    score += _phrase_match_bonus(claim_text, chunk.section_title, weight=9.0)
    score += _phrase_match_bonus(claim_text, chunk.speaker, weight=12.0)
    score += _phrase_match_bonus(claim_text, chunk.interlocutor, weight=8.0)
    return score


def _phrase_match_bonus(claim_text: str, phrase: str | None, *, weight: float) -> float:
    if not phrase:
        return 0.0

    normalized_claim = _normalize_name(claim_text)
    normalized_phrase = _normalize_name(phrase)
    if not normalized_claim or not normalized_phrase:
        return 0.0

    if normalized_phrase in normalized_claim:
        return weight
    return 0.0


def _compatible_token_sequence(pattern_tokens: list[str], target_tokens: list[str]) -> bool:
    if not pattern_tokens or not target_tokens:
        return False

    target_index = 0
    for pattern_token in pattern_tokens:
        while target_index < len(target_tokens) and not _tokens_compatible(
            pattern_token,
            target_tokens[target_index],
        ):
            target_index += 1
        if target_index == len(target_tokens):
            return False
        target_index += 1
    return True


def _tokens_compatible(expected: str, actual: str) -> bool:
    return (
        expected == actual
        or (len(expected) == 1 and actual.startswith(expected))
        or (len(actual) == 1 and expected.startswith(actual))
    )


def _split_location_range(value: str) -> tuple[str | None, str | None]:
    parts = RANGE_SEPARATOR_PATTERN.split(value.strip(), maxsplit=1)
    if not parts or not parts[0]:
        return None, None
    if len(parts) == 1:
        return parts[0].strip(), None
    return parts[0].strip(), parts[1].strip()


def _is_general_philosophy_question(question_profile: PhilosophyProfile) -> bool:
    return (
        bool(
            question_profile.topics
            & {
                "ethics",
                "metaphysics",
                "epistemology",
                "philosophy_of_mind",
                "philosophy_of_language",
                "logic",
            }
        )
        and not question_profile.traditions
    )
