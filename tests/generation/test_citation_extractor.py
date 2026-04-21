from __future__ import annotations

from uuid import uuid4

from plato_rag.domain.chunk import ChunkData
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import SourceClass
from plato_rag.generation.citation_extractor import BasicCitationExtractor


def _chunk(
    *,
    work_title: str,
    author: str,
    collection: str,
    text: str,
    location_ref: LocationRef | None = None,
) -> ChunkData:
    return ChunkData(
        id=uuid4(),
        document_id=uuid4(),
        text=text,
        source_class=(
            SourceClass.PRIMARY_TEXT
            if collection in {"platonic_dialogues", "aristotle_corpus"}
            else SourceClass.REFERENCE_ENCYCLOPEDIA
        ),
        collection=collection,
        work_title=work_title,
        author=author,
        location_ref=location_ref,
    )


def test_extractor_matches_iep_author_and_section_citations() -> None:
    extractor = BasicCitationExtractor()
    chunks = [
        _chunk(
            work_title="Epistemology",
            author="David A. Truncellito",
            collection="iep",
            text="Epistemology studies knowledge and justified belief.",
            location_ref=LocationRef(system=LocationSystem.SECTION, value="2"),
        )
    ]

    citations = extractor.extract(
        "The field centers on knowledge and justification [David A. Truncellito, IEP §2].",
        chunks,
    )

    assert len(citations) == 1
    assert citations[0].is_grounded is True
    assert citations[0].work == "Epistemology"
    assert citations[0].location == "\u00a72"


def test_extractor_matches_primary_text_ranges_to_overlapping_chunks() -> None:
    extractor = BasicCitationExtractor()
    chunks = [
        _chunk(
            work_title="Meno",
            author="Plato",
            collection="platonic_dialogues",
            text="Socrates begins the inquiry into virtue.",
            location_ref=LocationRef(system=LocationSystem.STEPHANUS, value="82b"),
        ),
        _chunk(
            work_title="Meno",
            author="Plato",
            collection="platonic_dialogues",
            text="The argument continues through the slave-boy demonstration.",
            location_ref=LocationRef(system=LocationSystem.STEPHANUS, value="83a"),
        ),
    ]

    citations = extractor.extract(
        "The inquiry begins early in the dialogue [Meno 82b-83a].",
        chunks,
    )

    assert len(citations) == 1
    assert citations[0].is_grounded is True
    assert citations[0].location == "82b-83a"
    assert citations[0].matched_chunk_id == chunks[0].id


def test_extractor_matches_bekker_range_citations() -> None:
    extractor = BasicCitationExtractor()
    chunks = [
        _chunk(
            work_title="Nicomachean Ethics",
            author="Aristotle",
            collection="aristotle_corpus",
            text="Every art and every inquiry seems to aim at some good.",
            location_ref=LocationRef(
                system=LocationSystem.BEKKER,
                value="1094a1",
                range_end="1094a20",
            ),
        )
    ]

    citations = extractor.extract(
        "Aristotle opens the Ethics teleologically [Nicomachean Ethics 1094a1-1094a20].",
        chunks,
    )

    assert len(citations) == 1
    assert citations[0].is_grounded is True
    assert citations[0].location == "1094a1-1094a20"


def test_extractor_splits_multiple_citations_in_one_bracket() -> None:
    extractor = BasicCitationExtractor()
    chunks = [
        _chunk(
            work_title="Meno",
            author="Plato",
            collection="platonic_dialogues",
            text="The dialogue frames inquiry in terms of knowledge and learning.",
            location_ref=LocationRef(system=LocationSystem.STEPHANUS, value="86b"),
        ),
        _chunk(
            work_title="Epistemology",
            author="David A. Truncellito",
            collection="iep",
            text="The entry introduces epistemology as the study of knowledge.",
            location_ref=LocationRef(system=LocationSystem.SECTION, value="1"),
        ),
    ]

    citations = extractor.extract(
        (
            "Both sources frame the issue in terms of knowledge and inquiry "
            "[Meno 86b; David A. Truncellito, IEP §1]."
        ),
        chunks,
        question="How is knowledge framed in philosophy?",
    )

    assert len(citations) == 2
    assert all(citation.is_grounded for citation in citations)


def test_extractor_leaves_ambiguous_work_only_citations_ungrounded() -> None:
    extractor = BasicCitationExtractor()
    chunks = [
        _chunk(
            work_title="Meno",
            author="Plato",
            collection="platonic_dialogues",
            text="First chunk from the Meno.",
            location_ref=LocationRef(system=LocationSystem.STEPHANUS, value="86b"),
        ),
        _chunk(
            work_title="Meno",
            author="Plato",
            collection="platonic_dialogues",
            text="Second chunk from the Meno.",
            location_ref=LocationRef(system=LocationSystem.STEPHANUS, value="87a"),
        ),
    ]

    citations = extractor.extract("Plato addresses the issue in the dialogue [Meno].", chunks)

    assert len(citations) == 1
    assert citations[0].is_grounded is False


def test_extractor_prefers_claim_aligned_chunk_with_same_section_reference() -> None:
    extractor = BasicCitationExtractor()
    chunks = [
        _chunk(
            work_title="Epistemology",
            author="David A. Truncellito",
            collection="iep",
            text="Knowledge is often analyzed as justified true belief.",
            location_ref=LocationRef(system=LocationSystem.SECTION, value="2"),
        ),
        _chunk(
            work_title="Epistemology",
            author="David A. Truncellito",
            collection="iep",
            text="Skepticism challenges whether knowledge is possible.",
            location_ref=LocationRef(system=LocationSystem.SECTION, value="2"),
        ),
    ]

    citations = extractor.extract(
        (
            "A standard analysis treats knowledge as justified true belief "
            "[David A. Truncellito, IEP §2]."
        ),
        chunks,
        question="What is knowledge in epistemology?",
    )

    assert len(citations) == 1
    assert citations[0].is_grounded is True
    assert citations[0].matched_chunk_id == chunks[0].id


def test_extractor_rejects_weak_claim_level_match() -> None:
    extractor = BasicCitationExtractor()
    chunks = [
        _chunk(
            work_title="Consciousness",
            author="Rocco J. Gennaro",
            collection="iep",
            text="Consciousness concerns phenomenal awareness and subjective experience.",
            location_ref=LocationRef(system=LocationSystem.SECTION, value="1"),
        ),
    ]

    citations = extractor.extract(
        "Kant's account of the categorical imperative is central here [Rocco J. Gennaro, IEP §1].",
        chunks,
        question="Explain Kantian ethics.",
    )

    assert len(citations) == 1
    assert citations[0].is_grounded is False
