"""Shared test fixtures."""

import uuid

import pytest

from plato_rag.domain.chunk import ChunkData
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import SourceClass


@pytest.fixture
def primary_chunk() -> ChunkData:
    return ChunkData(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text="The soul, then, as being immortal, and having been born many times...",
        source_class=SourceClass.PRIMARY_TEXT,
        collection="platonic_dialogues",
        work_title="Meno",
        author="Plato",
        location_ref=LocationRef(system=LocationSystem.STEPHANUS, value="86b"),
        section_title="Recollection Argument",
        speaker="Socrates",
        interlocutor="Meno",
    )


@pytest.fixture
def sep_chunk() -> ChunkData:
    return ChunkData(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text="Plato's doctrine of anamnesis is first explicitly articulated in the Meno...",
        source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
        collection="sep",
        work_title="Plato's Middle Period Metaphysics and Epistemology",
        author="Allan Silverman",
        location_ref=LocationRef(system=LocationSystem.SECTION, value="2.1"),
        section_title="Recollection and the Theory of Forms",
        extra_metadata={"entry_url": "https://plato.stanford.edu/entries/plato-metaphysics/"},
    )


@pytest.fixture
def peer_reviewed_chunk() -> ChunkData:
    return ChunkData(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text="The argument from recollection has been extensively debated...",
        source_class=SourceClass.PEER_REVIEWED,
        collection="jstor_philosophy",
        work_title="Recollection Reconsidered",
        author="Jane Scholar",
        location_ref=LocationRef(system=LocationSystem.PAGE, value="42"),
    )
