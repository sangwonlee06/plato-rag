from __future__ import annotations

from uuid import uuid4

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.source import SourceClass
from plato_rag.ingestion.service import _merged_chunk_metadata


def test_merged_chunk_metadata_propagates_discipline_fields() -> None:
    metadata = DocumentMetadata(
        id=uuid4(),
        title="Epistemology",
        author="David A. Truncellito",
        source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
        collection="iep",
        tradition="cross_tradition",
        period="historical_and_contemporary",
        topics=["epistemology", "knowledge", "justification"],
        source_url="https://iep.utm.edu/epistemo/",
    )

    merged = _merged_chunk_metadata(metadata, {"entry_url": "https://iep.utm.edu/epistemo/"})

    assert merged is not None
    assert merged["tradition"] == "cross_tradition"
    assert merged["period"] == "historical_and_contemporary"
    assert merged["topics"] == ["epistemology", "knowledge", "justification"]
    assert merged["source_url"] == "https://iep.utm.edu/epistemo/"
    assert merged["entry_url"] == "https://iep.utm.edu/epistemo/"
