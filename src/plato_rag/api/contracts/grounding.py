"""Grounding and limitations response models."""

from __future__ import annotations

from pydantic import BaseModel

from plato_rag.api.contracts.common import InterpretationLevel, SourceClass


class SourceCoverageResponse(BaseModel):
    """Chunk counts by source class — extensible via dict."""

    counts_by_class: dict[SourceClass, int] = {}
    total_chunks_searched: int = 0
    total_chunks_returned: int = 0

    @property
    def primary_count(self) -> int:
        return self.counts_by_class.get(SourceClass.PRIMARY_TEXT, 0)

    @property
    def reference_count(self) -> int:
        return self.counts_by_class.get(SourceClass.REFERENCE_ENCYCLOPEDIA, 0)


class GroundingResponse(BaseModel):
    """Grounding quality assessment for a generated answer."""

    interpretation_level: InterpretationLevel
    confidence_summary: str
    limitations: str | None = None
    source_coverage: SourceCoverageResponse
    grounding_notes: list[str] = []
