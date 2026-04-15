"""Citation response models."""

from __future__ import annotations

from pydantic import BaseModel

from plato_rag.api.contracts.common import CompatSourceType, SourceClass


class CitationResponse(BaseModel):
    """A verified citation extracted from the generated answer."""

    work: str
    author: str
    location: str | None = None
    excerpt: str | None = None

    source_type: CompatSourceType
    source_class: SourceClass
    trust_tier: int

    access_url: str | None = None
    translation: str | None = None
