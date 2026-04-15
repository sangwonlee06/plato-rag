"""GET /v1/sources — corpus metadata and source class information."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from plato_rag.domain.source import (
    COLLECTION_REGISTRY,
    SOURCE_CLASS_REGISTRY,
    SourceClass,
)

router = APIRouter()


class CollectionInfo(BaseModel):
    name: str
    label: str
    parser_type: str
    chunker_type: str


class SourceClassResponse(BaseModel):
    source_class: SourceClass
    trust_tier: int
    label: str
    description: str
    collections: list[CollectionInfo]


class SourcesResponse(BaseModel):
    source_classes: list[SourceClassResponse]


@router.get("/sources", response_model=SourcesResponse)
async def sources() -> SourcesResponse:
    result: list[SourceClassResponse] = []

    for sc, info in SOURCE_CLASS_REGISTRY.items():
        collections = [
            CollectionInfo(
                name=col.name,
                label=col.label,
                parser_type=col.parser_type,
                chunker_type=col.chunker_type,
            )
            for col in COLLECTION_REGISTRY.values()
            if col.source_class == sc
        ]
        result.append(SourceClassResponse(
            source_class=sc,
            trust_tier=info.trust_tier,
            label=info.label,
            description=info.description,
            collections=collections,
        ))

    return SourcesResponse(source_classes=result)
