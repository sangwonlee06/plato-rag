"""GET /v1/sources — corpus metadata and source class information."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from plato_rag.config import Settings
from plato_rag.dependencies import get_settings
from plato_rag.domain.source import (
    COLLECTION_REGISTRY,
    SOURCE_CLASS_REGISTRY,
    SourceClass,
)
from plato_rag.guardrails.source_access import visible_collection_names

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
async def sources(
    settings: Settings = Depends(get_settings),
) -> SourcesResponse:
    result: list[SourceClassResponse] = []
    visible = set(visible_collection_names(settings))

    for sc, info in SOURCE_CLASS_REGISTRY.items():
        collections = [
            CollectionInfo(
                name=col.name,
                label=col.label,
                parser_type=col.parser_type,
                chunker_type=col.chunker_type,
            )
            for col in COLLECTION_REGISTRY.values()
            if col.source_class == sc and col.name in visible
        ]
        if not collections:
            continue
        result.append(
            SourceClassResponse(
                source_class=sc,
                trust_tier=info.trust_tier,
                label=info.label,
                description=info.description,
                collections=collections,
            )
        )

    return SourcesResponse(source_classes=result)
