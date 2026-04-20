"""Domain models for the Plato RAG service."""

from plato_rag.domain.chunk import ChunkData, ScoredChunk
from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import (
    COLLECTION_REGISTRY,
    SOURCE_CLASS_REGISTRY,
    SourceClass,
    SourceClassInfo,
    SourceCollectionInfo,
    SourceExposure,
    collection_exposure,
    collection_source_class,
    is_high_trust,
    is_local_only_collection,
    local_only_collection_names,
    public_collection_names,
    trust_tier_for,
)

__all__ = [
    "COLLECTION_REGISTRY",
    "ChunkData",
    "DocumentMetadata",
    "LocationRef",
    "LocationSystem",
    "SOURCE_CLASS_REGISTRY",
    "ScoredChunk",
    "SourceExposure",
    "SourceClass",
    "SourceClassInfo",
    "SourceCollectionInfo",
    "collection_exposure",
    "collection_source_class",
    "is_high_trust",
    "is_local_only_collection",
    "local_only_collection_names",
    "public_collection_names",
    "trust_tier_for",
]
