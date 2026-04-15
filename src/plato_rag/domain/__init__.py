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
    collection_source_class,
    is_high_trust,
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
    "SourceClass",
    "SourceClassInfo",
    "SourceCollectionInfo",
    "collection_source_class",
    "is_high_trust",
    "trust_tier_for",
]
