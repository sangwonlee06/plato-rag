"""Source classification model.

Every document and chunk carries a source class that governs retrieval
priority, reranking behavior, grounding assessment, and citation formatting.

Source classes are ordered by epistemic role, not quality:
- A primary text IS the thing being studied
- A reference encyclopedia is a high-trust interpretation of it
- Peer-reviewed scholarship is a narrower scholarly contribution
- A curated bibliography is a discovery resource, not a ground-truth source

Trust tiers are ordinal (1 = most trusted) and derived from source class
via the registry. They are NOT stored on individual documents or chunks —
the registry is the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceClass(StrEnum):
    """Classification of a source by its epistemic role."""

    PRIMARY_TEXT = "primary_text"
    REFERENCE_ENCYCLOPEDIA = "reference_encyclopedia"
    PEER_REVIEWED = "peer_reviewed"
    CURATED_BIBLIOGRAPHY = "curated_bibliography"


class SourceExposure(StrEnum):
    """Whether a collection is deployable in the public service."""

    PUBLIC = "PUBLIC"
    LOCAL_ONLY = "LOCAL_ONLY"


# ---------------------------------------------------------------------------
# Source class registry — single source of truth for trust tiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceClassInfo:
    """Metadata about a source class."""

    source_class: SourceClass
    trust_tier: int
    label: str
    description: str


SOURCE_CLASS_REGISTRY: dict[SourceClass, SourceClassInfo] = {
    SourceClass.PRIMARY_TEXT: SourceClassInfo(
        source_class=SourceClass.PRIMARY_TEXT,
        trust_tier=1,
        label="Primary Philosophical Texts",
        description="Original philosophical works in translation",
    ),
    SourceClass.REFERENCE_ENCYCLOPEDIA: SourceClassInfo(
        source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
        trust_tier=2,
        label="Reference Encyclopedias",
        description="High-trust peer-reviewed encyclopedic reference (SEP, IEP)",
    ),
    SourceClass.PEER_REVIEWED: SourceClassInfo(
        source_class=SourceClass.PEER_REVIEWED,
        trust_tier=3,
        label="Peer-Reviewed Scholarship",
        description="Journal articles and scholarly monographs",
    ),
    SourceClass.CURATED_BIBLIOGRAPHY: SourceClassInfo(
        source_class=SourceClass.CURATED_BIBLIOGRAPHY,
        trust_tier=4,
        label="Curated Bibliographies",
        description="Bibliographic and discovery resources (PhilPapers, Oxford Bibliographies)",
    ),
}


# ---------------------------------------------------------------------------
# Collection registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceCollectionInfo:
    """Metadata about a named collection within a source class."""

    name: str
    source_class: SourceClass
    exposure: SourceExposure
    label: str
    parser_type: str
    chunker_type: str
    notes: str = ""


COLLECTION_REGISTRY: dict[str, SourceCollectionInfo] = {
    "platonic_dialogues": SourceCollectionInfo(
        name="platonic_dialogues",
        source_class=SourceClass.PRIMARY_TEXT,
        exposure=SourceExposure.PUBLIC,
        label="Platonic Dialogues",
        parser_type="plaintext",
        chunker_type="section",
        notes="Stephanus numbering must be preserved. Speaker attribution required. "
        "Supports prepared plaintext and Perseus TEI ingestion.",
    ),
    "presocratic_fragments": SourceCollectionInfo(
        name="presocratic_fragments",
        source_class=SourceClass.PRIMARY_TEXT,
        exposure=SourceExposure.PUBLIC,
        label="Presocratic Fragments",
        parser_type="plaintext",
        chunker_type="sliding_window",
        notes="DK numbering must be preserved.",
    ),
    "aristotle_corpus": SourceCollectionInfo(
        name="aristotle_corpus",
        source_class=SourceClass.PRIMARY_TEXT,
        exposure=SourceExposure.PUBLIC,
        label="Aristotle Corpus",
        parser_type="plaintext",
        chunker_type="section",
        notes="Bekker numbering must be preserved. Supports prepared plaintext "
        "and Perseus TEI ingestion.",
    ),
    # Future: descartes_meditations, hume_treatise, kant_critique, mill_liberty, etc.
    "sep": SourceCollectionInfo(
        name="sep",
        source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
        exposure=SourceExposure.LOCAL_ONLY,
        label="Stanford Encyclopedia of Philosophy",
        parser_type="html",
        chunker_type="section",
        notes="Section structure is authoritative. Preserve entry author, "
        "revision date, and stable URL. Respect SEP terms of use. "
        "Local-only: do not expose through public deployments.",
    ),
    "iep": SourceCollectionInfo(
        name="iep",
        source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
        exposure=SourceExposure.PUBLIC,
        label="Internet Encyclopedia of Philosophy",
        parser_type="html",
        chunker_type="section",
        notes="Preserve section headings and author attribution.",
    ),
}


# ---------------------------------------------------------------------------
# Lookup utilities
# ---------------------------------------------------------------------------


def trust_tier_for(source_class: SourceClass) -> int:
    """Return the trust tier (1 = most trusted) for a source class."""
    return SOURCE_CLASS_REGISTRY[source_class].trust_tier


def collection_source_class(collection_name: str) -> SourceClass:
    """Return the source class for a named collection."""
    info = COLLECTION_REGISTRY.get(collection_name)
    if info is None:
        raise ValueError(f"Unknown collection: {collection_name!r}")
    return info.source_class


def collection_exposure(collection_name: str) -> SourceExposure:
    """Return whether a collection is public or local-only."""
    info = COLLECTION_REGISTRY.get(collection_name)
    if info is None:
        raise ValueError(f"Unknown collection: {collection_name!r}")
    return info.exposure


def public_collection_names() -> set[str]:
    """Return all collections safe for public deployment."""
    return {
        name for name, info in COLLECTION_REGISTRY.items() if info.exposure == SourceExposure.PUBLIC
    }


def local_only_collection_names() -> set[str]:
    """Return collections that must remain local-only."""
    return {
        name
        for name, info in COLLECTION_REGISTRY.items()
        if info.exposure == SourceExposure.LOCAL_ONLY
    }


def is_local_only_collection(collection_name: str) -> bool:
    """Return True when a collection is restricted to local/internal use."""
    return collection_exposure(collection_name) == SourceExposure.LOCAL_ONLY


def is_high_trust(source_class: SourceClass) -> bool:
    """Return True if the source class is tier 1 or tier 2."""
    return trust_tier_for(source_class) <= 2
