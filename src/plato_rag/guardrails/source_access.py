"""Source-access guardrails for public vs local-only collections."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import (
    COLLECTION_REGISTRY,
    is_local_only_collection,
    local_only_collection_names,
    public_collection_names,
)
from plato_rag.protocols.generation import ExtractedCitation

if TYPE_CHECKING:
    from plato_rag.config import Settings

logger = logging.getLogger(__name__)


class SourceAccessPolicyError(ValueError):
    """Raised when a request or deployment violates source access policy."""


def validate_source_access_settings(settings: Settings) -> None:
    """Fail fast on restricted-source deployment mistakes."""
    public = set(_configured_public_collections(settings))
    local_only = set(_configured_local_only_collections(settings))
    configured = public | local_only

    unknown = configured - set(COLLECTION_REGISTRY)
    if unknown:
        _fail_or_warn(
            settings,
            "Unknown source collections in configuration: " + ", ".join(sorted(unknown)),
        )

    restricted_public = {name for name in public if is_local_only_collection(name)}
    if restricted_public:
        _fail_or_warn(
            settings,
            "Public allowed collections include local-only sources: "
            + ", ".join(sorted(restricted_public)),
        )

    non_local_only = {name for name in local_only if not is_local_only_collection(name)}
    if non_local_only:
        _fail_or_warn(
            settings,
            "Local-only allowed collections include public-safe sources: "
            + ", ".join(sorted(non_local_only)),
        )

    if settings.deployment_scope == "public" and settings.enable_local_only_sep:
        _fail_or_warn(
            settings,
            "PLATO_RAG_ENABLE_LOCAL_ONLY_SEP cannot be true in public deployments.",
        )

    if (
        settings.enable_local_only_sep
        and settings.deployment_scope in {"local", "internal"}
        and not settings.local_only_manifest_path.exists()
    ):
        _fail_or_warn(
            settings,
            "Local-only SEP is enabled but the local-only manifest path does not exist: "
            f"{settings.local_only_manifest_path}",
        )


def visible_collection_names(settings: Settings) -> list[str]:
    """Collections that should be visible to the running deployment."""
    return resolve_allowed_collections(settings, None)


def resolve_allowed_collections(
    settings: Settings,
    requested_collections: list[str] | None,
) -> list[str]:
    """Resolve the final collection allowlist for a request."""
    allowed = set(_configured_public_collections(settings))

    if settings.deployment_scope in {"local", "internal"} and settings.enable_local_only_sep:
        allowed.update(_configured_local_only_collections(settings))

    requested = set(requested_collections or allowed)
    unknown = requested - set(COLLECTION_REGISTRY)
    if unknown:
        raise SourceAccessPolicyError(
            "Unknown collections requested: " + ", ".join(sorted(unknown)),
        )

    disallowed = requested - allowed
    if disallowed:
        raise SourceAccessPolicyError(
            "Requested collections are not available in this deployment: "
            + ", ".join(sorted(disallowed)),
        )

    return sorted(requested)


def assert_retrieved_chunks_match_allowed_collections(
    chunks: list[ScoredChunk],
    allowed_collections: Iterable[str],
) -> None:
    """Detect any retrieval result that escaped the configured allowlist."""
    allowed = set(allowed_collections)
    leaked = {sc.chunk.collection for sc in chunks if sc.chunk.collection not in allowed}
    if leaked:
        raise SourceAccessPolicyError(
            "Retrieved chunks from disallowed collections: " + ", ".join(sorted(leaked)),
        )


def assert_citations_match_allowed_collections(
    citations: list[ExtractedCitation],
    allowed_collections: Iterable[str],
) -> None:
    """Detect any citation that references a disallowed collection."""
    allowed = set(allowed_collections)
    leaked = {
        citation.collection
        for citation in citations
        if citation.collection is not None and citation.collection not in allowed
    }
    if leaked:
        raise SourceAccessPolicyError(
            "Generated citations reference disallowed collections: " + ", ".join(sorted(leaked)),
        )


def _configured_public_collections(settings: Settings) -> list[str]:
    configured = _parse_collection_csv(settings.public_allowed_collections)
    if configured:
        return configured
    return sorted(public_collection_names())


def _configured_local_only_collections(settings: Settings) -> list[str]:
    configured = _parse_collection_csv(settings.local_only_allowed_collections)
    if configured:
        return configured
    return sorted(local_only_collection_names())


def _parse_collection_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _fail_or_warn(settings: Settings, message: str) -> None:
    if settings.fail_start_on_restricted_config:
        raise SourceAccessPolicyError(message)
    logger.warning(message)
