"""Runtime guardrails for deployable vs local-only sources."""

from plato_rag.guardrails.source_access import (
    SourceAccessPolicyError,
    assert_citations_match_allowed_collections,
    assert_retrieved_chunks_match_allowed_collections,
    resolve_allowed_collections,
    validate_source_access_settings,
    visible_collection_names,
)

__all__ = [
    "SourceAccessPolicyError",
    "assert_citations_match_allowed_collections",
    "assert_retrieved_chunks_match_allowed_collections",
    "resolve_allowed_collections",
    "validate_source_access_settings",
    "visible_collection_names",
]
