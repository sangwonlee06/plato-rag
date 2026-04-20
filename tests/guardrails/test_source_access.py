"""Tests for deployment-time and request-time source access guardrails."""

from pathlib import Path

import pytest

from plato_rag.config import Settings
from plato_rag.guardrails.source_access import (
    SourceAccessPolicyError,
    resolve_allowed_collections,
    validate_source_access_settings,
    visible_collection_names,
)


def _settings(**overrides: object) -> Settings:
    defaults = {
        "database_url": "postgresql+asyncpg://test:test@localhost:5432/plato_rag",
        "database_url_sync": "postgresql://test:test@localhost:5432/plato_rag",
        "openai_api_key": "test-openai",
        "anthropic_api_key": "test-anthropic",
        "bootstrap_manifest_path": Path("data/corpus_seed.json"),
        "local_only_manifest_path": Path("local_only/sep/corpus_seed.local.json"),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_validate_settings_rejects_sep_in_public_allowlist() -> None:
    settings = _settings(
        deployment_scope="public",
        public_allowed_collections="platonic_dialogues,sep",
    )

    with pytest.raises(SourceAccessPolicyError, match="local-only sources"):
        validate_source_access_settings(settings)


def test_validate_settings_rejects_sep_enablement_in_public_scope() -> None:
    settings = _settings(
        deployment_scope="public",
        enable_local_only_sep=True,
    )

    with pytest.raises(SourceAccessPolicyError, match="cannot be true in public"):
        validate_source_access_settings(settings)


def test_validate_settings_rejects_public_collection_in_local_only_allowlist() -> None:
    settings = _settings(
        deployment_scope="local",
        enable_local_only_sep=True,
        local_only_allowed_collections="sep,iep",
    )

    with pytest.raises(SourceAccessPolicyError, match="Local-only allowed collections"):
        validate_source_access_settings(settings)


def test_resolve_allowed_collections_defaults_to_public_safe_set() -> None:
    settings = _settings(
        deployment_scope="public",
        public_allowed_collections="platonic_dialogues,iep",
    )

    assert resolve_allowed_collections(settings, None) == ["iep", "platonic_dialogues"]
    assert visible_collection_names(settings) == ["iep", "platonic_dialogues"]


def test_resolve_allowed_collections_allows_sep_only_in_local_mode() -> None:
    settings = _settings(
        deployment_scope="local",
        enable_local_only_sep=True,
        public_allowed_collections="platonic_dialogues",
        local_only_allowed_collections="sep",
    )

    assert resolve_allowed_collections(settings, ["sep"]) == ["sep"]


def test_resolve_allowed_collections_rejects_sep_when_not_enabled() -> None:
    settings = _settings(
        deployment_scope="internal",
        enable_local_only_sep=False,
        public_allowed_collections="platonic_dialogues",
    )

    with pytest.raises(SourceAccessPolicyError, match="not available"):
        resolve_allowed_collections(settings, ["sep"])
