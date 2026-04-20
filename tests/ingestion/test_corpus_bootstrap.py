"""Tests for corpus bootstrap orchestration."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from plato_rag.ingestion.corpus import CorpusBootstrapError, ensure_seed_corpus
from plato_rag.ingestion.service import IngestResult
from plato_rag.protocols.ingestion import ChunkConfig


@dataclass
class _BootstrapState:
    existing_entry_ids: set[str] = field(default_factory=set)
    total_documents: int = 0
    total_chunks: int = 0
    existing_document_id: uuid.UUID = field(default_factory=uuid.uuid4)
    ingested_entry_ids: list[str] = field(default_factory=list)
    linked_assignments: list[tuple[uuid.UUID, str]] = field(default_factory=list)


class _FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakeSession:
    def __init__(self, state: _BootstrapState) -> None:
        self.state = state
        self.executed: list[dict[str, object]] = []

    def begin(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, statement: object, params: dict[str, object] | None = None) -> None:
        self.executed.append({"statement": statement, "params": params or {}})


class _FakeDocumentRepository:
    def __init__(self, session: _FakeSession) -> None:
        self._state = session.state

    async def list_corpus_entry_ids(self) -> set[str]:
        return set(self._state.existing_entry_ids)

    async def count_total(self) -> int:
        return self._state.total_documents

    async def assign_corpus_entry_id(self, document_id: uuid.UUID, corpus_entry_id: str) -> None:
        self._state.existing_entry_ids.add(corpus_entry_id)
        self._state.linked_assignments.append((document_id, corpus_entry_id))


class _FakeChunkRepository:
    def __init__(self, session: _FakeSession) -> None:
        self._state = session.state

    async def count_total(self) -> int:
        return self._state.total_chunks


class _FakeIngestionService:
    def __init__(self, session: _FakeSession, **_: object) -> None:
        self._state = session.state

    async def ingest(
        self,
        raw_content: str,
        metadata: Any,
        chunk_config: ChunkConfig,
        *,
        commit: bool = True,
    ) -> IngestResult:
        del chunk_config, commit
        if raw_content == "already-there":
            return IngestResult(
                document_id=self._state.existing_document_id,
                chunk_count=0,
                skipped=True,
                skip_reason="Document already ingested (same hash)",
            )
        if raw_content == "too-short":
            return IngestResult(
                document_id=metadata.id,
                chunk_count=0,
                skipped=True,
                skip_reason="No chunks produced (content too short?)",
            )

        self._state.total_documents += 1
        self._state.total_chunks += 3
        if metadata.corpus_entry_id is not None:
            self._state.existing_entry_ids.add(metadata.corpus_entry_id)
            self._state.ingested_entry_ids.append(metadata.corpus_entry_id)
        return IngestResult(document_id=metadata.id, chunk_count=3)


class _FakeHttpResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeHttpClient:
    def __init__(self, text: str) -> None:
        self._text = text

    async def __aenter__(self) -> _FakeHttpClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def get(self, url: str) -> _FakeHttpResponse:
        del url
        return _FakeHttpResponse(self._text)


@pytest.fixture
def fake_bootstrap_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("plato_rag.ingestion.corpus.DocumentRepository", _FakeDocumentRepository)
    monkeypatch.setattr("plato_rag.ingestion.corpus.ChunkRepository", _FakeChunkRepository)
    monkeypatch.setattr("plato_rag.ingestion.corpus.IngestionService", _FakeIngestionService)


async def _run_bootstrap(
    manifest_path: Path,
    state: _BootstrapState,
    *,
    allow_local_only_collections: bool = False,
) -> object:
    session = _FakeSession(state)
    return await ensure_seed_corpus(
        session,
        embedder=object(),
        manifest_path=manifest_path,
        allow_local_only_collections=allow_local_only_collections,
        chunk_config=ChunkConfig(),
        advisory_lock_id=42,
        http_timeout_seconds=1.0,
    )


def _write_manifest(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    manifest_path = tmp_path / "corpus_seed.json"
    manifest_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return manifest_path


def _prepared_entry(entry_id: str, relative_path: str) -> dict[str, object]:
    return {
        "id": entry_id,
        "kind": "prepared_text",
        "collection": "platonic_dialogues",
        "title": entry_id.title(),
        "author": "Plato",
        "input_path": relative_path,
    }


@pytest.mark.asyncio
async def test_bootstrap_ingests_only_missing_entries(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
) -> None:
    (tmp_path / "existing.txt").write_text("existing", encoding="utf-8")
    (tmp_path / "missing.txt").write_text("new-document", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            _prepared_entry("existing", "existing.txt"),
            _prepared_entry("missing", "missing.txt"),
        ],
    )
    state = _BootstrapState(existing_entry_ids={"existing"}, total_documents=1, total_chunks=4)

    result = await _run_bootstrap(manifest_path, state)

    assert result.status == "bootstrapped"
    assert result.existing_entries == 1
    assert result.attempted_entries == 1
    assert result.ingested_entries == 1
    assert result.linked_entries == 0
    assert result.total_documents_before == 1
    assert result.total_documents_after == 2
    assert result.total_chunks_before == 4
    assert result.total_chunks_after == 7
    assert state.ingested_entry_ids == ["missing"]


@pytest.mark.asyncio
async def test_bootstrap_links_existing_documents_by_hash(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
) -> None:
    (tmp_path / "existing.txt").write_text("already-there", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, [_prepared_entry("existing", "existing.txt")])
    state = _BootstrapState(total_documents=1, total_chunks=4)

    result = await _run_bootstrap(manifest_path, state)

    assert result.status == "bootstrapped"
    assert result.attempted_entries == 1
    assert result.ingested_entries == 0
    assert result.linked_entries == 1
    assert state.linked_assignments == [(state.existing_document_id, "existing")]


@pytest.mark.asyncio
async def test_bootstrap_fails_when_manifest_entry_cannot_produce_chunks(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
) -> None:
    (tmp_path / "broken.txt").write_text("too-short", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, [_prepared_entry("broken", "broken.txt")])

    with pytest.raises(CorpusBootstrapError, match="could not be ingested"):
        await _run_bootstrap(manifest_path, _BootstrapState())


@pytest.mark.asyncio
async def test_bootstrap_rejects_local_only_collections_by_default(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "sep-entry",
                "kind": "sep_html_file",
                "collection": "sep",
                "title": "SEP Entry",
                "author": "Example Author",
                "input_path": "sep.html",
            },
        ],
    )
    (tmp_path / "sep.html").write_text("<html></html>", encoding="utf-8")

    with pytest.raises(ValueError, match="local-only"):
        await _run_bootstrap(manifest_path, _BootstrapState())


@pytest.mark.asyncio
async def test_bootstrap_allows_local_only_collections_when_explicitly_enabled(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "sep-entry",
                "kind": "sep_html_file",
                "collection": "sep",
                "title": "SEP Entry",
                "author": "Example Author",
                "input_path": "sep.html",
            },
        ],
    )
    (tmp_path / "sep.html").write_text("local-only sep content", encoding="utf-8")

    result = await _run_bootstrap(
        manifest_path,
        _BootstrapState(),
        allow_local_only_collections=True,
    )

    assert result.status == "bootstrapped"
    assert result.ingested_entries == 1


@pytest.mark.asyncio
async def test_bootstrap_supports_public_iep_url_entries(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "plato-iep",
                "kind": "iep_url",
                "collection": "iep",
                "title": "Plato",
                "author": "Example Author",
                "source_url": "https://iep.utm.edu/plato/",
            },
        ],
    )
    monkeypatch.setattr(
        "plato_rag.ingestion.corpus.httpx.AsyncClient",
        lambda **_: _FakeHttpClient("public iep html"),
    )

    result = await _run_bootstrap(manifest_path, _BootstrapState())

    assert result.status == "bootstrapped"
    assert result.ingested_entries == 1


@pytest.mark.asyncio
async def test_bootstrap_supports_public_perseus_tei_entries(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "protagoras",
                "kind": "perseus_tei",
                "collection": "platonic_dialogues",
                "title": "Protagoras",
                "author": "Plato",
                "source_url": "https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0178",
                "source_config": {"text_id": "Prot."},
            },
        ],
    )
    monkeypatch.setattr(
        "plato_rag.ingestion.corpus.httpx.AsyncClient",
        lambda **_: _FakeHttpClient("perseus tei xml"),
    )

    result = await _run_bootstrap(manifest_path, _BootstrapState())

    assert result.status == "bootstrapped"
    assert result.ingested_entries == 1


@pytest.mark.asyncio
async def test_bootstrap_rejects_perseus_entries_without_text_identifier(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "gorgias",
                "kind": "perseus_tei",
                "collection": "platonic_dialogues",
                "title": "Gorgias",
                "author": "Plato",
                "source_url": "https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0178",
            },
        ],
    )

    with pytest.raises(ValueError, match="source_config.text_id"):
        await _run_bootstrap(manifest_path, _BootstrapState())


@pytest.mark.asyncio
async def test_bootstrap_allows_aristotle_perseus_entries_without_text_identifier(
    tmp_path: Path,
    fake_bootstrap_dependencies: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "nicomachean-ethics",
                "kind": "perseus_tei",
                "collection": "aristotle_corpus",
                "title": "Nicomachean Ethics",
                "author": "Aristotle",
                "source_url": "https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0054",
            },
        ],
    )
    monkeypatch.setattr(
        "plato_rag.ingestion.corpus.httpx.AsyncClient",
        lambda **_: _FakeHttpClient("aristotle perseus tei xml"),
    )

    result = await _run_bootstrap(manifest_path, _BootstrapState())

    assert result.status == "bootstrapped"
    assert result.ingested_entries == 1
