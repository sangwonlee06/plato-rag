"""Curated corpus manifest loading and startup bootstrap."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from plato_rag.db.repositories.chunk import ChunkRepository
from plato_rag.db.repositories.document import DocumentRepository
from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.source import (
    COLLECTION_REGISTRY,
    collection_source_class,
    is_local_only_collection,
)
from plato_rag.ingestion.chunkers.section import SectionChunker
from plato_rag.ingestion.parsers.plaintext import PlaintextParser
from plato_rag.ingestion.service import IngestionService
from plato_rag.protocols.embedding import Embedder
from plato_rag.protocols.ingestion import ChunkConfig, Chunker, Parser

logger = logging.getLogger(__name__)

FILE_BACKED_ENTRY_KINDS = {"prepared_text", "sep_html_file", "iep_html_file"}
URL_BACKED_ENTRY_KINDS = {"sep_url", "iep_url"}


class CorpusBootstrapError(RuntimeError):
    """Raised when the seed corpus cannot be bootstrapped safely."""


@dataclass(frozen=True)
class CorpusEntry:
    id: str
    kind: str
    collection: str
    title: str
    author: str
    tradition: str | None = None
    period: str | None = None
    topics: list[str] = field(default_factory=list)
    translation: str | None = None
    edition: str | None = None
    source_url: str | None = None
    input_path: str | None = None


@dataclass(frozen=True)
class CorpusBootstrapResult:
    status: Literal["noop", "bootstrapped"]
    manifest_entries: int
    existing_entries: int
    attempted_entries: int
    ingested_entries: int
    linked_entries: int
    total_documents_before: int
    total_documents_after: int
    total_chunks_before: int
    total_chunks_after: int


def load_manifest(manifest_path: Path) -> list[CorpusEntry]:
    with manifest_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain an 'entries' list")

    manifest_entries: list[CorpusEntry] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each manifest entry must be a mapping")
        manifest_entries.append(
            CorpusEntry(
                id=str(raw_entry["id"]),
                kind=str(raw_entry["kind"]),
                collection=str(raw_entry["collection"]),
                title=str(raw_entry["title"]),
                author=str(raw_entry["author"]),
                tradition=_optional_str(raw_entry.get("tradition")),
                period=_optional_str(raw_entry.get("period")),
                topics=_string_list(raw_entry.get("topics", [])),
                translation=_optional_str(raw_entry.get("translation")),
                edition=_optional_str(raw_entry.get("edition")),
                source_url=_optional_str(raw_entry.get("source_url")),
                input_path=_optional_str(raw_entry.get("input_path")),
            )
        )
    return manifest_entries


def validate_manifest_entries(
    entries: list[CorpusEntry],
    *,
    allow_local_only_collections: bool = False,
) -> None:
    seen_ids: set[str] = set()
    for entry in entries:
        if entry.id in seen_ids:
            raise ValueError(f"Duplicate manifest entry id {entry.id!r}")
        seen_ids.add(entry.id)

        if entry.collection not in COLLECTION_REGISTRY:
            raise ValueError(f"Unknown collection {entry.collection!r} in entry {entry.id!r}")
        if is_local_only_collection(entry.collection) and not allow_local_only_collections:
            raise ValueError(
                f"Collection {entry.collection!r} in entry {entry.id!r} is local-only "
                "and cannot be bootstrapped in public-safe mode"
            )
        if entry.kind in FILE_BACKED_ENTRY_KINDS and entry.input_path is None:
            raise ValueError(f"Entry {entry.id!r} is missing input_path")
        if entry.kind in URL_BACKED_ENTRY_KINDS and entry.source_url is None:
            raise ValueError(f"Entry {entry.id!r} is missing source_url")
        if entry.kind not in FILE_BACKED_ENTRY_KINDS | URL_BACKED_ENTRY_KINDS:
            raise ValueError(f"Unsupported manifest entry kind {entry.kind!r}")


def select_entries(entries: list[CorpusEntry], selected_ids: set[str] | None) -> list[CorpusEntry]:
    if selected_ids is None:
        return entries
    return [entry for entry in entries if entry.id in selected_ids]


def parser_for(collection: str) -> Parser:
    parser_type = COLLECTION_REGISTRY[collection].parser_type
    if parser_type == "plaintext":
        return PlaintextParser()
    if parser_type == "html" and collection == "sep":
        try:
            from plato_rag.local_only.sep_html import SepHtmlParser
        except ModuleNotFoundError as exc:
            raise CorpusBootstrapError(
                "SEP parser support is unavailable in this build. Local-only SEP "
                "components must remain excluded from public deployments."
            ) from exc

        return SepHtmlParser()
    if parser_type == "html" and collection == "iep":
        from plato_rag.ingestion.parsers.iep_html import IepHtmlParser

        return IepHtmlParser()
    raise ValueError(f"Unsupported parser for collection {collection!r}")


def chunker_for(collection: str) -> Chunker:
    chunker_type = COLLECTION_REGISTRY[collection].chunker_type
    if chunker_type == "section":
        return SectionChunker()
    raise ValueError(f"Unsupported chunker for collection {collection!r}")


def metadata_for(entry: CorpusEntry) -> DocumentMetadata:
    return DocumentMetadata(
        id=uuid.uuid4(),
        corpus_entry_id=entry.id,
        title=entry.title,
        author=entry.author,
        source_class=collection_source_class(entry.collection),
        collection=entry.collection,
        tradition=entry.tradition,
        period=entry.period,
        topics=entry.topics,
        translation=entry.translation,
        edition=entry.edition,
        source_url=entry.source_url,
    )


async def load_raw_content(
    entry: CorpusEntry,
    manifest_dir: Path,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> str:
    if entry.kind in FILE_BACKED_ENTRY_KINDS:
        if entry.input_path is None:
            raise ValueError(f"Entry {entry.id!r} is missing input_path")
        path = manifest_dir / entry.input_path
        return path.read_text(encoding="utf-8")

    if entry.kind in URL_BACKED_ENTRY_KINDS:
        if entry.source_url is None:
            raise ValueError(f"Entry {entry.id!r} is missing source_url")
        if http_client is None:
            raise ValueError("http_client is required for URL-backed manifest entries")
        response = await http_client.get(entry.source_url)
        response.raise_for_status()
        return response.text

    raise ValueError(f"Unsupported manifest entry kind {entry.kind!r}")


async def ensure_seed_corpus(
    session: AsyncSession,
    *,
    embedder: Embedder,
    manifest_path: Path,
    allow_local_only_collections: bool = False,
    chunk_config: ChunkConfig,
    advisory_lock_id: int,
    http_timeout_seconds: float,
) -> CorpusBootstrapResult:
    manifest_path = manifest_path.resolve()
    entries = load_manifest(manifest_path)
    validate_manifest_entries(
        entries,
        allow_local_only_collections=allow_local_only_collections,
    )
    manifest_dir = manifest_path.parent

    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)

    async with session.begin():
        await session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": advisory_lock_id},
        )

        existing_entry_ids = await doc_repo.list_corpus_entry_ids()
        total_documents_before = await doc_repo.count_total()
        total_chunks_before = await chunk_repo.count_total()
        pending_entries = [entry for entry in entries if entry.id not in existing_entry_ids]

        if not pending_entries:
            return CorpusBootstrapResult(
                status="noop",
                manifest_entries=len(entries),
                existing_entries=len(existing_entry_ids),
                attempted_entries=0,
                ingested_entries=0,
                linked_entries=0,
                total_documents_before=total_documents_before,
                total_documents_after=total_documents_before,
                total_chunks_before=total_chunks_before,
                total_chunks_after=total_chunks_before,
            )

        ingested_entries = 0
        linked_entries = 0
        timeout = httpx.Timeout(http_timeout_seconds)
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "plato-rag-bootstrap/1.0"},
        ) as http_client:
            for entry in pending_entries:
                raw_content = await load_raw_content(entry, manifest_dir, http_client=http_client)
                service = IngestionService(
                    session=session,
                    parser=parser_for(entry.collection),
                    chunker=chunker_for(entry.collection),
                    embedder=embedder,
                )
                result = await service.ingest(
                    raw_content,
                    metadata_for(entry),
                    chunk_config,
                    commit=False,
                )
                if result.skipped:
                    if result.skip_reason == "Document already ingested (same hash)":
                        await doc_repo.assign_corpus_entry_id(result.document_id, entry.id)
                        linked_entries += 1
                        continue
                    raise CorpusBootstrapError(
                        f"Bootstrap entry {entry.id!r} could not be ingested: {result.skip_reason}"
                    )
                ingested_entries += 1
                logger.info(
                    "Bootstrapped corpus entry %s (%s) with %d chunks",
                    entry.id,
                    entry.title,
                    result.chunk_count,
                )

        total_documents_after = await doc_repo.count_total()
        total_chunks_after = await chunk_repo.count_total()

    return CorpusBootstrapResult(
        status="bootstrapped",
        manifest_entries=len(entries),
        existing_entries=len(existing_entry_ids),
        attempted_entries=len(pending_entries),
        ingested_entries=ingested_entries,
        linked_entries=linked_entries,
        total_documents_before=total_documents_before,
        total_documents_after=total_documents_after,
        total_chunks_before=total_chunks_before,
        total_chunks_after=total_chunks_after,
    )


def dry_run_entry(
    entry: CorpusEntry,
    raw_content: str,
    parser: Parser,
    chunker: Chunker,
    chunk_config: ChunkConfig,
) -> tuple[int, int, str]:
    metadata = metadata_for(entry)
    parsed = parser.parse(raw_content, metadata)
    chunks = chunker.chunk(parsed, chunk_config)
    return len(parsed.sections), len(chunks), parser.parser_version()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string value, got {type(value).__name__}")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Expected a list of strings")
        values.append(item)
    return values
