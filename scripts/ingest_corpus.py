#!/usr/bin/env python3
"""Ingest a curated corpus manifest into the Plato RAG database.

Supports two workflows:

- `prepared_text`: local primary-text files already in `[SECTION]` format
- `sep_url`: fetch SEP HTML directly from the stable entry URL and parse it

Use `--dry-run` to validate parsing and chunking without touching the database
or calling the embedding API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.source import COLLECTION_REGISTRY, collection_source_class
from plato_rag.ingestion.chunkers.section import SectionChunker
from plato_rag.ingestion.parsers.plaintext import PlaintextParser
from plato_rag.ingestion.parsers.sep_html import SepHtmlParser
from plato_rag.protocols.ingestion import ChunkConfig, Chunker, Parser

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def _load_manifest(manifest_path: Path) -> list[CorpusEntry]:
    with manifest_path.open() as handle:
        data = json.load(handle)

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain an 'entries' list")

    manifest_entries: list[CorpusEntry] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each manifest entry must be a mapping")
        manifest_entries.append(CorpusEntry(
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
        ))
    return manifest_entries


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


def _parser_for(collection: str) -> Parser:
    parser_type = COLLECTION_REGISTRY[collection].parser_type
    if parser_type == "plaintext":
        return PlaintextParser()
    if collection == "sep":
        return SepHtmlParser()
    raise ValueError(f"Unsupported parser for collection {collection!r}")


def _chunker_for(collection: str) -> Chunker:
    chunker_type = COLLECTION_REGISTRY[collection].chunker_type
    if chunker_type == "section":
        return SectionChunker()
    raise ValueError(f"Unsupported chunker for collection {collection!r}")


def _metadata_for(entry: CorpusEntry) -> DocumentMetadata:
    return DocumentMetadata(
        id=uuid.uuid4(),
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


def _load_raw_content(entry: CorpusEntry, manifest_dir: Path) -> str:
    if entry.kind == "prepared_text":
        if entry.input_path is None:
            raise ValueError(f"Entry {entry.id!r} is missing input_path")
        path = manifest_dir / entry.input_path
        return path.read_text()

    if entry.kind == "sep_url":
        if entry.source_url is None:
            raise ValueError(f"Entry {entry.id!r} is missing source_url")
        request = urllib.request.Request(
            entry.source_url,
            headers={"User-Agent": "plato-rag-ingestion/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset)
        except urllib.error.URLError:
            logger.warning(
                "urllib could not fetch %s; falling back to curl",
                entry.source_url,
            )
            return subprocess.check_output(
                ["curl", "-L", "-s", "--max-time", "30", entry.source_url],
                text=True,
            )

    if entry.kind == "sep_html_file":
        if entry.input_path is None:
            raise ValueError(f"Entry {entry.id!r} is missing input_path")
        path = manifest_dir / entry.input_path
        return path.read_text()

    raise ValueError(f"Unsupported manifest entry kind {entry.kind!r}")


def _select_entries(entries: list[CorpusEntry], selected_ids: set[str] | None) -> list[CorpusEntry]:
    if selected_ids is None:
        return entries
    return [entry for entry in entries if entry.id in selected_ids]


def _dry_run_entry(
    entry: CorpusEntry,
    raw_content: str,
    parser: Parser,
    chunker: Chunker,
    chunk_config: ChunkConfig,
) -> None:
    metadata = _metadata_for(entry)
    parsed = parser.parse(raw_content, metadata)
    chunks = chunker.chunk(parsed, chunk_config)
    logger.info(
        "Validated %s (%s): %d sections, %d chunks, parser=%s",
        entry.id,
        metadata.title,
        len(parsed.sections),
        len(chunks),
        parser.parser_version(),
    )


async def _ingest_entries(
    entries: list[CorpusEntry],
    manifest_dir: Path,
    chunk_config: ChunkConfig,
) -> None:
    from plato_rag.config import Settings
    from plato_rag.db.engine import create_engine, create_session_factory, dispose_engine
    from plato_rag.ingestion.embedders.openai import OpenAIEmbedder
    from plato_rag.ingestion.service import IngestionService

    settings = Settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    try:
        async with session_factory() as session:
            for entry in entries:
                raw_content = _load_raw_content(entry, manifest_dir)
                parser = _parser_for(entry.collection)
                service = IngestionService(
                    session=session,
                    parser=parser,
                    chunker=_chunker_for(entry.collection),
                    embedder=embedder,
                )
                result = await service.ingest(raw_content, _metadata_for(entry), chunk_config)
                if result.skipped:
                    logger.info("Skipped %s: %s", entry.id, result.skip_reason)
                else:
                    logger.info(
                        "Ingested %s (%s): %d chunks",
                        entry.id,
                        entry.title,
                        result.chunk_count,
                    )
    finally:
        await dispose_engine(engine)


def _validate_manifest_entries(entries: list[CorpusEntry]) -> None:
    for entry in entries:
        if entry.collection not in COLLECTION_REGISTRY:
            raise ValueError(f"Unknown collection {entry.collection!r} in entry {entry.id!r}")
        if entry.kind == "prepared_text" and entry.input_path is None:
            raise ValueError(f"Prepared text entry {entry.id!r} is missing input_path")
        if entry.kind == "sep_url" and entry.source_url is None:
            raise ValueError(f"SEP URL entry {entry.id!r} is missing source_url")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the curated Plato corpus manifest")
    parser.add_argument("--manifest", default="data/corpus_seed.json")
    parser.add_argument("--only", nargs="+", default=None, help="Entry ids to process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chunk-tokens", type=int, default=512)
    parser.add_argument("--min-chunk-tokens", type=int, default=50)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    entries = _load_manifest(manifest_path)
    _validate_manifest_entries(entries)

    selected_ids = set(args.only) if args.only else None
    selected_entries = _select_entries(entries, selected_ids)
    if not selected_entries:
        logger.warning("No manifest entries selected")
        return

    chunk_config = ChunkConfig(
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
    )

    if args.dry_run:
        for entry in selected_entries:
            raw_content = _load_raw_content(entry, manifest_dir)
            _dry_run_entry(
                entry=entry,
                raw_content=raw_content,
                parser=_parser_for(entry.collection),
                chunker=_chunker_for(entry.collection),
                chunk_config=chunk_config,
            )
        return

    await _ingest_entries(selected_entries, manifest_dir, chunk_config)


if __name__ == "__main__":
    asyncio.run(main())
