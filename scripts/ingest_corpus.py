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
import logging
from pathlib import Path

import httpx

from plato_rag.ingestion.corpus import (
    CorpusEntry,
    chunker_for,
    dry_run_entry,
    load_manifest,
    load_raw_content,
    metadata_for,
    parser_for,
    select_entries,
    validate_manifest_entries,
)
from plato_rag.protocols.ingestion import ChunkConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    timeout = httpx.Timeout(settings.bootstrap_http_timeout_seconds)

    try:
        async with session_factory() as session, httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "plato-rag-cli-ingestion/1.0"},
        ) as http_client:
            for entry in entries:
                raw_content = await load_raw_content(entry, manifest_dir, http_client=http_client)
                service = IngestionService(
                    session=session,
                    parser=parser_for(entry.collection),
                    chunker=chunker_for(entry.collection),
                    embedder=embedder,
                )
                result = await service.ingest(raw_content, metadata_for(entry), chunk_config)
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
    entries = load_manifest(manifest_path)
    validate_manifest_entries(entries)

    selected_ids = set(args.only) if args.only else None
    selected_entries = select_entries(entries, selected_ids)
    if not selected_entries:
        logger.warning("No manifest entries selected")
        return

    chunk_config = ChunkConfig(
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
    )

    if args.dry_run:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": "plato-rag-cli-ingestion/1.0"},
        ) as http_client:
            for entry in selected_entries:
                raw_content = await load_raw_content(entry, manifest_dir, http_client=http_client)
                section_count, chunk_count, parser_version = dry_run_entry(
                    entry=entry,
                    raw_content=raw_content,
                    parser=parser_for(entry.collection),
                    chunker=chunker_for(entry.collection),
                    chunk_config=chunk_config,
                )
                logger.info(
                    "Validated %s (%s): %d sections, %d chunks, parser=%s",
                    entry.id,
                    entry.title,
                    section_count,
                    chunk_count,
                    parser_version,
                )
        return

    await _ingest_entries(selected_entries, manifest_dir, chunk_config)


if __name__ == "__main__":
    asyncio.run(main())
