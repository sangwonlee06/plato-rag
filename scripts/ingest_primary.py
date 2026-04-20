#!/usr/bin/env python3
"""Ingest a prepared primary text file into the Plato RAG corpus.

Requires the package to be installed (pip install -e ".[dev]").

Usage:
    python scripts/ingest_primary.py \
        --file data/prepared/primary/meno.txt \
        --title "Meno" \
        --author "Plato" \
        --collection platonic_dialogues \
        [--translation "G.M.A. Grube"]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid

from plato_rag.config import Settings
from plato_rag.db.engine import create_engine, create_session_factory, dispose_engine
from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.source import collection_source_class
from plato_rag.ingestion.chunkers.section import SectionChunker
from plato_rag.ingestion.embedders.openai import OpenAIEmbedder
from plato_rag.ingestion.parsers.plaintext import PlaintextParser
from plato_rag.ingestion.service import IngestionService
from plato_rag.protocols.ingestion import ChunkConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a primary text file")
    parser.add_argument("--file", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--translation", default=None)
    parser.add_argument("--tradition", default="ancient")
    parser.add_argument("--period", default="classical_greek")
    args = parser.parse_args()

    with open(args.file) as f:
        raw_content = f.read()

    settings = Settings()
    source_class = collection_source_class(args.collection)

    metadata = DocumentMetadata(
        id=uuid.uuid4(),
        title=args.title,
        author=args.author,
        source_class=source_class,
        collection=args.collection,
        tradition=args.tradition,
        period=args.period,
        translation=args.translation,
    )

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    async with session_factory() as session:
        service = IngestionService(
            session=session,
            parser=PlaintextParser(),
            chunker=SectionChunker(),
            embedder=embedder,
        )
        result = await service.ingest(raw_content, metadata, ChunkConfig())

    await dispose_engine(engine)

    if result.skipped:
        logger.info("Skipped: %s", result.skip_reason)
    else:
        logger.info(
            "Ingested '%s': %d chunks (document %s)",
            args.title,
            result.chunk_count,
            result.document_id,
        )


if __name__ == "__main__":
    asyncio.run(main())
