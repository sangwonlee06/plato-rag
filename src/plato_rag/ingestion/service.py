"""Ingestion service — orchestrates parse, chunk, embed, store."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from plato_rag.db.repositories.chunk import ChunkRepository
from plato_rag.db.repositories.document import DocumentRepository
from plato_rag.domain.chunk import ChunkData
from plato_rag.domain.document import DocumentMetadata
from plato_rag.protocols.embedding import Embedder
from plato_rag.protocols.ingestion import ChunkConfig, Chunker, Parser

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: uuid.UUID
    chunk_count: int
    skipped: bool = False
    skip_reason: str | None = None


class IngestionService:
    """Accepts any Parser, Chunker, and embedder that match the protocols."""

    def __init__(
        self,
        session: AsyncSession,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
    ) -> None:
        self._session = session
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._doc_repo = DocumentRepository(session)
        self._chunk_repo = ChunkRepository(session)

    async def ingest(
        self,
        raw_content: str,
        metadata: DocumentMetadata,
        chunk_config: ChunkConfig,
        *,
        commit: bool = True,
    ) -> IngestResult:
        # Dedup check
        content_hash = hashlib.sha256(raw_content.encode()).hexdigest()
        existing = await self._doc_repo.get_by_hash(content_hash)
        if existing is not None:
            return IngestResult(
                document_id=existing.id,
                chunk_count=0,
                skipped=True,
                skip_reason="Document already ingested (same hash)",
            )

        metadata.raw_hash = content_hash
        metadata.parser_version = self._parser.parser_version()

        parsed = self._parser.parse(raw_content, metadata)
        logger.info("Parsed %d sections from '%s'", len(parsed.sections), metadata.title)

        raw_chunks = self._chunker.chunk(parsed, chunk_config)
        logger.info("Produced %d chunks from '%s'", len(raw_chunks), metadata.title)

        if not raw_chunks:
            return IngestResult(
                document_id=metadata.id,
                chunk_count=0,
                skipped=True,
                skip_reason="No chunks produced (content too short?)",
            )

        await self._doc_repo.create(metadata)

        chunks: list[ChunkData] = []
        for rc in raw_chunks:
            chunks.append(
                ChunkData(
                    id=uuid.uuid4(),
                    document_id=metadata.id,
                    text=rc.text,
                    source_class=metadata.source_class,
                    collection=metadata.collection,
                    work_title=metadata.title,
                    author=metadata.author,
                    location_ref=rc.location_ref,
                    section_title=rc.section_title,
                    speaker=rc.speaker,
                    interlocutor=rc.interlocutor,
                    context_type=rc.context_type,
                    extra_metadata=_merged_chunk_metadata(metadata, rc.extra_metadata),
                    chunk_index=rc.chunk_index,
                    token_count=rc.token_count,
                    overlap_tokens=rc.overlap_tokens,
                    embedding_model=self._embedder.model_name(),
                )
            )

        texts = [c.text for c in chunks]
        embeddings = await self._embedder.embed(texts)

        count = await self._chunk_repo.bulk_create(chunks, embeddings)
        if commit:
            await self._session.commit()

        return IngestResult(document_id=metadata.id, chunk_count=count)


def _merged_chunk_metadata(
    metadata: DocumentMetadata,
    raw_chunk_metadata: dict[str, str] | None,
) -> dict[str, object] | None:
    merged: dict[str, object] = dict(raw_chunk_metadata or {})

    if metadata.tradition:
        merged.setdefault("tradition", metadata.tradition)
    if metadata.period:
        merged.setdefault("period", metadata.period)
    if metadata.topics:
        merged.setdefault("topics", list(metadata.topics))
    if metadata.translation:
        merged.setdefault("translation", metadata.translation)
    if metadata.edition:
        merged.setdefault("edition", metadata.edition)
    if metadata.source_url:
        merged.setdefault("source_url", metadata.source_url)

    return merged or None
