"""Document repository — CRUD operations for documents."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from plato_rag.db.models import DocumentModel
from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.source import SourceClass


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, meta: DocumentMetadata) -> uuid.UUID:
        model = DocumentModel(
            id=meta.id,
            corpus_entry_id=meta.corpus_entry_id,
            title=meta.title,
            author=meta.author,
            source_class=meta.source_class.value,
            collection=meta.collection,
            tradition=meta.tradition,
            period=meta.period,
            topics=meta.topics or None,
            translation=meta.translation,
            edition=meta.edition,
            source_url=meta.source_url,
            last_verified_at=meta.last_verified_at,
            ingested_at=datetime.now(UTC),
            parser_version=meta.parser_version,
            raw_hash=meta.raw_hash,
        )
        self._session.add(model)
        await self._session.flush()
        return model.id

    async def count_total(self) -> int:
        result = await self._session.execute(select(func.count(DocumentModel.id)))
        return result.scalar_one()

    async def list_corpus_entry_ids(self) -> set[str]:
        result = await self._session.execute(
            select(DocumentModel.corpus_entry_id).where(DocumentModel.corpus_entry_id.is_not(None))
        )
        return {row[0] for row in result.all() if row[0] is not None}

    async def assign_corpus_entry_id(self, document_id: uuid.UUID, corpus_entry_id: str) -> None:
        await self._session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(corpus_entry_id=corpus_entry_id)
        )
        await self._session.flush()

    async def get_by_hash(self, raw_hash: str) -> DocumentMetadata | None:
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.raw_hash == raw_hash)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_domain(model)

    async def count_by_collection(self) -> dict[str, int]:
        result = await self._session.execute(
            select(DocumentModel.collection, DocumentModel.source_class)
        )
        counts: dict[str, int] = {}
        for row in result.all():
            col = row[0]
            counts[col] = counts.get(col, 0) + 1
        return counts

    @staticmethod
    def _to_domain(model: DocumentModel) -> DocumentMetadata:
        return DocumentMetadata(
            id=model.id,
            corpus_entry_id=model.corpus_entry_id,
            title=model.title,
            author=model.author,
            source_class=SourceClass(model.source_class),
            collection=model.collection,
            tradition=model.tradition,
            period=model.period,
            topics=model.topics or [],
            translation=model.translation,
            edition=model.edition,
            source_url=model.source_url,
            last_verified_at=model.last_verified_at,
            ingested_at=model.ingested_at,
            parser_version=model.parser_version,
            raw_hash=model.raw_hash,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
