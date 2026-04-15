"""Chunk repository — CRUD and vector search for chunks."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from plato_rag.db.models import ChunkModel
from plato_rag.domain.chunk import ChunkData, ScoredChunk
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import SourceClass


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_create(self, chunks: list[ChunkData], embeddings: list[list[float]]) -> int:
        models = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            m = ChunkModel(
                id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding=embedding,
                source_class=chunk.source_class.value,
                collection=chunk.collection,
                work_title=chunk.work_title,
                author=chunk.author,
                location_system=chunk.location_ref.system.value if chunk.location_ref else None,
                location_value=chunk.location_ref.value if chunk.location_ref else None,
                location_range_end=chunk.location_ref.range_end if chunk.location_ref else None,
                section_title=chunk.section_title,
                speaker=chunk.speaker,
                interlocutor=chunk.interlocutor,
                context_type=chunk.context_type,
                extra_metadata=chunk.extra_metadata,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                overlap_tokens=chunk.overlap_tokens,
                embedding_model=chunk.embedding_model,
            )
            models.append(m)
        self._session.add_all(models)
        await self._session.flush()
        return len(models)

    async def vector_search(
        self,
        query_vector: list[float],
        source_classes: list[SourceClass] | None = None,
        collections: list[str] | None = None,
        limit: int = 10,
    ) -> list[ScoredChunk]:
        """Similarity search via pgvector cosine distance."""
        # Build distance expression: 1 - cosine_distance = cosine_similarity
        distance = ChunkModel.embedding.cosine_distance(query_vector)
        similarity = (1 - distance).label("similarity")

        stmt = select(ChunkModel, similarity).order_by(distance).limit(limit)

        if source_classes:
            stmt = stmt.where(
                ChunkModel.source_class.in_([sc.value for sc in source_classes])
            )
        if collections:
            stmt = stmt.where(ChunkModel.collection.in_(collections))

        result = await self._session.execute(stmt)
        scored = []
        for row in result.all():
            model = row[0]
            sim = float(row[1])
            chunk = self._to_domain(model)
            scored.append(ScoredChunk(chunk=chunk, similarity_score=sim))
        return scored

    async def count_total(self) -> int:
        result = await self._session.execute(select(func.count(ChunkModel.id)))
        return result.scalar_one()

    async def count_by_source_class(self) -> dict[str, int]:
        result = await self._session.execute(
            select(ChunkModel.source_class, func.count(ChunkModel.id)).group_by(
                ChunkModel.source_class
            )
        )
        return {row[0]: row[1] for row in result.all()}

    @staticmethod
    def _to_domain(model: ChunkModel) -> ChunkData:
        location_ref = None
        if model.location_system and model.location_value:
            location_ref = LocationRef(
                system=LocationSystem(model.location_system),
                value=model.location_value,
                range_end=model.location_range_end,
            )

        return ChunkData(
            id=model.id,
            document_id=model.document_id,
            text=model.text,
            source_class=SourceClass(model.source_class),
            collection=model.collection,
            work_title=model.work_title,
            author=model.author,
            location_ref=location_ref,
            section_title=model.section_title,
            speaker=model.speaker,
            interlocutor=model.interlocutor,
            context_type=model.context_type,
            extra_metadata=model.extra_metadata,
            chunk_index=model.chunk_index,
            token_count=model.token_count,
            overlap_tokens=model.overlap_tokens,
            embedding_model=model.embedding_model,
            created_at=model.created_at,
        )
