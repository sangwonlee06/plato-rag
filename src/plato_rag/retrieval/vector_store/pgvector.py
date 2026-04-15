"""pgvector search. Thin wrapper — the real query logic is in ChunkRepository."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from plato_rag.db.repositories.chunk import ChunkRepository
from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import SourceClass
from plato_rag.protocols.retrieval import SearchFilters


class PgVectorStore:
    """Delegates to ChunkRepository. Exists so the retrieval service
    doesn't import the repository layer directly."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = ChunkRepository(session)

    async def search(
        self,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> list[ScoredChunk]:
        return await self._repo.vector_search(
            query_vector=query_vector,
            source_classes=filters.source_classes if filters else None,
            collections=filters.collections if filters else None,
            limit=limit,
        )
