"""GET /v1/health — service health and corpus status."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from plato_rag.config import Settings
from plato_rag.db.repositories.chunk import ChunkRepository
from plato_rag.dependencies import get_session, get_settings

router = APIRouter()


class CorpusStatus(BaseModel):
    total_chunks: int
    chunks_by_source_class: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    version: str
    corpus: CorpusStatus


@router.get("/health", response_model=HealthResponse)
async def health(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    repo = ChunkRepository(session)
    total = await repo.count_total()
    by_class = await repo.count_by_source_class()

    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        corpus=CorpusStatus(
            total_chunks=total,
            chunks_by_source_class=by_class,
        ),
    )
