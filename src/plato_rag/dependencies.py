"""FastAPI dependency injection."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from plato_rag.config import Settings
from plato_rag.generation.service import GenerationService
from plato_rag.retrieval.service import RetrievalService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


async def get_session(request: Request) -> AsyncGenerator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


async def get_retrieval_service(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> RetrievalService:
    """Uses the shared embedder from app.state; only the DB session is per-request."""
    from plato_rag.retrieval.vector_store.pgvector import PgVectorStore

    store = PgVectorStore(session)
    return RetrievalService(vector_store=store, embedder=request.app.state.embedder)


async def get_generation_service(request: Request) -> GenerationService:
    """Uses the shared LLM from app.state."""
    return request.app.state.generation_service  # type: ignore[no-any-return]
