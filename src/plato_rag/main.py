"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from plato_rag.api.v1.router import v1_router
from plato_rag.config import Settings
from plato_rag.db.engine import create_engine, create_session_factory, dispose_engine
from plato_rag.generation.llm.anthropic import AnthropicLLM
from plato_rag.generation.service import GenerationService
from plato_rag.ingestion.corpus import ensure_seed_corpus
from plato_rag.ingestion.embedders.openai import OpenAIEmbedder
from plato_rag.protocols.ingestion import ChunkConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    engine = create_engine(settings.database_url)
    try:
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)

        # Shared service instances — created once, not per-request
        app.state.embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        if settings.bootstrap_enabled:
            async with app.state.session_factory() as session:
                bootstrap_result = await ensure_seed_corpus(
                    session,
                    embedder=app.state.embedder,
                    manifest_path=settings.bootstrap_manifest_path,
                    chunk_config=ChunkConfig(
                        max_chunk_tokens=settings.bootstrap_max_chunk_tokens,
                        min_chunk_tokens=settings.bootstrap_min_chunk_tokens,
                        overlap_tokens=settings.bootstrap_overlap_tokens,
                    ),
                    advisory_lock_id=settings.bootstrap_lock_id,
                    http_timeout_seconds=settings.bootstrap_http_timeout_seconds,
                )
            app.state.bootstrap_result = bootstrap_result
            logger.info(
                "Corpus bootstrap %s: existing=%d attempted=%d ingested=%d linked=%d chunks=%d->%d",
                bootstrap_result.status,
                bootstrap_result.existing_entries,
                bootstrap_result.attempted_entries,
                bootstrap_result.ingested_entries,
                bootstrap_result.linked_entries,
                bootstrap_result.total_chunks_before,
                bootstrap_result.total_chunks_after,
            )
        llm = AnthropicLLM(
            api_key=settings.anthropic_api_key,
            model=settings.generation_model,
            max_tokens=settings.generation_max_tokens,
        )
        app.state.generation_service = GenerationService(llm=llm)

        yield
    finally:
        await dispose_engine(engine)


app = FastAPI(
    title="Plato RAG Service",
    description=(
        "Philosophy RAG with source-priority retrieval. "
        "Early-stage — see README for status."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router, prefix="/v1")
