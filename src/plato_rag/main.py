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
from plato_rag.ingestion.embedders.openai import OpenAIEmbedder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    engine = create_engine(settings.database_url)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)

    # Shared service instances — created once, not per-request
    app.state.embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    llm = AnthropicLLM(
        api_key=settings.anthropic_api_key,
        model=settings.generation_model,
        max_tokens=settings.generation_max_tokens,
    )
    app.state.generation_service = GenerationService(llm=llm)

    yield

    await dispose_engine(engine)


app = FastAPI(
    title="Plato RAG Service",
    description="Philosophy RAG with source-priority retrieval. Early-stage — see README for status.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router, prefix="/v1")
