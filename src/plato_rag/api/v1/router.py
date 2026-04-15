"""v1 API router — aggregates all v1 endpoints."""

from fastapi import APIRouter

from plato_rag.api.v1.health import router as health_router
from plato_rag.api.v1.query import router as query_router
from plato_rag.api.v1.sources import router as sources_router

v1_router = APIRouter()
v1_router.include_router(health_router, tags=["health"])
v1_router.include_router(query_router, tags=["query"])
v1_router.include_router(sources_router, tags=["sources"])
