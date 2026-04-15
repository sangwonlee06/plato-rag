"""Retrieval protocols: vector store and reranker.

These define the contracts for the two retrieval-stage components:
- VectorStore: similarity search with metadata filtering
- Reranker: score adjustment based on source priority or models

The retrieval service orchestrates these components according to
the RetrievalPolicy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import SourceClass
from plato_rag.retrieval.policy import RetrievalPolicy

# ---------------------------------------------------------------------------
# Vector store types
# ---------------------------------------------------------------------------


@dataclass
class SearchFilters:
    """Metadata filters applied during vector search.

    These are pushed down into the database query so that
    irrelevant chunks are excluded before scoring, not after.
    """

    source_classes: list[SourceClass] | None = None
    collections: list[str] | None = None
    trust_tier_max: int | None = None
    work_titles: list[str] | None = None
    exclude_chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    """Similarity search over embedded chunks with metadata filtering.

    The primary implementation is PgVectorStore (PostgreSQL + pgvector).
    The protocol abstracts this so the retrieval service doesn't depend
    on the database implementation.

    search() returns ScoredChunk with similarity_score populated.
    The boosted_score field is left to the reranker.
    """

    async def search(
        self,
        query_vector: list[float],
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> list[ScoredChunk]:
        """Search for chunks similar to the query vector.

        Args:
            query_vector: The embedded query.
            filters: Optional metadata filters.
            limit: Maximum number of results.

        Returns:
            Chunks ordered by descending similarity score.
        """
        ...


class Reranker(Protocol):
    """Adjusts chunk scores based on source priority or model signals.

    The reranker sits between the vector store and the final selection.
    It reads the RetrievalPolicy to determine boost factors and
    applies them to the similarity scores.

    Implementations:
    - SourcePriorityReranker: applies tier_boosts from the policy
    - CrossEncoderReranker (future): uses a cross-encoder model to
      re-score query-chunk pairs
    """

    def rerank(
        self,
        chunks: list[ScoredChunk],
        query: str,
        policy: RetrievalPolicy,
    ) -> list[ScoredChunk]:
        """Rerank chunks by adjusting scores and re-sorting.

        The reranker MUST set boosted_score on each ScoredChunk.
        It MUST NOT mutate the input list — return a new sorted list.

        Args:
            chunks: Chunks from vector search, with similarity_score set.
            query: The original query text (needed for model-based reranking).
            policy: The retrieval policy governing boost factors.

        Returns:
            Chunks sorted by boosted_score descending.
        """
        ...
