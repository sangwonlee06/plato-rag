"""Retrieval service — staged search, reranking, grounding assessment.

NOTE: This is an early-stage implementation. The staged retrieval runs
searches per source class in priority order, applies a simple score
multiplier per trust tier, and enforces minimum quotas on a best-effort
basis (if no chunks from a required class pass the similarity threshold,
the quota is silently unmet). This is a reasonable starting point, not
a sophisticated retrieval system.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from plato_rag.api.contracts.common import InterpretationLevel
from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import SourceClass, is_high_trust, trust_tier_for
from plato_rag.ingestion.embedders.openai import OpenAIEmbedder
from plato_rag.protocols.retrieval import SearchFilters
from plato_rag.retrieval.policy import RetrievalPolicy
from plato_rag.retrieval.reranker.source_priority import SourcePriorityReranker
from plato_rag.retrieval.vector_store.pgvector import PgVectorStore

logger = logging.getLogger(__name__)


@dataclass
class GroundingAssessment:
    interpretation_level: InterpretationLevel
    confidence_summary: str
    limitations: str | None
    source_counts: dict[SourceClass, int]
    grounding_notes: list[str]
    total_searched: int


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    grounding: GroundingAssessment


class RetrievalService:
    def __init__(
        self,
        vector_store: PgVectorStore,
        embedder: OpenAIEmbedder,
        reranker: SourcePriorityReranker | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._reranker = reranker or SourcePriorityReranker()

    async def retrieve(
        self,
        query: str,
        policy: RetrievalPolicy,
        source_filter: list[SourceClass] | None = None,
    ) -> RetrievalResult:
        vectors = await self._embedder.embed([query])
        query_vector = vectors[0]

        # Staged retrieval: search each stage separately so high-priority
        # classes get dedicated search budget. Early exit when we have
        # enough above-threshold results from high-trust sources.
        all_candidates: list[ScoredChunk] = []
        results_returned = 0

        for stage in policy.search_stages:
            stage_classes = list(stage.source_classes)
            if source_filter:
                stage_classes = [sc for sc in stage_classes if sc in source_filter]
                if not stage_classes:
                    continue

            results = await self._store.search(
                query_vector=query_vector,
                filters=SearchFilters(source_classes=stage_classes),
                limit=stage.max_candidates,
            )
            above = [r for r in results if r.similarity_score >= policy.similarity_threshold]
            all_candidates.extend(above)
            results_returned += len(results)

            # Early exit: if we already have enough high-trust chunks,
            # skip lower-priority stages
            high_trust_count = sum(
                1 for c in all_candidates if is_high_trust(c.chunk.source_class)
            )
            if high_trust_count >= policy.max_chunks:
                break

        # Rerank with trust-tier boosts
        reranked = self._reranker.rerank(all_candidates, query, policy)

        # Enforce source quotas (best-effort: only from what passed threshold)
        selected = self._enforce_quotas(reranked, policy)

        grounding = self._assess_grounding(selected, policy, results_returned)
        return RetrievalResult(chunks=selected, grounding=grounding)

    def _enforce_quotas(
        self, ranked: list[ScoredChunk], policy: RetrievalPolicy
    ) -> list[ScoredChunk]:
        """Best-effort quota enforcement. If a required source class has no
        candidates above the similarity threshold, its quota is unmet."""
        selected: list[ScoredChunk] = []

        # First: fill minimum quotas from each required class
        for quota in policy.source_quotas:
            class_chunks = [sc for sc in ranked if sc.chunk.source_class == quota.source_class]
            for sc in class_chunks[: quota.min_chunks]:
                if sc not in selected:
                    selected.append(sc)

        # Then: fill remaining slots by score
        for sc in ranked:
            if len(selected) >= policy.max_chunks:
                break
            if sc not in selected:
                selected.append(sc)

        return selected[: policy.max_chunks]

    def _assess_grounding(
        self,
        chunks: list[ScoredChunk],
        policy: RetrievalPolicy,
        total_searched: int,
    ) -> GroundingAssessment:
        """Simple rule-based grounding assessment. Not a quality guarantee."""
        counts: dict[SourceClass, int] = Counter()
        for sc in chunks:
            counts[sc.chunk.source_class] += 1

        primary_count = counts.get(SourceClass.PRIMARY_TEXT, 0)
        high_trust_count = sum(c for cls, c in counts.items() if is_high_trust(cls))
        best_score = max((sc.effective_score for sc in chunks), default=0.0)
        notes: list[str] = []
        rules = policy.grounding

        if not chunks or best_score < rules.low_confidence_below_similarity:
            level = InterpretationLevel.LOW_CONFIDENCE
            summary = "Insufficient source material retrieved."
            limitations = "The corpus may not cover this topic, or the question may not match well."
        elif primary_count < rules.direct_min_primary or high_trust_count < rules.direct_min_high_trust:
            level = InterpretationLevel.INTERPRETIVE
            summary = "Answer relies on interpretation beyond direct primary source support."
            limitations = None
            if primary_count == 0:
                limitations = "No primary source text was retrieved."
                notes.append("Answer based on secondary/reference sources only.")
        else:
            level = InterpretationLevel.DIRECT
            summary = "Answer grounded in primary source text with reference support."
            limitations = None

        return GroundingAssessment(
            interpretation_level=level,
            confidence_summary=summary,
            limitations=limitations,
            source_counts=dict(counts),
            grounding_notes=notes,
            total_searched=total_searched,
        )
