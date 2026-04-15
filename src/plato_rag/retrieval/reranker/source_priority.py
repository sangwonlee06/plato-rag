"""Source-priority reranker.

Multiplies chunk similarity scores by a trust-tier factor from the
RetrievalPolicy. Primary texts get 1.3x, reference encyclopedias get
1.15x, peer-reviewed stays at 1.0x, bibliographies get 0.9x. This is
a simple heuristic, not a learned model.
"""

from __future__ import annotations

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import trust_tier_for
from plato_rag.retrieval.policy import RetrievalPolicy


class SourcePriorityReranker:
    def rerank(
        self,
        chunks: list[ScoredChunk],
        query: str,
        policy: RetrievalPolicy,
    ) -> list[ScoredChunk]:
        reranked = []
        for sc in chunks:
            tier = trust_tier_for(sc.chunk.source_class)
            boost = policy.boost_for_tier(tier)
            reranked.append(ScoredChunk(
                chunk=sc.chunk,
                similarity_score=sc.similarity_score,
                boosted_score=sc.similarity_score * boost,
            ))
        reranked.sort(key=lambda s: s.effective_score, reverse=True)
        return reranked
