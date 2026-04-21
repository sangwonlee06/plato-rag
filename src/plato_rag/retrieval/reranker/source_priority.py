"""Source-priority reranker.

Multiplies chunk similarity scores by a trust-tier factor from the
RetrievalPolicy. Primary texts get 1.3x, reference encyclopedias get
1.15x, peer-reviewed stays at 1.0x, bibliographies get 0.9x. This is
a simple heuristic, not a learned model.
"""

from __future__ import annotations

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.philosophy_profile import (
    PhilosophyProfile,
    is_explicit_ancient_query,
    profile_chunk,
    profile_text,
    significant_tokens,
)
from plato_rag.domain.source import trust_tier_for
from plato_rag.retrieval.policy import RetrievalPolicy


class SourcePriorityReranker:
    def rerank(
        self,
        chunks: list[ScoredChunk],
        query: str,
        policy: RetrievalPolicy,
    ) -> list[ScoredChunk]:
        query_profile = profile_text(query)
        query_tokens = significant_tokens(query)
        reranked = []
        for sc in chunks:
            tier = trust_tier_for(sc.chunk.source_class)
            boost = policy.boost_for_tier(tier) * self._discipline_multiplier(
                sc,
                query_tokens=query_tokens,
                query_profile=query_profile,
            )
            reranked.append(
                ScoredChunk(
                    chunk=sc.chunk,
                    similarity_score=sc.similarity_score,
                    boosted_score=sc.similarity_score * boost,
                )
            )
        reranked.sort(key=lambda s: s.effective_score, reverse=True)
        return reranked

    def _discipline_multiplier(
        self,
        scored_chunk: ScoredChunk,
        *,
        query_tokens: set[str],
        query_profile: PhilosophyProfile,
    ) -> float:
        chunk = scored_chunk.chunk
        chunk_profile = profile_chunk(chunk)
        chunk_tokens = significant_tokens(
            " ".join(
                part
                for part in (
                    chunk.work_title,
                    chunk.section_title or "",
                    chunk.author,
                    chunk.text,
                )
                if part
            )
        )

        multiplier = 1.0
        lexical_overlap = len(query_tokens & chunk_tokens)
        multiplier += min(lexical_overlap * 0.02, 0.12)

        multiplier += 0.06 * len(query_profile.topics & chunk_profile.topics)
        multiplier += 0.04 * len(query_profile.traditions & chunk_profile.traditions)
        multiplier += 0.03 * len(query_profile.periods & chunk_profile.periods)

        if (
            chunk.collection == "platonic_dialogues"
            and "ancient" in chunk_profile.traditions
            and not is_explicit_ancient_query(query_profile)
            and (
                (query_profile.topics and not (query_profile.topics & chunk_profile.topics))
                or _is_general_philosophy_query(query_tokens)
            )
        ):
            multiplier -= 0.18

        if (
            chunk.collection == "aristotle_corpus"
            and "ancient" in chunk_profile.traditions
            and not is_explicit_ancient_query(query_profile)
            and (
                (query_profile.topics and not (query_profile.topics & chunk_profile.topics))
                or _is_general_philosophy_query(query_tokens)
            )
        ):
            multiplier -= 0.12

        if (
            query_profile.traditions
            and chunk_profile.traditions
            and not (query_profile.traditions & chunk_profile.traditions)
        ):
            multiplier -= 0.08

        if (
            query_profile.periods
            and chunk_profile.periods
            and not (query_profile.periods & chunk_profile.periods)
        ):
            multiplier -= 0.05

        return max(0.7, min(multiplier, 1.45))


def _is_general_philosophy_query(query_tokens: set[str]) -> bool:
    return bool(
        query_tokens
        & {
            "philosophy",
            "ethics",
            "metaphysics",
            "epistemology",
            "mind",
            "language",
            "logic",
            "consciousness",
            "knowledge",
        }
    )
