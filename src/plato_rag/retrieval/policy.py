"""Retrieval policy — declarative source-priority rules.

The RetrievalPolicy is a frozen dataclass that the retrieval service
reads to decide how to search, rank, and select chunks. It is
inspectable, testable, and serializable.

The current implementation is straightforward:
- search_stages: search source classes in priority order
- tier_boosts: multiply scores by a constant per trust tier
- source_quotas: guarantee minimum representation (best-effort)
- grounding rules: simple threshold checks for interpretation level
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plato_rag.domain.source import SourceClass


@dataclass(frozen=True)
class TierBoost:
    """Score multiplier per trust tier. > 1.0 promotes, < 1.0 demotes."""
    trust_tier: int
    boost_factor: float


@dataclass(frozen=True)
class SourceQuota:
    """Best-effort minimum/maximum chunk count per source class.

    If the corpus has no matching chunks above the similarity threshold,
    the min_chunks guarantee is silently unmet.
    """
    source_class: SourceClass
    min_chunks: int
    max_chunks: int | None = None


@dataclass(frozen=True)
class SearchStage:
    """One stage in the multi-stage retrieval strategy.

    Stages are executed in order. The retrieval service may skip later
    stages if earlier stages produced enough high-trust results.
    """
    source_classes: tuple[SourceClass, ...]
    max_candidates: int


@dataclass(frozen=True)
class GroundingRule:
    """Thresholds for interpretation level assessment."""
    direct_min_primary: int = 1
    direct_min_high_trust: int = 2
    flag_low_trust_only: bool = True
    low_confidence_below_chunks: int = 1
    low_confidence_below_similarity: float = 0.35


@dataclass(frozen=True)
class RetrievalPolicy:
    """Declarative policy governing source-priority retrieval."""

    max_chunks: int = 5
    similarity_threshold: float = 0.30
    source_quotas: tuple[SourceQuota, ...] = ()
    tier_boosts: tuple[TierBoost, ...] = ()
    search_stages: tuple[SearchStage, ...] = ()
    grounding: GroundingRule = field(default_factory=GroundingRule)

    def boost_for_tier(self, trust_tier: int) -> float:
        for tb in self.tier_boosts:
            if tb.trust_tier == trust_tier:
                return tb.boost_factor
        return 1.0

    def quota_for(self, source_class: SourceClass) -> SourceQuota | None:
        for sq in self.source_quotas:
            if sq.source_class == source_class:
                return sq
        return None


PLATO_RETRIEVAL_POLICY = RetrievalPolicy(
    max_chunks=5,
    similarity_threshold=0.30,
    source_quotas=(
        SourceQuota(source_class=SourceClass.PRIMARY_TEXT, min_chunks=1),
        SourceQuota(source_class=SourceClass.REFERENCE_ENCYCLOPEDIA, min_chunks=1),
    ),
    tier_boosts=(
        TierBoost(trust_tier=1, boost_factor=1.30),
        TierBoost(trust_tier=2, boost_factor=1.15),
        TierBoost(trust_tier=3, boost_factor=1.00),
        TierBoost(trust_tier=4, boost_factor=0.90),
    ),
    search_stages=(
        SearchStage(source_classes=(SourceClass.PRIMARY_TEXT,), max_candidates=10),
        SearchStage(source_classes=(SourceClass.REFERENCE_ENCYCLOPEDIA,), max_candidates=10),
        SearchStage(source_classes=(SourceClass.PEER_REVIEWED,), max_candidates=5),
    ),
    grounding=GroundingRule(
        direct_min_primary=1,
        direct_min_high_trust=2,
        flag_low_trust_only=True,
        low_confidence_below_chunks=1,
        low_confidence_below_similarity=0.35,
    ),
)
