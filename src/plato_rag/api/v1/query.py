"""POST /v1/query — main query endpoint.

This is an early-stage endpoint. The retrieval quality depends entirely
on corpus coverage, and the citation extractor is regex-based. See README
for current implementation status.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException

from plato_rag.api.contracts import (
    ChatMode,
    ChunkMetadataResponse,
    CitationResponse,
    CompatSourceType,
    DebugResponse,
    GroundingResponse,
    LocationRefResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
    SourceCoverageResponse,
    compat_source_type_for,
)
from plato_rag.config import Settings
from plato_rag.dependencies import get_generation_service, get_retrieval_service, get_settings
from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import SourceClass, trust_tier_for
from plato_rag.generation.service import GenerationService
from plato_rag.retrieval.policy import PLATO_RETRIEVAL_POLICY
from plato_rag.retrieval.service import RetrievalService

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
) -> QueryResponse:
    if request.mode != ChatMode.PLATO:
        msg = f"Mode {request.mode.value} is not yet implemented."
        raise HTTPException(status_code=501, detail=msg)

    request_id = f"req_{uuid.uuid4().hex[:12]}"

    # Apply request options to policy
    policy = PLATO_RETRIEVAL_POLICY
    if request.options.max_chunks != policy.max_chunks:
        policy = replace(policy, max_chunks=request.options.max_chunks)

    # Retrieve
    retrieval_result = await retrieval_service.retrieve(
        query=request.question,
        policy=policy,
        source_filter=request.options.source_filter,
    )

    # Generate
    history = [(turn.role.value.lower(), turn.content) for turn in request.conversation_history]
    gen_result = await generation_service.generate(
        question=request.question,
        chunks=retrieval_result.chunks,
        conversation_history=history or None,
    )

    # Build response — mapping domain types to API contracts
    chunk_responses = [_chunk_to_response(sc) for sc in retrieval_result.chunks]

    citation_responses = []
    for cit in gen_result.citations:
        tier = trust_tier_for(cit.source_class) if cit.source_class else 0
        citation_responses.append(CitationResponse(
            work=cit.work,
            author=cit.author or "",
            location=cit.location,
            excerpt=cit.excerpt,
            source_type=(
                compat_source_type_for(cit.source_class)
                if cit.source_class
                else CompatSourceType.SECONDARY
            ),
            source_class=cit.source_class or SourceClass.PEER_REVIEWED,
            trust_tier=tier,
            access_url=cit.access_url,
        ))

    grounding = retrieval_result.grounding
    grounding_response = GroundingResponse(
        interpretation_level=grounding.interpretation_level,
        confidence_summary=grounding.confidence_summary,
        limitations=grounding.limitations,
        source_coverage=SourceCoverageResponse(
            counts_by_class=grounding.source_counts,
            total_chunks_searched=grounding.total_searched,
            total_chunks_returned=len(retrieval_result.chunks),
        ),
        grounding_notes=grounding.grounding_notes,
    )

    debug = None
    if request.options.include_debug:
        debug = DebugResponse(
            retrieval_policy_applied=f"PLATO_RETRIEVAL_POLICY (max_chunks={policy.max_chunks})",
            raw_similarity_scores=[sc.similarity_score for sc in retrieval_result.chunks],
            boosted_scores=[sc.effective_score for sc in retrieval_result.chunks],
            ungrounded_citations=gen_result.ungrounded_citations,
        )

    return QueryResponse(
        answer=gen_result.answer,
        retrieved_chunks=chunk_responses,
        citations=citation_responses,
        grounding=grounding_response,
        debug=debug,
        api_version=settings.api_version,
        request_id=request_id,
    )


def _chunk_to_response(sc: ScoredChunk) -> RetrievedChunkResponse:
    """Map a ScoredChunk to the API response model."""
    chunk = sc.chunk
    tier = trust_tier_for(chunk.source_class)

    loc_ref = None
    if chunk.location_ref:
        loc_ref = LocationRefResponse(
            system=chunk.location_ref.system.value,
            value=chunk.location_ref.value,
            range_end=chunk.location_ref.range_end,
        )

    return RetrievedChunkResponse(
        id=str(chunk.id),
        text=chunk.text,
        source_type=compat_source_type_for(chunk.source_class),
        source_class=chunk.source_class,
        trust_tier=tier,
        work=chunk.work_title,
        author=chunk.author,
        location=chunk.location_display,
        location_ref=loc_ref,
        collection=chunk.collection,
        chunk_metadata=ChunkMetadataResponse(
            speaker=chunk.speaker,
            interlocutor=chunk.interlocutor,
            section_title=chunk.section_title,
            context_type=chunk.context_type,
            chunk_index=chunk.chunk_index,
            token_count=chunk.token_count,
            entry_url=chunk.extra_metadata.get("entry_url") if chunk.extra_metadata else None,
            last_updated=chunk.extra_metadata.get("last_updated") if chunk.extra_metadata else None,
        ),
        similarity_score=sc.similarity_score,
    )
