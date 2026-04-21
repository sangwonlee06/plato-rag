"""Query request and response contracts for POST /v1/query."""

from __future__ import annotations

from pydantic import BaseModel, Field

from plato_rag.api.contracts.chunks import RetrievedChunkResponse
from plato_rag.api.contracts.citations import CitationResponse
from plato_rag.api.contracts.common import ChatMode, ConversationTurn, SourceClass
from plato_rag.api.contracts.grounding import GroundingResponse


class QueryOptions(BaseModel):
    max_chunks: int = Field(default=5, ge=1, le=20)
    source_filter: list[SourceClass] | None = None
    allowed_collections: list[str] | None = None
    include_debug: bool = False


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    mode: ChatMode = Field(...)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    options: QueryOptions = Field(default_factory=QueryOptions)


class DebugResponse(BaseModel):
    """Diagnostic info — unstable, do not build production logic against this."""

    retrieval_policy_applied: str | None = None
    raw_similarity_scores: list[float] = []
    boosted_scores: list[float] = []
    generation_prompt_preview: str | None = None
    ungrounded_citations: list[str] = []
    unsupported_claims: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
    citations: list[CitationResponse]
    grounding: GroundingResponse
    debug: DebugResponse | None = None
    api_version: str = "1.0.0"
    request_id: str
