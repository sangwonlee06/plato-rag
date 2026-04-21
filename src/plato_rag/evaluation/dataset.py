"""Evaluation dataset models for retrieval and grounding checks."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from plato_rag.api.contracts.common import ChatMode, ConversationTurn, InterpretationLevel
from plato_rag.api.contracts.query import QueryOptions
from plato_rag.domain.source import SourceClass


class CitationExpectation(BaseModel):
    """A minimal citation expectation for one grounded source."""

    work: str | None = None
    author: str | None = None
    collection: str | None = None
    location: str | None = None
    source_class: SourceClass | None = None

    @model_validator(mode="after")
    def _require_match_anchor(self) -> CitationExpectation:
        if (
            self.work is None
            and self.author is None
            and self.collection is None
            and self.location is None
            and self.source_class is None
        ):
            msg = "Citation expectation must constrain at least one field"
            raise ValueError(msg)
        return self


class GenerationFixture(BaseModel):
    """Deterministic generation output used for evaluation fixtures."""

    raw_output: str

    @model_validator(mode="after")
    def _require_raw_output(self) -> GenerationFixture:
        if not self.raw_output.strip():
            msg = "Generation fixture raw_output must be non-empty"
            raise ValueError(msg)
        return self


class EvaluationExpectations(BaseModel):
    """Checks that a response must satisfy for a case to pass."""

    answer_must_contain: list[str] = Field(default_factory=list)
    answer_must_not_contain: list[str] = Field(default_factory=list)
    required_retrieved_works: list[str] = Field(default_factory=list)
    required_retrieved_works_any_of: list[str] = Field(default_factory=list)
    forbidden_retrieved_works: list[str] = Field(default_factory=list)
    required_retrieved_collections: list[str] = Field(default_factory=list)
    forbidden_retrieved_collections: list[str] = Field(default_factory=list)
    required_citations: list[CitationExpectation] = Field(default_factory=list)
    required_citations_any_of: list[CitationExpectation] = Field(default_factory=list)
    min_citations: int = Field(default=1, ge=0)
    max_ungrounded_citations: int = Field(default=0, ge=0)
    max_unsupported_claims: int = Field(default=0, ge=0)
    allowed_interpretation_levels: list[InterpretationLevel] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    """One evaluation prompt and its expected grounding behavior."""

    id: str
    question: str
    mode: ChatMode = ChatMode.PLATO
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    options: QueryOptions = Field(default_factory=QueryOptions)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    generation_fixture: GenerationFixture | None = None
    expectations: EvaluationExpectations


class EvaluationDataset(BaseModel):
    """A curated set of evaluation cases."""

    version: str
    name: str
    description: str | None = None
    cases: list[EvaluationCase]

    @model_validator(mode="after")
    def _require_unique_ids(self) -> EvaluationDataset:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            msg = "Evaluation case ids must be unique"
            raise ValueError(msg)
        return self


class EvaluationCaseResult(BaseModel):
    """Outcome for one evaluation case."""

    case_id: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    citation_count: int = 0
    retrieved_chunk_count: int = 0
    ungrounded_citation_count: int = 0
    unsupported_claim_count: int = 0


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate an evaluation dataset from YAML."""

    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return EvaluationDataset.model_validate(raw)
