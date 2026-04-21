"""Evaluation dataset loading and scoring utilities."""

from plato_rag.evaluation.dataset import (
    CitationExpectation,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationExpectations,
    GenerationFixture,
    load_dataset,
)
from plato_rag.evaluation.runner import evaluate_case_response

__all__ = [
    "CitationExpectation",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationDataset",
    "EvaluationExpectations",
    "GenerationFixture",
    "evaluate_case_response",
    "load_dataset",
]
