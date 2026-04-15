"""Shared types for API contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from plato_rag.domain.source import SourceClass

__all__ = [
    "ChatMode",
    "CompatSourceType",
    "ConversationRole",
    "ConversationTurn",
    "InterpretationLevel",
    "SourceClass",
    "compat_source_type_for",
]


class CompatSourceType(StrEnum):
    """NestJS-compatible source type (PRIMARY / SECONDARY).

    The NestJS app currently uses this simpler classification.
    The RAG service returns both this and the richer source_class.
    """

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"


def compat_source_type_for(source_class: SourceClass) -> CompatSourceType:
    """Map a SourceClass to the NestJS-compatible PRIMARY/SECONDARY."""
    if source_class == SourceClass.PRIMARY_TEXT:
        return CompatSourceType.PRIMARY
    return CompatSourceType.SECONDARY


class InterpretationLevel(StrEnum):
    """How directly the answer is grounded in retrieved sources."""

    DIRECT = "DIRECT"
    INTERPRETIVE = "INTERPRETIVE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class ChatMode(StrEnum):
    """Product modes. Only PLATO is implemented."""

    PLATO = "PLATO"
    FREGE = "FREGE"


class ConversationRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class ConversationTurn(BaseModel):
    role: ConversationRole
    content: str
