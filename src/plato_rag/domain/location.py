"""Structured location reference for philosophical texts.

Location references are the basis of academic citation. Different
philosophical traditions use different reference systems:

- Stephanus numbers for Plato (e.g., 86b, 514a-520a)
- Bekker numbers for Aristotle (e.g., 1094a1)
- DK numbers for Presocratics (e.g., DK 22 B30)
- Section numbers for encyclopedias (e.g., §2.1)
- Page numbers for modern works (e.g., p. 42)

Storing these as typed LocationRef rather than opaque strings enables:
- Correct citation formatting per reference system
- Reliable citation matching in the CitationExtractor
- Range reference support (86b-86d)
- Validation that a location is syntactically correct for its system
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LocationSystem(StrEnum):
    """Reference system used for locating passages in a work."""

    STEPHANUS = "stephanus"
    BEKKER = "bekker"
    DK = "dk"
    SECTION = "section"
    PAGE = "page"
    PARAGRAPH = "paragraph"
    LINE = "line"
    CHAPTER = "chapter"
    CUSTOM = "custom"


@dataclass(frozen=True)
class LocationRef:
    """A structured reference to a location within a philosophical work.

    The system field identifies the reference convention.
    The value field is the primary reference (e.g., "86b").
    The range_end field, when present, marks the end of a range (e.g., "86d" for "86b-86d").
    """

    system: LocationSystem
    value: str
    range_end: str | None = None

    def display(self) -> str:
        """Format as a display string (e.g., '86b' or '86b-86d')."""
        if self.range_end:
            return f"{self.value}\u2013{self.range_end}"
        return self.value

    def display_with_prefix(self) -> str:
        """Format with system-appropriate prefix where helpful.

        Sections get a § prefix. Other systems display as-is
        since the system is clear from context (work title).
        """
        base = self.display()
        if self.system == LocationSystem.SECTION and not base.startswith("\u00a7"):
            return f"\u00a7{base}"
        if self.system == LocationSystem.PAGE and not base.startswith("p."):
            return f"p. {base}"
        return base

    def matches_value(self, raw_value: str) -> bool:
        """Check if a raw string matches this location's value.

        Used by the CitationExtractor to match LLM-generated
        references like '86b' against structured LocationRefs.
        Handles minor formatting variations.
        """
        normalized_self = self.value.strip().lower()
        normalized_other = raw_value.strip().lower()
        if normalized_self == normalized_other:
            return True
        # Handle section prefix variations: "2.1" matches "Section 2.1" matches "§2.1"
        for prefix in ("section ", "\u00a7", "sec. ", "sec "):
            stripped_self = normalized_self[len(prefix):].strip()
            stripped_other = normalized_other[len(prefix):].strip()
            if normalized_self.startswith(prefix) and stripped_self == normalized_other:
                return True
            if normalized_other.startswith(prefix) and normalized_self == stripped_other:
                return True
        return False
