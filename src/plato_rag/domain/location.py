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

import re
from dataclasses import dataclass
from enum import StrEnum

_RANGE_SEPARATOR_PATTERN = re.compile(r"\s*[\u2013-]\s*")
_SECTION_PREFIX_PATTERN = re.compile(r"^(?:section|sec\.?)\s+", re.IGNORECASE)
_PAGE_PREFIX_PATTERN = re.compile(r"^(?:p(?:age)?\.?)\s*", re.IGNORECASE)
_CHAPTER_PREFIX_PATTERN = re.compile(r"^(?:chapter|chap\.?)\s+", re.IGNORECASE)
_STEPHANUS_PATTERN = re.compile(r"^(?P<page>\d+)(?P<column>[a-e])$")
_BEKKER_PATTERN = re.compile(r"^(?P<page>\d+)(?P<column>[ab])(?P<line>\d+)?$")
_PAGE_PATTERN = re.compile(r"^\d+$")


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
        normalized_self_start = _normalize_location_value(self.system, self.value)
        normalized_self_end = (
            _normalize_location_value(self.system, self.range_end)
            if self.range_end is not None
            else None
        )
        normalized_other = _normalize_raw_reference(self.system, raw_value)
        if normalized_other is None:
            return False

        other_start, other_end = normalized_other
        if other_end is None:
            if other_start == normalized_self_start:
                return True
            if normalized_self_end is not None:
                return _range_contains(
                    self.system,
                    normalized_self_start,
                    normalized_self_end,
                    other_start,
                )
            return False

        return normalized_self_start == other_start and normalized_self_end == other_end

    def overlaps_raw_value(self, raw_value: str) -> bool:
        """Return True when the raw citation overlaps this location.

        This is stricter than substring matching and supports scholarly
        range citations like ``82b-85b`` or ``1094a1-1094a20``.
        """
        normalized_self_start = _normalize_location_value(self.system, self.value)
        normalized_self_end = (
            _normalize_location_value(self.system, self.range_end)
            if self.range_end is not None
            else normalized_self_start
        )
        normalized_other = _normalize_raw_reference(self.system, raw_value)
        if normalized_other is None:
            return False

        other_start, other_end = normalized_other
        if other_end is None:
            return _range_contains(
                self.system,
                normalized_self_start,
                normalized_self_end,
                other_start,
            )

        return _ranges_overlap(
            self.system,
            normalized_self_start,
            normalized_self_end,
            other_start,
            other_end,
        )


def _normalize_raw_reference(
    system: LocationSystem,
    raw_value: str,
) -> tuple[str, str | None] | None:
    normalized = _normalize_location_value(system, raw_value)
    if normalized is None:
        return None

    parts = _RANGE_SEPARATOR_PATTERN.split(normalized, maxsplit=1)
    start = parts[0]
    end = parts[1] if len(parts) == 2 else None
    if end == start:
        end = None
    return start, end


def _normalize_location_value(system: LocationSystem, raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u2212", "-")
    normalized = normalized.replace("\u2014", "-")
    normalized = normalized.replace("\u2013", "-")
    normalized = re.sub(r"\s+", " ", normalized)

    if system == LocationSystem.SECTION:
        normalized = normalized.lstrip("\u00a7").strip()
        normalized = _SECTION_PREFIX_PATTERN.sub("", normalized).strip()
        return normalized or None

    if system == LocationSystem.PAGE:
        normalized = _PAGE_PREFIX_PATTERN.sub("", normalized).strip()
        return normalized or None

    if system == LocationSystem.CHAPTER:
        normalized = _CHAPTER_PREFIX_PATTERN.sub("", normalized).strip()
        return normalized or None

    return normalized or None


def _range_contains(
    system: LocationSystem,
    start: str,
    end: str,
    value: str,
) -> bool:
    start_key = _ordered_value(system, start)
    end_key = _ordered_value(system, end)
    value_key = _ordered_value(system, value)
    if start_key is None or end_key is None or value_key is None:
        return value in (start, end)
    return start_key <= value_key <= end_key


def _ranges_overlap(
    system: LocationSystem,
    first_start: str,
    first_end: str,
    second_start: str,
    second_end: str,
) -> bool:
    first_start_key = _ordered_value(system, first_start)
    first_end_key = _ordered_value(system, first_end)
    second_start_key = _ordered_value(system, second_start)
    second_end_key = _ordered_value(system, second_end)
    if (
        first_start_key is None
        or first_end_key is None
        or second_start_key is None
        or second_end_key is None
    ):
        return first_start in (second_start, second_end) or first_end in (
            second_start,
            second_end,
        )
    return first_start_key <= second_end_key and second_start_key <= first_end_key


def _ordered_value(system: LocationSystem, raw_value: str) -> tuple[int, ...] | None:
    if system == LocationSystem.STEPHANUS:
        match = _STEPHANUS_PATTERN.fullmatch(raw_value)
        if match is None:
            return None
        return (int(match.group("page")), ord(match.group("column")) - ord("a"))

    if system == LocationSystem.BEKKER:
        match = _BEKKER_PATTERN.fullmatch(raw_value)
        if match is None:
            return None
        line_value = match.group("line")
        return (
            int(match.group("page")),
            ord(match.group("column")) - ord("a"),
            int(line_value) if line_value is not None else 0,
        )

    if system == LocationSystem.SECTION:
        parts = raw_value.split(".")
        if not parts:
            return None

        ordered_parts: list[int] = []
        for part in parts:
            if part.isdigit():
                ordered_parts.extend((0, int(part)))
                continue
            if part.isalpha():
                ordered_parts.extend((1, _alpha_value(part)))
                continue
            return None
        return tuple(ordered_parts)

    if system == LocationSystem.PAGE:
        if _PAGE_PATTERN.fullmatch(raw_value) is None:
            return None
        return (int(raw_value),)

    if system == LocationSystem.CHAPTER:
        if raw_value.isdigit():
            return (int(raw_value),)
        return None

    return None


def _alpha_value(value: str) -> int:
    result = 0
    for character in value:
        result = result * 26 + (ord(character) - ord("a") + 1)
    return result
