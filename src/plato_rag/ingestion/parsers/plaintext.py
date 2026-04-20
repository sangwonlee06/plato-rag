"""Plaintext parser for primary philosophical texts.

Expects prepared text files with section markers in the format:

    [SECTION title="Recollection Argument" location="82b" speaker="Socrates"]
    The soul, then, as being immortal...

    [SECTION title="Slave Boy Demonstration" location="82b-85b" speaker="Socrates" interlocutor="Slave Boy"]
    Come here to me. Tell me, boy...

This is a deliberate design choice: rather than trying to parse arbitrary
PDFs or raw text, we require a lightweight preparation step that preserves
the critical metadata (location references, speaker attribution) that
makes academic citation possible. This preparation step is where human
editorial judgment enters the pipeline.
"""

from __future__ import annotations

import re

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.protocols.ingestion import ParsedDocument, ParsedSection

SECTION_PATTERN = re.compile(
    r"\[SECTION"
    r'(?:\s+title="(?P<title>[^"]*)")?'
    r'(?:\s+location="(?P<location>[^"]*)")?'
    r'(?:\s+speaker="(?P<speaker>[^"]*)")?'
    r'(?:\s+interlocutor="(?P<interlocutor>[^"]*)")?'
    r"\s*\]"
)

# Map collection names to their location reference system
COLLECTION_LOCATION_SYSTEMS: dict[str, LocationSystem] = {
    "platonic_dialogues": LocationSystem.STEPHANUS,
    "aristotle_corpus": LocationSystem.BEKKER,
    "presocratic_fragments": LocationSystem.DK,
}


def _parse_location(raw: str, collection: str) -> LocationRef | None:
    """Parse a location string into a LocationRef using the collection's system."""
    if not raw.strip():
        return None
    system = COLLECTION_LOCATION_SYSTEMS.get(collection, LocationSystem.CUSTOM)
    # Handle range references like "82b-85b"
    parts = re.split(r"[-\u2013]", raw.strip(), maxsplit=1)
    value = parts[0].strip()
    range_end = parts[1].strip() if len(parts) > 1 else None
    return LocationRef(system=system, value=value, range_end=range_end)


class PlaintextParser:
    """Parses prepared plaintext files with [SECTION] markers."""

    def parser_version(self) -> str:
        return "plaintext:1.0"

    def parse(self, raw_content: str, metadata: DocumentMetadata) -> ParsedDocument:
        sections: list[ParsedSection] = []
        current_text_lines: list[str] = []
        current_meta: dict[str, str | None] = {}

        def flush_section() -> None:
            text = "\n".join(current_text_lines).strip()
            if not text and not current_meta.get("title"):
                return
            loc_raw = current_meta.get("location") or ""
            sections.append(
                ParsedSection(
                    title=current_meta.get("title"),
                    text=text,
                    location_ref=_parse_location(loc_raw, metadata.collection),
                    speaker=current_meta.get("speaker"),
                    interlocutor=current_meta.get("interlocutor"),
                )
            )

        for line in raw_content.splitlines():
            match = SECTION_PATTERN.match(line.strip())
            if match:
                flush_section()
                current_text_lines = []
                current_meta = {
                    "title": match.group("title"),
                    "location": match.group("location"),
                    "speaker": match.group("speaker"),
                    "interlocutor": match.group("interlocutor"),
                }
            else:
                current_text_lines.append(line)

        flush_section()

        # If no sections were found, treat entire content as one section
        if not sections:
            sections.append(
                ParsedSection(
                    title=metadata.title,
                    text=raw_content.strip(),
                    location_ref=None,
                )
            )

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            raw_text=raw_content,
        )
