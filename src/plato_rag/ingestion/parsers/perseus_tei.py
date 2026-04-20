"""TEI parser for public-domain Plato texts from Perseus.

The Perseus ``dltext`` endpoint exposes TEI XML with stable ``milestone``
markers for Stephanus references and ``sp``/``speaker`` tags for dialogues.
This parser walks the TEI document in order so section breaks are preserved
even when milestones appear inside a paragraph or speech.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.protocols.ingestion import ParsedDocument, ParsedSection

_WHITESPACE_PATTERN = re.compile(r"\s+")
_SKIP_TEXT_TAGS = {
    "castList",
    "note",
    "bibl",
    "speaker",
}


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", maxsplit=1)[1]
    return tag


def _normalize_whitespace(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _normalize_section_text(value: str) -> str:
    paragraphs = [_normalize_whitespace(paragraph) for paragraph in value.split("\n\n")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _location_ref(raw_value: str | None) -> LocationRef | None:
    if raw_value is None:
        return None
    normalized = _normalize_whitespace(raw_value)
    if not normalized:
        return None
    return LocationRef(system=LocationSystem.STEPHANUS, value=normalized)


@dataclass
class _SectionAccumulator:
    title: str | None
    location_value: str | None
    fragments: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    pending_speaker: str | None = None

    def has_text(self) -> bool:
        return bool(_normalize_section_text("".join(self.fragments)))

    def start_paragraph(self) -> None:
        if self.fragments and not self.fragments[-1].endswith("\n\n"):
            self.fragments.append("\n\n")

    def append_text(self, value: str) -> None:
        if not value:
            return
        normalized = _normalize_whitespace(value)
        if not normalized:
            return
        if self.pending_speaker is not None:
            self.fragments.append(f"{self.pending_speaker}: ")
            if self.pending_speaker not in self.speakers:
                self.speakers.append(self.pending_speaker)
            self.pending_speaker = None
        if (
            self.fragments
            and self.fragments[-1] not in {" ", "\n\n"}
            and not self.fragments[-1].endswith((" ", "\n", "“", '"', "‘", "("))
        ):
            self.fragments.append(" ")
        self.fragments.append(normalized)

    def as_parsed_section(self) -> ParsedSection | None:
        text = _normalize_section_text("".join(self.fragments))
        if not text:
            return None

        speaker: str | None = None
        if len(self.speakers) == 1:
            speaker = self.speakers[0]

        return ParsedSection(
            title=self.title,
            text=text,
            location_ref=_location_ref(self.location_value),
            speaker=speaker,
        )


class PerseusTeiParser:
    """Parse a Plato dialogue from Perseus TEI XML."""

    def __init__(self, *, text_identifier: str) -> None:
        self._text_identifier = text_identifier

    def parser_version(self) -> str:
        return "perseus_tei:1.0"

    def parse(self, raw_content: str, metadata: DocumentMetadata) -> ParsedDocument:
        root = ET.fromstring(raw_content.lstrip())
        work_text = self._find_work_text(root)
        body = work_text.find("body")
        if body is None:
            raise ValueError(f"Perseus text {self._text_identifier!r} is missing a body element")

        title = self._extract_title(work_text)
        if title is not None:
            metadata.title = title

        sections: list[ParsedSection] = []
        current_section: _SectionAccumulator | None = None
        active_speaker: str | None = None

        def flush_section() -> None:
            nonlocal current_section
            if current_section is None:
                return
            parsed_section = current_section.as_parsed_section()
            if parsed_section is not None:
                sections.append(parsed_section)
            current_section = None

        def ensure_section() -> _SectionAccumulator:
            nonlocal current_section
            if current_section is None:
                current_section = _SectionAccumulator(title=metadata.title, location_value=None)
                if active_speaker is not None:
                    current_section.pending_speaker = active_speaker
            return current_section

        def start_section(location_value: str | None) -> None:
            nonlocal current_section
            flush_section()
            current_section = _SectionAccumulator(
                title=location_value,
                location_value=location_value,
            )
            if active_speaker is not None:
                current_section.pending_speaker = active_speaker

        def start_speech(speaker_name: str | None) -> None:
            if speaker_name is None:
                return
            section = ensure_section()
            if section.has_text():
                section.start_paragraph()
            section.pending_speaker = speaker_name

        def walk(element: ET.Element) -> None:
            nonlocal active_speaker

            tag = _local_name(element.tag)
            if tag in _SKIP_TEXT_TAGS:
                return

            if tag == "milestone":
                if element.attrib.get("unit") == "section":
                    start_section(element.attrib.get("n"))
                return

            if tag == "head":
                return

            if tag == "sp":
                speaker_element = element.find("speaker")
                speaker_name = None
                if speaker_element is not None:
                    speaker_name = _normalize_whitespace("".join(speaker_element.itertext()))

                previous_speaker = active_speaker
                active_speaker = speaker_name or previous_speaker
                if active_speaker is not None:
                    start_speech(active_speaker)

                for child in element:
                    walk(child)
                    if child.tail:
                        ensure_section().append_text(child.tail)

                active_speaker = previous_speaker
                return

            if tag == "p":
                section = ensure_section()
                if section.has_text():
                    section.start_paragraph()
                if element.text:
                    section.append_text(element.text)
                for child in element:
                    walk(child)
                    if child.tail:
                        ensure_section().append_text(child.tail)
                return

            if element.text:
                ensure_section().append_text(element.text)

            for child in element:
                walk(child)
                if child.tail:
                    ensure_section().append_text(child.tail)

        walk(body)
        flush_section()

        if not sections:
            raise ValueError(f"Perseus text {self._text_identifier!r} did not produce any sections")

        extra_metadata: dict[str, str] | None = None
        if metadata.source_url is not None:
            extra_metadata = {"source_url": metadata.source_url}

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            raw_text=raw_content,
            extra_metadata=extra_metadata,
        )

    def _find_work_text(self, root: ET.Element) -> ET.Element:
        normalized_target = _normalize_whitespace(self._text_identifier).casefold()
        for text_element in root.findall(".//text"):
            candidate = _normalize_whitespace(text_element.attrib.get("n", "")).casefold()
            if candidate == normalized_target:
                return text_element

            title = self._extract_title(text_element)
            if title is not None and title.casefold() == normalized_target:
                return text_element

        raise ValueError(f"Unable to locate Perseus text {self._text_identifier!r}")

    def _extract_title(self, text_element: ET.Element) -> str | None:
        head = text_element.find("body/head")
        if head is None:
            return None
        title = _normalize_whitespace("".join(head.itertext()))
        return title or None
