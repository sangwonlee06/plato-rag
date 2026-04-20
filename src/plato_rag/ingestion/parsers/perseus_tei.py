"""TEI parser for public-domain Perseus primary texts.

Perseus TEI is structurally consistent enough to support a shared ingestion
path, but not uniform enough to justify a single parsing strategy for all
works. Plato dialogues expose Stephanus milestones and speaker turns; Aristotle
texts expose Bekker page/line milestones inside prose treatises.

This parser supports both modes while preserving citation-grade references.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Literal

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.protocols.ingestion import ParsedDocument, ParsedSection

_WHITESPACE_PATTERN = re.compile(r"\s+")
_ENGLISH_TITLE_SUFFIX_PATTERN = re.compile(
    r"\s*\(English\)\.?\s*(?:Machine readable text)?\s*$",
    re.IGNORECASE,
)
_BEKKER_PAGE_PATTERN = re.compile(r"^\d+[ab]$")
_SKIP_TEXT_TAGS = {
    "castList",
    "note",
    "bibl",
    "speaker",
}

PerseusParseMode = Literal["dialogue", "bekker_treatise"]


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", maxsplit=1)[1]
    return tag


def _normalize_whitespace(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _normalize_section_text(value: str) -> str:
    paragraphs = [_normalize_whitespace(paragraph) for paragraph in value.split("\n\n")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _build_location_ref(
    system: LocationSystem,
    start_value: str | None,
    end_value: str | None = None,
) -> LocationRef | None:
    if start_value is None:
        return None

    normalized_start = _normalize_whitespace(start_value)
    if not normalized_start:
        return None

    normalized_end = None
    if end_value is not None:
        normalized_end = _normalize_whitespace(end_value)
        if normalized_end == normalized_start:
            normalized_end = None

    return LocationRef(system=system, value=normalized_start, range_end=normalized_end)


@dataclass
class _DialogueSectionAccumulator:
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
            location_ref=_build_location_ref(LocationSystem.STEPHANUS, self.location_value),
            speaker=speaker,
        )


@dataclass
class _TreatiseSectionAccumulator:
    title: str | None
    fragments: list[str] = field(default_factory=list)
    start_location: str | None = None
    end_location: str | None = None

    def has_text(self) -> bool:
        return bool(_normalize_section_text("".join(self.fragments)))

    def append_text(self, value: str, location_value: str | None) -> None:
        normalized = _normalize_whitespace(value)
        if not normalized:
            return

        if (
            self.fragments
            and self.fragments[-1] not in {" ", "\n\n"}
            and not self.fragments[-1].endswith((" ", "\n", "“", '"', "‘", "("))
        ):
            self.fragments.append(" ")

        if location_value is not None:
            if self.start_location is None:
                self.start_location = location_value
            self.end_location = location_value

        self.fragments.append(normalized)

    def as_parsed_section(self) -> ParsedSection | None:
        text = _normalize_section_text("".join(self.fragments))
        if not text:
            return None

        return ParsedSection(
            title=self.title,
            text=text,
            location_ref=_build_location_ref(
                LocationSystem.BEKKER,
                self.start_location,
                self.end_location,
            ),
        )


class PerseusTeiParser:
    """Parse a Perseus TEI document into citation-grade sections."""

    def __init__(
        self,
        *,
        text_identifier: str | None = None,
        parse_mode: PerseusParseMode = "dialogue",
    ) -> None:
        self._text_identifier = text_identifier
        self._parse_mode = parse_mode

    def parser_version(self) -> str:
        return f"perseus_tei:{self._parse_mode}:1.1"

    def parse(self, raw_content: str, metadata: DocumentMetadata) -> ParsedDocument:
        root = ET.fromstring(raw_content.lstrip())
        work_text = self._find_work_text(root)
        body = work_text.find("body")
        if body is None:
            raise ValueError("Perseus TEI document is missing a body element")

        title = self._extract_document_title(root, work_text)
        if title is not None:
            metadata.title = title

        if self._parse_mode == "dialogue":
            sections = self._parse_dialogue_sections(body, metadata)
        else:
            sections = self._parse_bekker_treatise_sections(body, metadata)

        if not sections:
            raise ValueError("Perseus TEI document did not produce any sections")

        extra_metadata: dict[str, str] | None = None
        if metadata.source_url is not None:
            extra_metadata = {"source_url": metadata.source_url}

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            raw_text=raw_content,
            extra_metadata=extra_metadata,
        )

    def _parse_dialogue_sections(
        self,
        body: ET.Element,
        metadata: DocumentMetadata,
    ) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        current_section: _DialogueSectionAccumulator | None = None
        active_speaker: str | None = None

        def flush_section() -> None:
            nonlocal current_section
            if current_section is None:
                return
            parsed_section = current_section.as_parsed_section()
            if parsed_section is not None:
                sections.append(parsed_section)
            current_section = None

        def ensure_section() -> _DialogueSectionAccumulator:
            nonlocal current_section
            if current_section is None:
                current_section = _DialogueSectionAccumulator(
                    title=metadata.title,
                    location_value=None,
                )
                if active_speaker is not None:
                    current_section.pending_speaker = active_speaker
            return current_section

        def start_section(location_value: str | None) -> None:
            nonlocal current_section
            flush_section()
            current_section = _DialogueSectionAccumulator(
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
        return sections

    def _parse_bekker_treatise_sections(
        self,
        body: ET.Element,
        metadata: DocumentMetadata,
    ) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        current_section: _TreatiseSectionAccumulator | None = None
        current_book: str | None = None
        current_chapter: str | None = None
        current_bekker_page: str | None = None
        current_bekker_line: str | None = None

        def flush_section() -> None:
            nonlocal current_section
            if current_section is None:
                return
            parsed_section = current_section.as_parsed_section()
            if parsed_section is not None:
                sections.append(parsed_section)
            current_section = None

        def current_location() -> str | None:
            if current_bekker_page is None:
                return None
            if current_bekker_line is None:
                return current_bekker_page
            return f"{current_bekker_page}{current_bekker_line}"

        def current_title() -> str:
            if current_book is not None and current_chapter is not None:
                chapter_label = current_chapter
                prefix = f"{current_book}."
                if chapter_label.startswith(prefix):
                    chapter_label = chapter_label[len(prefix) :]
                return f"Book {current_book}, Chapter {chapter_label}"
            if current_book is not None:
                return f"Book {current_book}"
            return metadata.title

        def ensure_section() -> _TreatiseSectionAccumulator:
            nonlocal current_section
            if current_section is None:
                current_section = _TreatiseSectionAccumulator(title=current_title())
            return current_section

        def append_text(value: str) -> None:
            ensure_section().append_text(value, current_location())

        def update_book(value: str | None) -> None:
            nonlocal current_book, current_chapter
            normalized = _normalize_whitespace(value or "")
            if not normalized:
                return
            flush_section()
            current_book = normalized
            current_chapter = None

        def update_chapter(value: str | None) -> None:
            nonlocal current_chapter
            normalized = _normalize_whitespace(value or "")
            if not normalized:
                return
            flush_section()
            current_chapter = normalized

        def update_bekker_page(value: str | None) -> None:
            nonlocal current_bekker_page
            normalized = _normalize_whitespace(value or "")
            if not normalized:
                return
            current_bekker_page = normalized

        def update_bekker_line(value: str | None) -> None:
            nonlocal current_bekker_line
            normalized = _normalize_whitespace(value or "")
            if not normalized:
                return
            current_bekker_line = normalized

        def walk(element: ET.Element) -> None:
            tag = _local_name(element.tag)
            if tag in _SKIP_TEXT_TAGS:
                return

            if tag == "div1":
                update_book(element.attrib.get("n"))
                for child in element:
                    walk(child)
                    if child.tail:
                        append_text(child.tail)
                return

            if tag == "milestone":
                unit = _normalize_whitespace(element.attrib.get("unit", "")).casefold()
                value = element.attrib.get("n")
                if self._is_bekker_page_milestone(unit, value):
                    update_bekker_page(value)
                    return
                if self._is_bekker_line_milestone(unit, element.attrib.get("ed")):
                    update_bekker_line(value)
                    return
                if unit in {"chapter", "loeb chap"}:
                    update_chapter(value)
                    return
                return

            if tag == "head":
                return

            if tag == "p":
                flush_section()
                if element.text:
                    append_text(element.text)
                for child in element:
                    walk(child)
                    if child.tail:
                        append_text(child.tail)
                flush_section()
                return

            if element.text:
                append_text(element.text)

            for child in element:
                walk(child)
                if child.tail:
                    append_text(child.tail)

        walk(body)
        flush_section()
        return sections

    def _find_work_text(self, root: ET.Element) -> ET.Element:
        text_elements = [text for text in root.findall(".//text") if text.find("body") is not None]
        if root.find("body") is not None and root not in text_elements:
            text_elements.insert(0, root)

        if self._text_identifier is None:
            if len(text_elements) == 1:
                return text_elements[0]
            raise ValueError(
                "Perseus TEI document contains multiple works; source_config.text_id is required"
            )

        normalized_target = _normalize_whitespace(self._text_identifier).casefold()
        for text_element in text_elements:
            candidate = _normalize_whitespace(text_element.attrib.get("n", "")).casefold()
            if candidate == normalized_target:
                return text_element

            title = self._extract_body_title(text_element)
            if title is not None and title.casefold() == normalized_target:
                return text_element

        raise ValueError(f"Unable to locate Perseus text {self._text_identifier!r}")

    def _extract_document_title(self, root: ET.Element, work_text: ET.Element) -> str | None:
        if self._parse_mode == "dialogue":
            body_title = self._extract_body_title(work_text)
            if body_title is not None:
                return body_title

        title_stmt = root.find("./teiHeader/fileDesc/titleStmt")
        if title_stmt is None:
            return None

        for title_element in title_stmt.findall("title"):
            title = _normalize_whitespace("".join(title_element.itertext()))
            if not title:
                continue
            title = _ENGLISH_TITLE_SUFFIX_PATTERN.sub("", title).strip(" .")
            if title and title.casefold() != "machine readable text":
                return title
        return None

    def _extract_body_title(self, text_element: ET.Element) -> str | None:
        head = text_element.find("body/head")
        if head is None:
            return None
        title = _normalize_whitespace("".join(head.itertext()))
        return title or None

    def _is_bekker_page_milestone(
        self,
        unit: str,
        value: str | None,
    ) -> bool:
        normalized = _normalize_whitespace(value or "")
        if not normalized:
            return False
        if unit == "bekker page":
            return True
        return unit == "section" and _BEKKER_PAGE_PATTERN.fullmatch(normalized) is not None

    def _is_bekker_line_milestone(
        self,
        unit: str,
        edition: str | None,
    ) -> bool:
        normalized_edition = _normalize_whitespace(edition or "").casefold()
        return unit in {"bekker line", "line"} and normalized_edition in {"", "bekker"}
