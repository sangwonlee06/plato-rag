"""HTML parser for Internet Encyclopedia of Philosophy entries.

The parser extracts:

- entry title from the article body (or page title as fallback)
- author attribution from the "Author Information" section
- preamble paragraphs before the table of contents
- numbered and lettered section headings as structured sections
- hierarchical section references such as ``1`` and ``1.a``

It intentionally excludes the table of contents, references/further reading,
author information details, and the surrounding WordPress site chrome.
The goal is to embed the article's substantive philosophical content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.protocols.ingestion import ParsedDocument, ParsedSection

_TOP_SECTION_PATTERN = re.compile(r"^\s*(?P<number>\d+)\.?\s+(?P<title>.+?)\s*$")
_LETTERED_SECTION_PATTERN = re.compile(r"^\s*(?P<label>[a-z])\.\s*(?P<title>.+?)\s*$", re.I)
_TOP_SECTION_ANCHOR_PATTERN = re.compile(r"^H(?P<number>\d+)$", re.I)
_SUBSECTION_ANCHOR_PATTERN = re.compile(r"^SH(?P<number>\d+)(?P<label>[a-z])$", re.I)
_TITLE_SUFFIX = " | Internet Encyclopedia of Philosophy"


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_block_text(value: str) -> str:
    lines = [_normalize_whitespace(line) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def _strip_title_suffix(value: str) -> str:
    normalized = _normalize_whitespace(value)
    if normalized.endswith(_TITLE_SUFFIX):
        return normalized[: -len(_TITLE_SUFFIX)].strip()
    return normalized


def _extract_author_name(block_text: str) -> str | None:
    for line in block_text.splitlines():
        normalized = _normalize_whitespace(line)
        if not normalized:
            continue
        if normalized.casefold() == "and":
            continue
        if normalized.casefold().startswith("email:"):
            continue
        if normalized.casefold() in {"u. s. a.", "usa"}:
            continue
        return normalized
    return None


def _heading_parts(
    heading_text: str,
    heading_anchor: str | None,
    current_top_section: str | None,
) -> tuple[str | None, str, str | None]:
    normalized = _normalize_whitespace(heading_text)
    if not normalized:
        return None, "", current_top_section

    anchor = heading_anchor or ""

    match = _TOP_SECTION_PATTERN.match(normalized)
    if match is not None:
        number = match.group("number")
        return number, match.group("title"), number

    match = _LETTERED_SECTION_PATTERN.match(normalized)
    if match is not None:
        title = match.group("title")
        label = match.group("label").lower()
        if current_top_section is None:
            return label, title, current_top_section
        return f"{current_top_section}.{label}", title, current_top_section

    match = _TOP_SECTION_ANCHOR_PATTERN.match(anchor)
    if match is not None:
        number = match.group("number")
        return number, normalized, number

    match = _SUBSECTION_ANCHOR_PATTERN.match(anchor)
    if match is not None:
        number = match.group("number")
        label = match.group("label").lower()
        return f"{number}.{label}", normalized, number

    return None, normalized, current_top_section


@dataclass
class _SectionAccumulator:
    title: str | None
    location_value: str | None
    level: int
    blocks: list[str] = field(default_factory=list)


class _IepArticleCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.document_title: str | None = None
        self.entry_title: str | None = None
        self.entry_author: str | None = None

        self._inside_entry = False
        self._entry_depth = 0
        self._mode = "normal"
        self._current_top_section: str | None = None

        self._capturing_document_title = False
        self._document_title_parts: list[str] = []
        self._capturing_h1 = False
        self._h1_parts: list[str] = []
        self._capturing_heading = False
        self._heading_level = 0
        self._heading_parts: list[str] = []
        self._heading_anchor: str | None = None
        self._capturing_block = False
        self._block_parts: list[str] = []

        self._preamble_blocks: list[str] = []
        self._author_names: list[str] = []
        self._current_section: _SectionAccumulator | None = None
        self.sections: list[ParsedSection] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value for name, value in attrs}

        if tag == "title" and self.document_title is None:
            self._capturing_document_title = True
            self._document_title_parts = []
            return

        if not self._inside_entry:
            if tag == "div" and _has_css_class(attrs_dict.get("class"), "entry-content"):
                self._inside_entry = True
                self._entry_depth = 1
            return

        self._entry_depth += 1

        if tag == "h1":
            self._flush_block()
            self._capturing_h1 = True
            self._h1_parts = []
            return

        if tag in {"h2", "h3", "h4"}:
            if self._mode == "skip_toc":
                self._mode = "normal"
            self._flush_block()
            self._capturing_heading = True
            self._heading_level = int(tag[1])
            self._heading_parts = []
            self._heading_anchor = attrs_dict.get("id")
            return

        if self._capturing_heading and tag == "a" and self._heading_anchor is None:
            self._heading_anchor = attrs_dict.get("name") or attrs_dict.get("id")
            return

        if tag in {"p", "li"} and self._mode in {"normal", "author_info"}:
            self._capturing_block = True
            self._block_parts = []
            return

        if tag == "br" and self._capturing_block:
            self._block_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._capturing_document_title:
            self._capturing_document_title = False
            self.document_title = _strip_title_suffix("".join(self._document_title_parts))
            self._document_title_parts = []
            return

        if not self._inside_entry:
            return

        if tag == "h1" and self._capturing_h1:
            self._capturing_h1 = False
            self.entry_title = _normalize_whitespace("".join(self._h1_parts))
            self._h1_parts = []
        elif tag in {"h2", "h3", "h4"} and self._capturing_heading:
            self._capturing_heading = False
            self._start_section_from_heading()
            self._heading_level = 0
            self._heading_parts = []
            self._heading_anchor = None
        elif tag in {"p", "li"} and self._capturing_block:
            self._flush_block()

        self._entry_depth -= 1
        if self._entry_depth == 0:
            self._finalize_entry()

    def handle_data(self, data: str) -> None:
        if self._capturing_document_title:
            self._document_title_parts.append(data)
            return

        if not self._inside_entry:
            return

        if self._capturing_h1:
            self._h1_parts.append(data)
        elif self._capturing_heading:
            self._heading_parts.append(data)
        elif self._capturing_block:
            self._block_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")

    def _flush_block(self) -> None:
        if not self._capturing_block:
            return

        text = _normalize_block_text("".join(self._block_parts))
        self._capturing_block = False
        self._block_parts = []
        if not text:
            return

        if self._mode == "author_info":
            author_name = _extract_author_name(text)
            if author_name and author_name not in self._author_names:
                self._author_names.append(author_name)
            return

        if self._mode != "normal":
            return

        if self._current_section is None:
            self._preamble_blocks.append(text)
            return

        self._current_section.blocks.append(text)

    def _start_section_from_heading(self) -> None:
        heading_text = _normalize_whitespace("".join(self._heading_parts))
        if not heading_text:
            return

        _, normalized_title, _ = _heading_parts(
            heading_text,
            self._heading_anchor,
            self._current_top_section,
        )
        lowered = normalized_title.casefold()
        if lowered == "table of contents":
            self._mode = "skip_toc"
            return
        if lowered == "references and further reading":
            self._flush_current_section()
            self._mode = "skip_references"
            return
        if lowered == "author information":
            self._flush_current_section()
            self._mode = "author_info"
            return
        if self._mode == "skip_references":
            return

        location_value, title, current_top = _heading_parts(
            heading_text,
            self._heading_anchor,
            self._current_top_section,
        )
        self._current_top_section = current_top

        self._flush_current_section()
        self._current_section = _SectionAccumulator(
            title=title,
            location_value=location_value,
            level=self._heading_level - 1,
        )

    def _flush_current_section(self) -> None:
        if self._current_section is None:
            return

        text = "\n\n".join(self._current_section.blocks).strip()
        if text:
            location_ref = None
            if self._current_section.location_value is not None:
                location_ref = LocationRef(
                    system=LocationSystem.SECTION,
                    value=self._current_section.location_value,
                )
            self.sections.append(
                ParsedSection(
                    title=self._current_section.title,
                    text=text,
                    location_ref=location_ref,
                    level=self._current_section.level,
                )
            )
        self._current_section = None

    def _finalize_entry(self) -> None:
        self._flush_block()
        self._flush_current_section()

        if self._preamble_blocks:
            self.sections.insert(
                0,
                ParsedSection(
                    title="Preamble",
                    text="\n\n".join(self._preamble_blocks),
                    level=0,
                ),
            )
            self._preamble_blocks = []

        if self._author_names and self.entry_author is None:
            if len(self._author_names) == 1:
                self.entry_author = self._author_names[0]
            elif len(self._author_names) == 2:
                self.entry_author = " and ".join(self._author_names)
            else:
                self.entry_author = "; ".join(self._author_names)

        self._inside_entry = False


class IepHtmlParser:
    """Parse Internet Encyclopedia of Philosophy entry HTML."""

    def parser_version(self) -> str:
        return "iep_html:1.0"

    def parse(self, raw_content: str, metadata: DocumentMetadata) -> ParsedDocument:
        collector = _IepArticleCollector()
        collector.feed(raw_content)
        collector.close()

        if collector.entry_title:
            metadata.title = collector.entry_title
        elif collector.document_title:
            metadata.title = collector.document_title

        if collector.entry_author:
            metadata.author = collector.entry_author

        extra_metadata: dict[str, str] = {}
        if metadata.source_url:
            extra_metadata["entry_url"] = metadata.source_url

        if not collector.sections:
            collector.sections.append(
                ParsedSection(
                    title=metadata.title,
                    text=_normalize_whitespace(raw_content),
                )
            )

        return ParsedDocument(
            metadata=metadata,
            sections=collector.sections,
            raw_text=raw_content,
            extra_metadata=extra_metadata or None,
        )


def _has_css_class(raw_classes: str | None, expected: str) -> bool:
    if raw_classes is None:
        return False
    return expected in {item.strip() for item in raw_classes.split() if item.strip()}
