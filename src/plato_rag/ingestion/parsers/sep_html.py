"""HTML parser for Stanford Encyclopedia of Philosophy entries.

The parser extracts:

- entry-level metadata from standard SEP meta tags
- preamble text
- numbered section headings (`h2`, `h3`, `h4`) as structured sections
- the SEP section number as a `LocationRef(system=SECTION, value=...)`

It intentionally excludes the table of contents, bibliography, academic
tools, related entries, and footer navigation. The goal is to embed the
argumentative content of the article, not its navigation shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.protocols.ingestion import ParsedDocument, ParsedSection

_SECTION_NUMBER_PATTERN = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?)\s*$")
_STOP_SECTION_IDS = {"Bib", "Aca", "Oth", "Rel"}
_STOP_SECTION_TITLES = {
    "bibliography",
    "academic tools",
    "other internet resources",
    "related entries",
    "acknowledgments",
}


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_author(value: str) -> str:
    normalized = _normalize_whitespace(value)
    if "," not in normalized:
        return normalized

    pieces = [piece.strip() for piece in normalized.split(";") if piece.strip()]
    if not pieces:
        return normalized

    reordered: list[str] = []
    for piece in pieces:
        last_first = [part.strip() for part in piece.split(",", maxsplit=1)]
        if len(last_first) == 2 and last_first[1]:
            reordered.append(f"{last_first[1]} {last_first[0]}")
        else:
            reordered.append(piece)
    return "; ".join(reordered)


def _section_parts(heading_text: str) -> tuple[str | None, str]:
    normalized = _normalize_whitespace(heading_text)
    if not normalized:
        return None, ""

    match = _SECTION_NUMBER_PATTERN.match(normalized)
    if match is None:
        return None, normalized
    return match.group("number"), match.group("title")


@dataclass
class _SectionAccumulator:
    title: str | None
    location_value: str | None
    level: int
    blocks: list[str] = field(default_factory=list)


class _SepArticleCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entry_title: str | None = None
        self.entry_author: str | None = None
        self.last_updated: str | None = None

        self._inside_article = False
        self._article_depth = 0
        self._inside_preamble = False
        self._skip_depth = 0
        self._stopped = False

        self._capturing_pubinfo = False
        self._pubinfo_parts: list[str] = []
        self._capturing_h1 = False
        self._h1_parts: list[str] = []
        self._capturing_heading = False
        self._heading_level = 0
        self._heading_parts: list[str] = []
        self._heading_anchor: str | None = None
        self._capturing_block = False
        self._block_parts: list[str] = []

        self._preamble_blocks: list[str] = []
        self._current_section: _SectionAccumulator | None = None
        self.sections: list[ParsedSection] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value for name, value in attrs}

        if tag == "meta":
            self._handle_meta(attrs_dict)
            return

        if not self._inside_article:
            if tag == "div" and attrs_dict.get("id") == "aueditable":
                self._inside_article = True
                self._article_depth = 1
            return

        self._article_depth += 1

        if self._skip_depth > 0:
            self._skip_depth += 1
            return

        if tag == "div" and attrs_dict.get("id") == "toc":
            self._skip_depth = 1
            return
        if tag == "div" and attrs_dict.get("id") == "preamble":
            self._inside_preamble = True
            return
        if tag == "div" and attrs_dict.get("id") == "pubinfo":
            self._capturing_pubinfo = True
            self._pubinfo_parts = []
            return
        if tag == "h1":
            self._capturing_h1 = True
            self._h1_parts = []
            return
        if tag in {"h2", "h3", "h4"} and not self._stopped:
            self._flush_block()
            self._capturing_heading = True
            self._heading_level = int(tag[1])
            self._heading_parts = []
            self._heading_anchor = attrs_dict.get("id")
            return
        if self._capturing_heading and tag == "a" and self._heading_anchor is None:
            self._heading_anchor = attrs_dict.get("name")
            return
        if tag in {"p", "li"} and not self._stopped:
            self._capturing_block = True
            self._block_parts = []
            return
        if tag == "br" and self._capturing_block:
            self._block_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._inside_article:
            return

        if self._skip_depth > 0:
            self._skip_depth -= 1
            self._article_depth -= 1
            if self._article_depth == 0:
                self._finalize_article()
            return

        if tag == "div" and self._capturing_pubinfo:
            self._capturing_pubinfo = False
            if self.last_updated is None:
                self.last_updated = self._parse_pubinfo(self._pubinfo_parts)
            self._pubinfo_parts = []
        elif tag == "div" and self._inside_preamble:
            self._inside_preamble = False
        elif tag == "h1" and self._capturing_h1:
            self._capturing_h1 = False
            if self.entry_title is None:
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

        self._article_depth -= 1
        if self._article_depth == 0:
            self._finalize_article()

    def handle_data(self, data: str) -> None:
        if self._capturing_pubinfo:
            self._pubinfo_parts.append(data)
            return

        if not self._inside_article or self._skip_depth > 0 or self._stopped:
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

    def _handle_meta(self, attrs: dict[str, str | None]) -> None:
        key = attrs.get("property") or attrs.get("name")
        content = attrs.get("content")
        if key is None or content is None:
            return

        if key == "citation_title":
            self.entry_title = _normalize_whitespace(content)
        elif key == "citation_author":
            self.entry_author = _normalize_author(content)
        elif key == "DCTERMS.modified":
            self.last_updated = _normalize_whitespace(content)

    def _parse_pubinfo(self, parts: list[str]) -> str | None:
        text = _normalize_whitespace("".join(parts))
        if not text:
            return None
        match = re.search(
            r"substantive revision\s+\w+\s+(?P<month>\w+)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})",
            text,
        )
        if match is not None:
            candidate = f"{match.group('month')} {match.group('day')} {match.group('year')}"
            for fmt in ("%b %d %Y", "%B %d %Y"):
                try:
                    parsed = datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
                return parsed.date().isoformat()
        return text

    def _flush_block(self) -> None:
        if not self._capturing_block:
            return

        text = _normalize_whitespace("".join(self._block_parts))
        self._capturing_block = False
        self._block_parts = []
        if not text:
            return

        if self._inside_preamble and not self._stopped:
            self._preamble_blocks.append(text)
            return

        if self._current_section is None or self._stopped:
            return
        self._current_section.blocks.append(text)

    def _start_section_from_heading(self) -> None:
        heading_text = _normalize_whitespace("".join(self._heading_parts))
        if not heading_text:
            return

        location_value, title = _section_parts(heading_text)
        normalized_title = title.casefold()
        if self._heading_anchor in _STOP_SECTION_IDS or normalized_title in _STOP_SECTION_TITLES:
            self._flush_current_section()
            self._stopped = True
            return

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
            self.sections.append(ParsedSection(
                title=self._current_section.title,
                text=text,
                location_ref=location_ref,
                level=self._current_section.level,
            ))
        self._current_section = None

    def _finalize_article(self) -> None:
        self._flush_block()
        if self._preamble_blocks:
            self.sections.insert(0, ParsedSection(
                title="Preamble",
                text="\n\n".join(self._preamble_blocks),
                level=0,
            ))
            self._preamble_blocks = []
        self._flush_current_section()
        self._inside_article = False


class SepHtmlParser:
    """Parse Stanford Encyclopedia of Philosophy entry HTML."""

    def parser_version(self) -> str:
        return "sep_html:1.0"

    def parse(self, raw_content: str, metadata: DocumentMetadata) -> ParsedDocument:
        collector = _SepArticleCollector()
        collector.feed(raw_content)
        collector.close()

        if collector.entry_title:
            metadata.title = collector.entry_title
        if collector.entry_author:
            metadata.author = collector.entry_author

        extra_metadata: dict[str, str] = {}
        if metadata.source_url:
            extra_metadata["entry_url"] = metadata.source_url
        if collector.last_updated:
            extra_metadata["last_updated"] = collector.last_updated

        if not collector.sections:
            collector.sections.append(ParsedSection(
                title=metadata.title,
                text=_normalize_whitespace(raw_content),
            ))

        return ParsedDocument(
            metadata=metadata,
            sections=collector.sections,
            raw_text=raw_content,
            extra_metadata=extra_metadata or None,
        )
