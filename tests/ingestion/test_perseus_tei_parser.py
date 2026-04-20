from __future__ import annotations

import uuid

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationSystem
from plato_rag.domain.source import SourceClass
from plato_rag.ingestion.chunkers.section import SectionChunker
from plato_rag.ingestion.parsers.perseus_tei import PerseusTeiParser
from plato_rag.protocols.ingestion import ChunkConfig

PERSEUS_TEI = """
<?xml version="1.0" encoding="utf-8"?>
<TEI.2>
  <text>
    <group>
      <text n="Meno">
        <body>
          <head>Meno</head>
          <sp>
            <speaker>Meno</speaker>
            <p>
              <milestone unit="section" n="70a" />
              Can you tell me, Socrates, whether virtue can be taught?
            </p>
          </sp>
          <sp>
            <speaker>Socrates</speaker>
            <p>
              Not yet, Meno.
              <milestone unit="section" n="70b" />
              I do not even know what virtue is.
            </p>
          </sp>
          <sp>
            <speaker>Meno</speaker>
            <p><note>Editorial note.</note>Then what shall we do?</p>
          </sp>
          <sp>
            <speaker>Socrates</speaker>
            <p><milestone unit="section" n="70c" />We shall inquire together.</p>
          </sp>
        </body>
      </text>
    </group>
  </text>
</TEI.2>
"""

ARISTOTLE_TEI = """
<?xml version="1.0" encoding="utf-8"?>
<TEI.2>
  <teiHeader type="text" status="new">
    <fileDesc>
      <titleStmt>
        <title>Nicomachean Ethics (English)</title>
      </titleStmt>
    </fileDesc>
  </teiHeader>
  <text lang="en">
    <body>
      <div1 n="1" type="Book">
        <head>Book 1</head>
        <milestone n="1094a" unit="bekker page" />
        <milestone n="1" unit="bekker line" />
        <milestone n="1" unit="chapter" />
        <p>
          Every art and every inquiry seems to aim at some good.
          <milestone n="20" ed="Bekker" unit="line" />
          Hence it has been said that the good is that at which all things aim.
          <milestone n="2" unit="chapter" />
          If there is an end of our actions that we wish for itself, this will be the good.
        </p>
        <p>
          <milestone n="1094b" unit="bekker page" />
          <milestone n="1" ed="Bekker" unit="line" />
          Political science ordains which sciences should exist in the city.
        </p>
      </div1>
    </body>
  </text>
</TEI.2>
"""


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        id=uuid.uuid4(),
        title="placeholder",
        author="Plato",
        source_class=SourceClass.PRIMARY_TEXT,
        collection="platonic_dialogues",
        source_url="https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0178",
    )


def _aristotle_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        id=uuid.uuid4(),
        title="placeholder",
        author="Aristotle",
        source_class=SourceClass.PRIMARY_TEXT,
        collection="aristotle_corpus",
        source_url="https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0054",
    )


def test_perseus_tei_parser_preserves_stephanus_sections_and_speakers() -> None:
    parser = PerseusTeiParser(text_identifier="Meno")

    parsed = parser.parse(PERSEUS_TEI, _metadata())

    assert parsed.metadata.title == "Meno"
    assert parsed.extra_metadata == {
        "source_url": "https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0178",
    }
    assert [section.title for section in parsed.sections] == ["70a", "70b", "70c"]

    first_section = parsed.sections[0]
    assert first_section.location_ref is not None
    assert first_section.location_ref.system == LocationSystem.STEPHANUS
    assert first_section.location_ref.value == "70a"
    assert first_section.speaker is None
    assert first_section.text == (
        "Meno: Can you tell me, Socrates, whether virtue can be taught?\n\nSocrates: Not yet, Meno."
    )

    second_section = parsed.sections[1]
    assert second_section.location_ref is not None
    assert second_section.location_ref.value == "70b"
    assert second_section.speaker is None
    assert second_section.text == (
        "Socrates: I do not even know what virtue is.\n\nMeno: Then what shall we do?"
    )

    third_section = parsed.sections[2]
    assert third_section.location_ref is not None
    assert third_section.location_ref.value == "70c"
    assert third_section.speaker == "Socrates"
    assert third_section.text == "Socrates: We shall inquire together."


def test_section_chunker_propagates_perseus_source_metadata() -> None:
    parser = PerseusTeiParser(text_identifier="Meno")
    parsed = parser.parse(PERSEUS_TEI, _metadata())
    parsed.sections[0].text = " ".join(["Virtue can be taught"] * 30)

    chunker = SectionChunker()
    chunks = chunker.chunk(parsed, ChunkConfig(min_chunk_tokens=5))

    assert chunks
    assert chunks[0].extra_metadata == {
        "source_url": "https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0178",
    }


def test_perseus_tei_parser_preserves_bekker_ranges_for_treatises() -> None:
    parser = PerseusTeiParser(parse_mode="bekker_treatise")

    parsed = parser.parse(ARISTOTLE_TEI, _aristotle_metadata())

    assert parsed.metadata.title == "Nicomachean Ethics"
    assert parsed.extra_metadata == {
        "source_url": "https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0054",
    }
    assert [section.title for section in parsed.sections] == [
        "Book 1, Chapter 1",
        "Book 1, Chapter 2",
        "Book 1, Chapter 2",
    ]

    first_section = parsed.sections[0]
    assert first_section.location_ref is not None
    assert first_section.location_ref.system == LocationSystem.BEKKER
    assert first_section.location_ref.value == "1094a1"
    assert first_section.location_ref.range_end == "1094a20"
    assert first_section.text == (
        "Every art and every inquiry seems to aim at some good. "
        "Hence it has been said that the good is that at which all things aim."
    )

    second_section = parsed.sections[1]
    assert second_section.location_ref is not None
    assert second_section.location_ref.value == "1094a20"
    assert second_section.location_ref.range_end is None
    assert second_section.text == (
        "If there is an end of our actions that we wish for itself, this will be the good."
    )

    third_section = parsed.sections[2]
    assert third_section.location_ref is not None
    assert third_section.location_ref.value == "1094b1"
    assert third_section.text == (
        "Political science ordains which sciences should exist in the city."
    )
