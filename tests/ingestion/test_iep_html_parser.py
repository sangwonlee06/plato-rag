from __future__ import annotations

import uuid

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationSystem
from plato_rag.domain.source import SourceClass
from plato_rag.ingestion.chunkers.section import SectionChunker
from plato_rag.ingestion.parsers.iep_html import IepHtmlParser
from plato_rag.protocols.ingestion import ChunkConfig

IEP_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <title>Plato | Internet Encyclopedia of Philosophy</title>
  </head>
  <body>
    <article>
      <div class="entry-content">
        <h1>Plato (427-347 B.C.E.)</h1>
        <p>Plato is one of the most influential philosophers in the ancient world.</p>
        <p>His work spans ethics, epistemology, and metaphysics.</p>
        <h3>Table of Contents</h3>
        <ol>
          <li><a href="#H1">1. Biography</a></li>
          <li><a href="#H2">2. Plato's Writings</a></li>
        </ol>
        <h2><a name="H1"></a> 1. Biography</h2>
        <p>Plato was born into an aristocratic Athenian family.</p>
        <h3><a name="SH1a"></a> a. Birth</h3>
        <p>Ancient testimony places Plato's birth in the late fifth century BCE.</p>
        <h3><a name="SH1b"></a> b. Family</h3>
        <p>His relatives were deeply connected to Athenian political life.</p>
        <h2><a name="H2"></a> 2. Plato's Writings</h2>
        <p>The dialogues are normally grouped into early, middle, and late periods.</p>
        <h2><a name="H8"></a> 8. References and Further Reading</h2>
        <p>This should not appear in parsed sections.</p>
        <h3>Author Information</h3>
        <p>Thomas Brickhouse<br />Email: brickhouse@example.edu<br />Lynchburg College</p>
        <p>and</p>
        <p>Nicholas D. Smith<br />Email: ndsmith@example.edu<br />Lewis &amp; Clark College</p>
      </div>
    </article>
  </body>
</html>
"""


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        id=uuid.uuid4(),
        title="placeholder",
        author="placeholder",
        source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
        collection="iep",
        source_url="https://iep.utm.edu/plato/",
    )


def test_iep_html_parser_extracts_sections_and_metadata() -> None:
    parser = IepHtmlParser()

    parsed = parser.parse(IEP_HTML, _metadata())

    assert parsed.metadata.title == "Plato (427-347 B.C.E.)"
    assert parsed.metadata.author == "Thomas Brickhouse and Nicholas D. Smith"
    assert parsed.extra_metadata == {
        "entry_url": "https://iep.utm.edu/plato/",
    }

    assert [section.title for section in parsed.sections] == [
        "Preamble",
        "Biography",
        "Birth",
        "Family",
        "Plato's Writings",
    ]
    assert parsed.sections[0].location_ref is None
    assert parsed.sections[1].location_ref is not None
    assert parsed.sections[1].location_ref.system == LocationSystem.SECTION
    assert parsed.sections[1].location_ref.value == "1"
    assert parsed.sections[2].location_ref is not None
    assert parsed.sections[2].location_ref.value == "1.a"
    assert parsed.sections[3].location_ref is not None
    assert parsed.sections[3].location_ref.value == "1.b"
    assert parsed.sections[4].location_ref is not None
    assert parsed.sections[4].location_ref.value == "2"
    assert "References and Further Reading" not in [section.title for section in parsed.sections]
    assert "This should not appear" not in parsed.sections[-1].text


def test_section_chunker_propagates_iep_document_extra_metadata() -> None:
    parser = IepHtmlParser()
    parsed = parser.parse(IEP_HTML, _metadata())

    parsed.sections[1].text = " ".join(["Plato founded the Academy"] * 30)
    chunker = SectionChunker()

    chunks = chunker.chunk(parsed, ChunkConfig(min_chunk_tokens=5))

    assert chunks
    assert chunks[0].extra_metadata == {
        "entry_url": "https://iep.utm.edu/plato/",
    }
