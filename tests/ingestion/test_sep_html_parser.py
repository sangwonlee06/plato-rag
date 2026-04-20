from __future__ import annotations

import uuid

from plato_rag.domain.document import DocumentMetadata
from plato_rag.domain.location import LocationSystem
from plato_rag.domain.source import SourceClass
from plato_rag.ingestion.chunkers.section import SectionChunker
from plato_rag.local_only.sep_html import SepHtmlParser
from plato_rag.protocols.ingestion import ChunkConfig

SEP_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <meta property="citation_title" content="Plato's Middle Period Metaphysics and Epistemology" />
    <meta property="citation_author" content="Silverman, Allan" />
    <meta name="DCTERMS.modified" content="2014-07-14" />
  </head>
  <body>
      <div id="aueditable">
        <h1>Plato's Middle Period Metaphysics and Epistemology</h1>
      <div id="pubinfo">
        <em>First published Mon Jun 9, 2003; substantive revision Mon Jul 14, 2014</em>
      </div>
      <div id="preamble">
        <p>Plato's metaphysics and epistemology are tightly linked in the middle dialogues.</p>
      </div>
      <div id="toc">
        <ul>
          <li><a href="#1">1. The Background to Plato's Metaphysics</a></li>
        </ul>
      </div>
      <h2><a name="1">1. The Background to Plato's Metaphysics</a></h2>
      <p>The forms are introduced as explanatory entities in the middle dialogues.</p>
      <h3 id="1.1">1.1 The Meno and Recollection</h3>
      <p>
        The argument from recollection supports the claim that learning can be
        a recovery of knowledge.
      </p>
      <h2><a name="Bib">Bibliography</a></h2>
      <p>This should not appear in parsed sections.</p>
    </div>
  </body>
</html>
"""


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        id=uuid.uuid4(),
        title="placeholder",
        author="placeholder",
        source_class=SourceClass.REFERENCE_ENCYCLOPEDIA,
        collection="sep",
        source_url="https://plato.stanford.edu/entries/plato-metaphysics/",
    )


def test_sep_html_parser_extracts_sections_and_metadata() -> None:
    parser = SepHtmlParser()

    parsed = parser.parse(SEP_HTML, _metadata())

    assert parsed.metadata.title == "Plato's Middle Period Metaphysics and Epistemology"
    assert parsed.metadata.author == "Allan Silverman"
    assert parsed.extra_metadata == {
        "entry_url": "https://plato.stanford.edu/entries/plato-metaphysics/",
        "last_updated": "2014-07-14",
    }

    assert [section.title for section in parsed.sections] == [
        "Preamble",
        "The Background to Plato's Metaphysics",
        "The Meno and Recollection",
    ]
    assert parsed.sections[0].location_ref is None
    assert parsed.sections[1].location_ref is not None
    assert parsed.sections[1].location_ref.system == LocationSystem.SECTION
    assert parsed.sections[1].location_ref.value == "1"
    assert parsed.sections[2].location_ref is not None
    assert parsed.sections[2].location_ref.value == "1.1"
    assert "Bibliography" not in [section.title for section in parsed.sections]
    assert "This should not appear" not in parsed.sections[-1].text


def test_section_chunker_propagates_document_extra_metadata() -> None:
    parser = SepHtmlParser()
    parsed = parser.parse(SEP_HTML, _metadata())

    parsed.sections[1].text = " ".join(["forms explain participation"] * 30)
    chunker = SectionChunker()

    chunks = chunker.chunk(parsed, ChunkConfig(min_chunk_tokens=5))

    assert chunks
    assert chunks[0].extra_metadata == {
        "entry_url": "https://plato.stanford.edu/entries/plato-metaphysics/",
        "last_updated": "2014-07-14",
    }
