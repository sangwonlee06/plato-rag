"""Citation extraction and verification.

LIMITATIONS (current implementation):
- Regex-based parsing that handles only [Work Location] and [Author, SEP §Section] formats
- Exact or substring matching only — no fuzzy matching for typos or reformatting
- Cannot match range references like [Meno 82b-85b] to individual chunks at 82b, 83a, etc.
- Cannot detect inline citations (italicized work names in prose)
- Silent fallback: if a citation doesn't match any known format, it's treated as a work name
- Substring location matching may produce false positives (e.g., "86b" matches "286b")

This is adequate for an MVP where the LLM is instructed to use a specific
citation format and the retrieval set is small. It is not adequate for
production use where citation fidelity must be verified at scale.
"""

from __future__ import annotations

import re

from plato_rag.domain.chunk import ChunkData
from plato_rag.domain.source import collection_exposure
from plato_rag.protocols.generation import ExtractedCitation

CITATION_PATTERN = re.compile(r"\[(?P<content>[^\]]+)\]")


class BasicCitationExtractor:
    """Extracts citations from LLM output via regex and matches them to chunks.

    Only verifies that a citation references a chunk that was actually
    retrieved. Does not verify that the cited passage supports the claim
    being made — that would require deeper semantic analysis.
    """

    def extract(
        self,
        generated_text: str,
        retrieved_chunks: list[ChunkData],
    ) -> list[ExtractedCitation]:
        raw_citations = self._parse_citations(generated_text)
        verified: list[ExtractedCitation] = []

        for work, location in raw_citations:
            match = self._match_to_chunk(work, location, retrieved_chunks)
            if match:
                verified.append(
                    ExtractedCitation(
                        work=match.work_title,
                        location=match.location_ref.display() if match.location_ref else location,
                        excerpt=match.text[:200] if match.text else None,
                        matched_chunk_id=match.id,
                        is_grounded=True,
                        source_class=match.source_class,
                        collection=match.collection,
                        source_exposure=collection_exposure(match.collection),
                        author=match.author,
                        access_url=(
                            match.extra_metadata.get("entry_url") if match.extra_metadata else None
                        ),
                    )
                )
            else:
                verified.append(
                    ExtractedCitation(
                        work=work,
                        location=location,
                        is_grounded=False,
                    )
                )

        return verified

    def _parse_citations(self, text: str) -> list[tuple[str, str | None]]:
        """Extract (work, location) from [bracketed] markers. Limited format support."""
        results: list[tuple[str, str | None]] = []
        seen: set[str] = set()

        for match in CITATION_PATTERN.finditer(text):
            content = match.group("content").strip()
            if content in seen:
                continue
            seen.add(content)

            # Try "Author, SEP §Section" or "Author, IEP §Section"
            sep_match = re.match(r"(.+?),\s*(SEP|IEP)\s*§?\s*(.+)", content)
            if sep_match:
                results.append((sep_match.group(2), sep_match.group(3).strip()))
                continue

            # Try "Work Location" where location starts with a digit or §
            parts = content.rsplit(None, 1)
            if len(parts) == 2 and re.match(r"[\d§p]", parts[1]):
                results.append((parts[0], parts[1]))
                continue

            # Fallback: entire content is the work name, no location
            results.append((content, None))

        return results

    def _match_to_chunk(
        self, work: str, location: str | None, chunks: list[ChunkData]
    ) -> ChunkData | None:
        """Simple matching: work title substring + location value comparison."""
        work_lower = work.strip().lower()

        for chunk in chunks:
            # Match by work title (substring either direction)
            title_match = (
                work_lower in chunk.work_title.lower() or chunk.work_title.lower() in work_lower
            )
            # Match by collection for SEP/IEP references
            collection_match = work_lower in ("sep", "iep") and chunk.collection == work_lower

            if not (title_match or collection_match):
                continue

            if not location:
                return chunk

            # Use LocationRef.matches_value if available
            if chunk.location_ref and chunk.location_ref.matches_value(location):
                return chunk

        return None
