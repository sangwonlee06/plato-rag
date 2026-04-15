"""Section-aware chunker.

Respects section boundaries from the parser. If a section exceeds
max_chunk_tokens, splits at paragraph boundaries within the section
while preserving location reference and speaker metadata on each sub-chunk.
"""

from __future__ import annotations

import re

import tiktoken

from plato_rag.protocols.ingestion import ChunkConfig, ParsedDocument, ParsedSection, RawChunk


class SectionChunker:
    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk(self, document: ParsedDocument, config: ChunkConfig) -> list[RawChunk]:
        chunks: list[RawChunk] = []
        index = 0
        for section in document.sections:
            section_chunks = self._chunk_section(section, config, index)
            for c in section_chunks:
                chunks.append(c)
                index += 1
        return chunks

    def _chunk_section(
        self, section: ParsedSection, config: ChunkConfig, start_index: int
    ) -> list[RawChunk]:
        text = section.text.strip()
        if not text:
            return []

        token_count = len(self._enc.encode(text))

        if token_count <= config.max_chunk_tokens:
            if token_count < config.min_chunk_tokens:
                return []
            return [RawChunk(
                text=text,
                location_ref=section.location_ref,
                section_title=section.title,
                speaker=section.speaker,
                interlocutor=section.interlocutor,
                chunk_index=start_index,
                token_count=token_count,
            )]

        # Split oversized sections at paragraph boundaries
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = re.split(r'(?<=[.!?])\s+', text)
            paragraphs = [s for s in paragraphs if s.strip()]

        chunks: list[RawChunk] = []
        current_lines: list[str] = []
        current_tokens = 0
        idx = start_index

        for para in paragraphs:
            para_tokens = len(self._enc.encode(para))
            if current_tokens + para_tokens > config.max_chunk_tokens and current_lines:
                chunks.append(RawChunk(
                    text="\n\n".join(current_lines),
                    location_ref=section.location_ref,
                    section_title=section.title,
                    speaker=section.speaker,
                    interlocutor=section.interlocutor,
                    chunk_index=idx,
                    token_count=current_tokens,
                ))
                idx += 1
                current_lines = []
                current_tokens = 0
            current_lines.append(para)
            current_tokens += para_tokens

        if current_lines and current_tokens >= config.min_chunk_tokens:
            chunks.append(RawChunk(
                text="\n\n".join(current_lines),
                location_ref=section.location_ref,
                section_title=section.title,
                speaker=section.speaker,
                interlocutor=section.interlocutor,
                chunk_index=idx,
                token_count=current_tokens,
            ))

        return chunks
