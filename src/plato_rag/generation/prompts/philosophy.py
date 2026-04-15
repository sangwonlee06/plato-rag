"""Philosophy-domain prompt templates.

These prompts instruct the LLM to produce academically grounded answers
that cite retrieved sources using a specific format. The format enables
the CitationExtractor to verify citations against retrieved chunks.

The system prompt emphasizes:
- Grounding answers in retrieved passages
- Distinguishing primary text evidence from scholarly interpretation
- Using the citation format [Work Location] for verifiable references
- Being honest about what the sources do and do not say
"""

from __future__ import annotations

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import SourceClass, trust_tier_for
from plato_rag.protocols.generation import LLMMessage

SYSTEM_PROMPT = """You are a philosophy research assistant that answers questions grounded in specific textual evidence. You serve serious researchers and strong students.

RULES:
1. Ground your answer in the retrieved passages below. Do not invent claims that are not supported by these passages.
2. When you reference a specific passage, cite it using this format: [Work Location]
   - For primary texts: [Meno 86b] or [Republic 514a]
   - For encyclopedia entries: [Author, SEP §Section] or [Author, IEP §Section]
3. Distinguish between what a primary text says and what a scholarly source interprets.
   - Use phrases like "Plato writes..." or "In the Meno..." for primary textual claims.
   - Use phrases like "As [Author] argues..." or "According to [Author]..." for scholarly interpretation.
4. If the retrieved passages do not contain enough information to answer the question well, say so honestly. Do not hallucinate references or fabricate textual claims.
5. Be precise about philosophical concepts. Do not oversimplify for the sake of brevity.
6. If the question is ambiguous, state your interpretation before answering.

You are not a general chatbot. You are a source-grounded philosophy research tool."""


def build_query_messages(
    question: str,
    chunks: list[ScoredChunk],
    conversation_history: list[tuple[str, str]] | None = None,
) -> list[LLMMessage]:
    """Build the message sequence for the LLM."""
    messages: list[LLMMessage] = [LLMMessage(role="system", content=SYSTEM_PROMPT)]

    # Add conversation history if present
    if conversation_history:
        for role, content in conversation_history:
            messages.append(LLMMessage(role=role, content=content))

    # Build context block from retrieved chunks, ordered by trust tier
    sorted_chunks = sorted(chunks, key=lambda sc: trust_tier_for(sc.chunk.source_class))

    context_parts: list[str] = []
    for sc in sorted_chunks:
        chunk = sc.chunk
        loc = chunk.location_display or "no specific location"
        source_label = _source_label(chunk.source_class, chunk.collection)
        header = f"[{source_label}] {chunk.work_title} by {chunk.author}, {loc}"
        if chunk.speaker:
            header += f" (speaker: {chunk.speaker})"
        context_parts.append(f"---\n{header}\n\n{chunk.text}")

    context_block = "\n\n".join(context_parts)

    user_content = f"""Retrieved passages:

{context_block}

---

Question: {question}"""

    messages.append(LLMMessage(role="user", content=user_content))
    return messages


def _source_label(source_class: SourceClass, collection: str) -> str:
    if source_class == SourceClass.PRIMARY_TEXT:
        return "PRIMARY SOURCE"
    if source_class == SourceClass.REFERENCE_ENCYCLOPEDIA:
        label_map = {"sep": "SEP", "iep": "IEP"}
        return label_map.get(collection, "REFERENCE")
    if source_class == SourceClass.PEER_REVIEWED:
        return "SCHOLARSHIP"
    return "SOURCE"
