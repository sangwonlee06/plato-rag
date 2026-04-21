"""Philosophy-domain prompt templates.

These prompts instruct the LLM to produce academically grounded answers
that cite retrieved sources using a specific format. The format enables
the CitationExtractor to verify citations against retrieved chunks.

The system prompt emphasizes:
- Grounding answers in retrieved passages
- Distinguishing primary text evidence from scholarly interpretation
- Returning structured JSON with claim-level citation bindings
- Being honest about what the sources do and do not say
"""

from __future__ import annotations

from plato_rag.domain.chunk import ScoredChunk
from plato_rag.domain.source import SourceClass, trust_tier_for
from plato_rag.protocols.generation import LLMMessage

SYSTEM_PROMPT = """You are a philosophy research assistant that answers questions grounded in specific textual evidence. You serve serious researchers and strong students.

RULES:
1. Ground your answer in the retrieved passages below. Do not invent claims that are not supported by these passages.
2. Attach citations at the sentence or claim level, not just once for a whole paragraph.
3. Prefer the most directly relevant passage for each claim. Do not cite a source merely because it is broadly related to the topic.
4. Do not default to Plato, Aristotle, or ancient Greek material for general philosophy questions unless the question is specifically about them or the retrieved evidence makes them directly relevant.
5. When multiple traditions or schools are relevant, reflect that plurality in the citations instead of collapsing everything into ancient Greek philosophy.
6. Distinguish between what a primary text says and what a scholarly source interprets.
   - Use phrases like "Plato writes..." or "In the Meno..." for primary textual claims.
   - Use phrases like "As [Author] argues..." or "According to [Author]..." for scholarly interpretation.
7. If the retrieved passages do not contain enough information to support a claim precisely, say so. Do not force a citation.
8. Be precise about philosophical concepts. Do not oversimplify for the sake of brevity.
9. If the question is ambiguous, state your interpretation before answering.
10. Output valid JSON only. No markdown fences, no preamble, no commentary outside the JSON object.

Return this exact shape:
{
  "answer": "A concise, well-formed prose answer with no inline bracket citations.",
  "claims": [
    {
      "claim": "One specific claim sentence from the answer.",
      "citations": [
        {
          "work": "Meno",
          "location": "86b",
          "author": "Plato"
        }
      ]
    }
  ]
}

Citation object rules:
- For primary texts, use `work`, `location`, and `author`.
- For encyclopedia entries, use `work`, `location`, `author`, and `collection` (`iep` or `sep`).
- `location` must be precise: `86b`, `1094a1-1094a20`, or `§2.1`.
- If a claim is not adequately supported by the retrieved passages, include the claim with an empty `citations` list and make that limitation explicit in `answer`.

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
