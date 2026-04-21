"""Lightweight philosophy-domain profiling for retrieval and citation scoring.

This is intentionally heuristic rather than taxonomic perfection. The goal is
to make retrieval and citation behavior more discipline-aware with stable,
inspectable rules that can be tested and refined.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from plato_rag.domain.chunk import ChunkData

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_PHRASE_BOUNDARY_TEMPLATE = r"(?<!\w){phrase}(?!\w)"

_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "among",
    "and",
    "any",
    "are",
    "because",
    "been",
    "being",
    "between",
    "both",
    "does",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "much",
    "only",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ethics": (
        "ethics",
        "moral",
        "morality",
        "virtue",
        "good",
        "evil",
        "normative",
        "normativity",
        "deontology",
        "consequentialism",
        "metaethics",
    ),
    "metaphysics": (
        "metaphysics",
        "being",
        "ontology",
        "ontological",
        "substance",
        "causation",
        "identity",
        "modality",
        "truthmaker",
    ),
    "epistemology": (
        "epistemology",
        "knowledge",
        "justification",
        "skepticism",
        "belief",
        "evidence",
        "testimony",
        "perception",
        "certainty",
    ),
    "philosophy_of_mind": (
        "mind",
        "mental",
        "consciousness",
        "intentionality",
        "qualia",
        "self-consciousness",
        "representation",
        "mental causation",
        "propositional attitudes",
        "dualism",
    ),
    "philosophy_of_language": (
        "language",
        "meaning",
        "reference",
        "truth",
        "semantics",
        "pragmatics",
        "communication",
        "compositionality",
    ),
    "logic": (
        "logic",
        "argument",
        "inference",
        "validity",
        "logical consequence",
        "modal logic",
        "propositional logic",
        "deduction",
    ),
    "political_philosophy": (
        "political",
        "politics",
        "justice",
        "citizenship",
        "law",
        "state",
        "liberalism",
        "obligation",
    ),
    "aesthetics": (
        "aesthetics",
        "beauty",
        "art",
        "poetics",
        "tragedy",
        "mimesis",
    ),
}

_TRADITION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "analytic": ("analytic", "frege", "russell", "wittgenstein", "kripke", "quine"),
    "continental": (
        "continental",
        "phenomenology",
        "heidegger",
        "husserl",
        "nietzsche",
        "sartre",
        "de beauvoir",
    ),
    "ancient": ("ancient", "plato", "aristotle", "socrates", "stoic", "presocratic"),
    "classical_greek": ("plato", "aristotle", "socrates", "athenian", "stephanus", "bekker"),
    "medieval": ("medieval", "aquinas", "augustine", "scholastic"),
    "modern": ("modern", "descartes", "hume", "locke", "spinoza", "leibniz", "kant"),
    "contemporary": ("contemporary", "current debate", "recent debate"),
    "chinese": ("chinese", "confucian", "daoist", "mencius", "xunzi", "zhuangzi", "mozi"),
    "buddhist": ("buddhist", "buddha", "nagarjuna", "madhyamaka"),
    "islamic": ("islamic", "avicenna", "ibn sina", "averroes"),
    "jewish": ("jewish", "maimonides"),
    "african": ("african", "ethnophilosophy", "sage philosophy", "ubuntu"),
    "cross_tradition": ("cross tradition",),
}

_PERIOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "classical_greek": ("classical greek", "ancient greece"),
    "early_modern": ("early modern",),
    "18th_century": ("18th century", "eighteenth century"),
    "19th_century": ("19th century", "nineteenth century"),
    "20th_century": ("20th century", "twentieth century"),
    "20th_21st_century": ("20th", "21st", "contemporary"),
    "historical_and_contemporary": ("historical", "contemporary"),
    "contemporary": ("contemporary", "current debate"),
}

_PHILOSOPHER_METADATA: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "plato": ("ancient", "classical_greek", ("metaphysics", "epistemology", "ethics")),
    "aristotle": ("ancient", "classical_greek", ("metaphysics", "ethics", "logic")),
    "kant": ("modern", "18th_century", ("metaphysics", "epistemology", "ethics")),
    "hume": ("modern", "early_modern", ("epistemology", "metaphysics")),
    "heidegger": ("continental", "20th_century", ("metaphysics", "philosophy_of_mind")),
    "husserl": ("continental", "20th_century", ("philosophy_of_mind", "epistemology")),
    "frege": ("analytic", "19th_20th_century", ("logic", "philosophy_of_language")),
    "wittgenstein": ("analytic", "20th_century", ("philosophy_of_language", "logic")),
    "rawls": ("political", "20th_century", ("political_philosophy", "ethics")),
    "confucius": ("chinese", "classical_chinese", ("ethics", "political_philosophy")),
    "mencius": ("chinese", "classical_chinese", ("ethics", "political_philosophy")),
    "nagarjuna": ("buddhist", "classical_indian", ("metaphysics",)),
}


@dataclass(frozen=True)
class PhilosophyProfile:
    topics: set[str] = field(default_factory=set)
    traditions: set[str] = field(default_factory=set)
    periods: set[str] = field(default_factory=set)
    philosophers: set[str] = field(default_factory=set)


def profile_text(
    text: str,
    *,
    tradition: str | None = None,
    period: str | None = None,
    topics: list[str] | None = None,
) -> PhilosophyProfile:
    normalized = _normalize_text(text)

    detected_topics = _detect_labels(normalized, _TOPIC_KEYWORDS)
    detected_traditions = _detect_labels(normalized, _TRADITION_KEYWORDS)
    detected_periods = _detect_labels(normalized, _PERIOD_KEYWORDS)
    philosophers: set[str] = set()

    for philosopher, (phil_tradition, phil_period, phil_topics) in _PHILOSOPHER_METADATA.items():
        if _contains_phrase(normalized, philosopher):
            philosophers.add(philosopher)
            detected_traditions.add(phil_tradition)
            detected_periods.add(phil_period)
            detected_topics.update(phil_topics)

    if tradition:
        detected_traditions.add(tradition)
    if period:
        detected_periods.add(period)
    if topics:
        detected_topics.update(_normalize_label(topic) for topic in topics if topic)

    return PhilosophyProfile(
        topics={label for label in detected_topics if label},
        traditions={label for label in detected_traditions if label},
        periods={label for label in detected_periods if label},
        philosophers=philosophers,
    )


def profile_chunk(chunk: ChunkData) -> PhilosophyProfile:
    metadata = chunk.extra_metadata or {}
    raw_topics = metadata.get("topics")
    topic_list = raw_topics if isinstance(raw_topics, list) else None
    raw_tradition = metadata.get("tradition")
    raw_period = metadata.get("period")

    composite_text = " ".join(
        part
        for part in (
            chunk.work_title,
            chunk.author,
            chunk.section_title or "",
            chunk.text,
        )
        if part
    )

    return profile_text(
        composite_text,
        tradition=str(raw_tradition) if raw_tradition else None,
        period=str(raw_period) if raw_period else None,
        topics=[str(item) for item in topic_list] if topic_list else None,
    )


def significant_tokens(text: str) -> set[str]:
    tokens = {
        _normalize_label(match.group(0)) for match in _WORD_PATTERN.finditer(_normalize_text(text))
    }
    return {token for token in tokens if len(token) >= 4 and token not in _STOPWORDS}


def is_explicit_ancient_query(profile: PhilosophyProfile) -> bool:
    if {"ancient", "classical_greek"} & profile.traditions:
        return True
    return bool(profile.philosophers & {"plato", "aristotle", "socrates"})


def _detect_labels(normalized_text: str, mapping: dict[str, tuple[str, ...]]) -> set[str]:
    labels: set[str] = set()
    for label, phrases in mapping.items():
        if any(_contains_phrase(normalized_text, phrase) for phrase in phrases):
            labels.add(label)
    return labels


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    escaped_phrase = re.escape(_normalize_label(phrase))
    pattern = _PHRASE_BOUNDARY_TEMPLATE.format(phrase=escaped_phrase)
    return re.search(pattern, normalized_text) is not None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = normalized.casefold()
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_label(value: str) -> str:
    normalized = _normalize_text(value)
    return normalized.replace(" ", "_")
