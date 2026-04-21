from __future__ import annotations

from pathlib import Path

from plato_rag.ingestion.corpus import load_manifest, validate_manifest_entries

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEED_MANIFEST_PATH = _PROJECT_ROOT / "data" / "corpus_seed.json"


def test_repo_seed_manifest_is_valid_and_public_safe() -> None:
    entries = load_manifest(_SEED_MANIFEST_PATH)

    validate_manifest_entries(entries)


def test_public_iep_seed_spans_multiple_traditions() -> None:
    entries = load_manifest(_SEED_MANIFEST_PATH)

    iep_traditions = {
        entry.tradition
        for entry in entries
        if entry.collection == "iep" and entry.tradition is not None
    }

    assert {
        "ancient",
        "modern",
        "analytic",
        "continental",
        "political",
        "chinese",
        "buddhist",
        "medieval",
        "islamic",
        "jewish",
        "african",
    }.issubset(iep_traditions)


def test_public_iep_seed_includes_core_topic_entries() -> None:
    entries = load_manifest(_SEED_MANIFEST_PATH)

    iep_entry_ids = {entry.id for entry in entries if entry.collection == "iep"}

    assert {
        "epistemology-iep",
        "knowledge-iep",
        "epistemic-justification-iep",
        "causation-iep",
        "free-will-iep",
        "personal-identity-iep",
        "substance-iep",
        "metaethics-iep",
        "virtue-ethics-iep",
        "care-ethics-iep",
        "consequentialism-iep",
        "moral-realism-iep",
    }.issubset(iep_entry_ids)


def test_public_iep_seed_includes_language_mind_and_logic_entries() -> None:
    entries = load_manifest(_SEED_MANIFEST_PATH)

    iep_entry_ids = {entry.id for entry in entries if entry.collection == "iep"}

    assert {
        "philosophy-of-language-iep",
        "meaning-and-communication-iep",
        "truth-iep",
        "consciousness-iep",
        "intentionality-iep",
        "mental-causation-iep",
        "logical-consequence-iep",
        "modal-logic-iep",
        "propositional-logic-iep",
    }.issubset(iep_entry_ids)


def test_public_seed_includes_expanded_primary_text_entries() -> None:
    entries = load_manifest(_SEED_MANIFEST_PATH)

    entry_ids = {entry.id for entry in entries}

    assert {
        "crito",
        "theaetetus",
        "sophist",
        "phaedrus",
        "parmenides",
        "timaeus",
        "eudemian-ethics",
        "poetics",
    }.issubset(entry_ids)
