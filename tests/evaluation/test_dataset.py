from __future__ import annotations

from pathlib import Path

from plato_rag.evaluation import load_dataset


def test_public_seed_dataset_loads_and_has_real_coverage() -> None:
    dataset = load_dataset(Path("data/evaluation/public_seed.yaml"))

    assert dataset.name == "public-seed-philosophy-rag"
    assert len(dataset.cases) >= 10
    assert any("primary" in case.tags for case in dataset.cases)
    assert any("reference" in case.tags for case in dataset.cases)


def test_public_seed_dataset_case_ids_are_distinct() -> None:
    dataset = load_dataset(Path("data/evaluation/public_seed.yaml"))

    ids = [case.id for case in dataset.cases]
    assert len(ids) == len(set(ids))
