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


def test_malformed_output_dataset_loads_with_generation_fixtures() -> None:
    dataset = load_dataset(Path("data/evaluation/malformed_output.yaml"))

    assert dataset.name == "public-seed-philosophy-rag-malformed-output"
    assert len(dataset.cases) >= 2
    assert all(case.generation_fixture is not None for case in dataset.cases)
    assert any("malformed_output" in case.tags for case in dataset.cases)
