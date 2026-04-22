"""Tests for the source trust model and LocationRef."""

import pytest

from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import (
    SOURCE_CLASS_REGISTRY,
    SourceClass,
    SourceExposure,
    collection_exposure,
    collection_source_class,
    is_high_trust,
    is_local_only_collection,
    local_only_collection_names,
    public_collection_names,
    trust_tier_for,
)


class TestSourceClassRegistry:
    def test_all_source_classes_registered(self) -> None:
        for sc in SourceClass:
            assert sc in SOURCE_CLASS_REGISTRY

    def test_trust_tiers_are_ordered(self) -> None:
        tiers = [info.trust_tier for info in SOURCE_CLASS_REGISTRY.values()]
        assert tiers == sorted(tiers)

    def test_primary_is_tier_1(self) -> None:
        assert trust_tier_for(SourceClass.PRIMARY_TEXT) == 1

    def test_reference_is_tier_2(self) -> None:
        assert trust_tier_for(SourceClass.REFERENCE_ENCYCLOPEDIA) == 2

    def test_is_high_trust(self) -> None:
        assert is_high_trust(SourceClass.PRIMARY_TEXT)
        assert is_high_trust(SourceClass.REFERENCE_ENCYCLOPEDIA)
        assert not is_high_trust(SourceClass.PEER_REVIEWED)
        assert not is_high_trust(SourceClass.CURATED_BIBLIOGRAPHY)


class TestCollectionRegistry:
    def test_sep_is_reference_encyclopedia(self) -> None:
        assert collection_source_class("sep") == SourceClass.REFERENCE_ENCYCLOPEDIA

    def test_sep_is_local_only(self) -> None:
        assert collection_exposure("sep") == SourceExposure.LOCAL_ONLY
        assert is_local_only_collection("sep")

    def test_iep_is_reference_encyclopedia(self) -> None:
        assert collection_source_class("iep") == SourceClass.REFERENCE_ENCYCLOPEDIA

    def test_platonic_dialogues_is_primary(self) -> None:
        assert collection_source_class("platonic_dialogues") == SourceClass.PRIMARY_TEXT
        assert collection_exposure("platonic_dialogues") == SourceExposure.PUBLIC

    def test_modern_primary_collections_are_public(self) -> None:
        assert collection_source_class("cartesian_meditations") == SourceClass.PRIMARY_TEXT
        assert collection_exposure("cartesian_meditations") == SourceExposure.PUBLIC
        assert collection_source_class("hume_enquiry") == SourceClass.PRIMARY_TEXT
        assert collection_exposure("hume_enquiry") == SourceExposure.PUBLIC

    def test_public_collection_names_excludes_sep(self) -> None:
        assert "sep" not in public_collection_names()
        assert "platonic_dialogues" in public_collection_names()
        assert "cartesian_meditations" in public_collection_names()
        assert "hume_enquiry" in public_collection_names()

    def test_local_only_collection_names_contains_sep(self) -> None:
        assert local_only_collection_names() == {"sep"}

    def test_unknown_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown collection"):
            collection_source_class("nonexistent")


class TestLocationRef:
    def test_stephanus_display(self) -> None:
        ref = LocationRef(system=LocationSystem.STEPHANUS, value="86b")
        assert ref.display() == "86b"

    def test_stephanus_range_display(self) -> None:
        ref = LocationRef(system=LocationSystem.STEPHANUS, value="82b", range_end="85b")
        assert ref.display() == "82b\u201385b"

    def test_section_display_with_prefix(self) -> None:
        ref = LocationRef(system=LocationSystem.SECTION, value="2.1")
        assert ref.display_with_prefix() == "\u00a72.1"

    def test_matches_value_exact(self) -> None:
        ref = LocationRef(system=LocationSystem.STEPHANUS, value="86b")
        assert ref.matches_value("86b")
        assert ref.matches_value("86B")
        assert not ref.matches_value("86c")

    def test_matches_value_section_prefix(self) -> None:
        ref = LocationRef(system=LocationSystem.SECTION, value="2.1")
        assert ref.matches_value("2.1")
        assert ref.matches_value("Section 2.1")
        assert ref.matches_value("\u00a72.1")

    def test_page_display_with_prefix(self) -> None:
        ref = LocationRef(system=LocationSystem.PAGE, value="42")
        assert ref.display_with_prefix() == "p. 42"
