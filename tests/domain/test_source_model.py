"""Tests for the source trust model and LocationRef."""

from plato_rag.domain.location import LocationRef, LocationSystem
from plato_rag.domain.source import (
    COLLECTION_REGISTRY,
    SOURCE_CLASS_REGISTRY,
    SourceClass,
    collection_source_class,
    is_high_trust,
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

    def test_iep_is_reference_encyclopedia(self) -> None:
        assert collection_source_class("iep") == SourceClass.REFERENCE_ENCYCLOPEDIA

    def test_platonic_dialogues_is_primary(self) -> None:
        assert collection_source_class("platonic_dialogues") == SourceClass.PRIMARY_TEXT

    def test_unknown_collection_raises(self) -> None:
        import pytest
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
