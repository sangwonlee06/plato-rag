from __future__ import annotations

from plato_rag.domain.location import LocationRef, LocationSystem


def test_section_locations_match_prefix_variants() -> None:
    location = LocationRef(system=LocationSystem.SECTION, value="2.1")

    assert location.matches_value("2.1")
    assert location.matches_value("Section 2.1")
    assert location.matches_value("Sec. 2.1")
    assert location.matches_value("\u00a72.1")


def test_stephanus_locations_overlap_ranges() -> None:
    location = LocationRef(system=LocationSystem.STEPHANUS, value="82b")

    assert location.overlaps_raw_value("82b-85b")
    assert location.overlaps_raw_value("82b-83a")
    assert not location.overlaps_raw_value("83b-85b")


def test_bekker_ranges_contain_single_line_citations() -> None:
    location = LocationRef(
        system=LocationSystem.BEKKER,
        value="1094a1",
        range_end="1094a20",
    )

    assert location.matches_value("1094a1")
    assert location.overlaps_raw_value("1094a5")
    assert location.overlaps_raw_value("1094a1-1094a20")
    assert not location.overlaps_raw_value("1094b1")
