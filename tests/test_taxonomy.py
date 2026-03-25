"""
Tests for _taxonomy.py: CSV loading, deduplication, node counts,
embed text construction, parent-child map correctness, range dict.
"""

from __future__ import annotations

import pytest

from semtax._taxonomy import (
    LEVEL_CLASS,
    LEVEL_COMMODITY,
    LEVEL_FAMILY,
    LEVEL_SEGMENT,
    TaxonomyNode,
    TaxonomyStore,
    build_embed_text,
    load_unspsc,
)


# ---------------------------------------------------------------------------
# build_embed_text
# ---------------------------------------------------------------------------


def test_embed_text_segment_with_definition():
    node = TaxonomyNode("10000000", "Live Plant", "Live plants def", LEVEL_SEGMENT, None)
    text = build_embed_text(node, "Live Plant")
    assert text == "Live Plant — Live plants def"


def test_embed_text_segment_empty_definition():
    node = TaxonomyNode("10000000", "Live Plant", "", LEVEL_SEGMENT, None)
    text = build_embed_text(node, "Live Plant")
    assert text == "Live Plant"
    assert "—" not in text


def test_embed_text_family_path_separator():
    node = TaxonomyNode("10100000", "Live animals", "Animals def", LEVEL_FAMILY, "10000000")
    text = build_embed_text(node, "Live Plant > Live animals")
    assert ">" in text
    assert text.startswith("Live Plant > Live animals")


def test_embed_text_class_three_levels():
    node = TaxonomyNode("10101500", "Livestock", "Farm animals", LEVEL_CLASS, "10100000")
    prefix = "Live Plant > Live animals > Livestock"
    text = build_embed_text(node, prefix)
    assert text.startswith(prefix)
    assert "Farm animals" in text


def test_embed_text_commodity_includes_synonym():
    node = TaxonomyNode(
        "10101501", "Cats", "Domestic felines",
        LEVEL_COMMODITY, "10101500",
        synonym="cat", acronym="",
    )
    text = build_embed_text(node, "Live Plant > Live animals > Livestock > Cats")
    assert "Also known as: cat" in text
    assert "Acronyms" not in text  # acronym is empty


def test_embed_text_commodity_includes_acronym():
    node = TaxonomyNode(
        "10101502", "Dogs", "Domestic canines",
        LEVEL_COMMODITY, "10101500",
        synonym="", acronym="K9",
    )
    text = build_embed_text(node, "Live Plant > Live animals > Livestock > Dogs")
    assert "Acronyms: K9" in text
    assert "Also known as" not in text  # synonym is empty


def test_embed_text_commodity_both_empty():
    node = TaxonomyNode(
        "10101503", "Horses", "Equines",
        LEVEL_COMMODITY, "10101500",
        synonym="", acronym="",
    )
    text = build_embed_text(node, "... > Horses")
    assert "Also known as" not in text
    assert "Acronyms" not in text
    assert "." not in text or "Equines" in text  # no trailing ". Also known as: ."


def test_embed_text_commodity_whitespace_only_synonym_ignored():
    node = TaxonomyNode(
        "10101504", "Sheep", "Ovines",
        LEVEL_COMMODITY, "10101500",
        synonym="   ", acronym="  ",
    )
    text = build_embed_text(node, "... > Sheep")
    assert "Also known as" not in text
    assert "Acronyms" not in text


# ---------------------------------------------------------------------------
# load_unspsc — structural invariants
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store() -> TaxonomyStore:
    return load_unspsc()


def test_load_unspsc_returns_taxonomy_store(store):
    assert isinstance(store, TaxonomyStore)


def test_segment_count_reasonable(store):
    count = len(store.segments)
    assert 40 <= count <= 80, f"Unexpected segment count: {count}"


def test_family_count_reasonable(store):
    count = len(store.families)
    assert 300 <= count <= 700, f"Unexpected family count: {count}"


def test_class_count_reasonable(store):
    count = len(store.classes)
    assert 700 <= count <= 15000, f"Unexpected class count: {count}"


def test_commodity_count_reasonable(store):
    count = len(store.commodities)
    assert 100_000 <= count <= 200_000, f"Unexpected commodity count: {count}"


def test_no_duplicate_segment_codes(store):
    codes = [n.code for n in store.segments]
    assert len(set(codes)) == len(codes)


def test_no_duplicate_family_codes(store):
    codes = [n.code for n in store.families]
    assert len(set(codes)) == len(codes)


def test_no_duplicate_class_codes(store):
    codes = [n.code for n in store.classes]
    assert len(set(codes)) == len(codes)


def test_no_duplicate_commodity_codes(store):
    codes = [n.code for n in store.commodities]
    assert len(set(codes)) == len(codes)


def test_segment_embed_texts_aligned(store):
    assert len(store.segment_embed_texts) == len(store.segments)


def test_family_embed_texts_aligned(store):
    assert len(store.family_embed_texts) == len(store.families)


def test_class_embed_texts_aligned(store):
    assert len(store.class_embed_texts) == len(store.classes)


def test_commodity_embed_texts_aligned(store):
    assert len(store.commodity_embed_texts) == len(store.commodities)


def test_families_by_segment_completeness(store):
    """Every family should be reachable from its parent segment."""
    all_fam_codes = {n.code for n in store.families}
    reachable = set()
    for fams in store.families_by_segment.values():
        for f in fams:
            reachable.add(f.code)
    assert all_fam_codes == reachable


def test_classes_by_family_completeness(store):
    all_cls_codes = {n.code for n in store.classes}
    reachable = set()
    for clss in store.classes_by_family.values():
        for c in clss:
            reachable.add(c.code)
    assert all_cls_codes == reachable


def test_commodity_range_dict_covers_all_commodities(store):
    """The union of all ranges should cover the entire commodity list exactly."""
    total = 0
    for start, end in store.class_code_to_commodity_range.values():
        assert start < end
        total += end - start
    assert total == len(store.commodities)


def test_commodity_range_contiguous_within_class(store):
    """Within each range, all commodities should belong to the same class."""
    for cls_code, (start, end) in store.class_code_to_commodity_range.items():
        for com in store.commodities[start:end]:
            assert com.parent_code == cls_code, (
                f"Commodity {com.code} in range for {cls_code} "
                f"has parent {com.parent_code}"
            )


def test_segment_embed_text_has_label(store):
    """Every segment embed text should start with the segment label."""
    for node, text in zip(store.segments, store.segment_embed_texts):
        assert node.label in text, f"Segment label missing from embed text: {text[:80]}"


def test_class_embed_text_has_path_separator(store):
    """Class embed texts should contain '>' path separators."""
    # Check a sample (first 20) to avoid slow full scan
    for text in store.class_embed_texts[:20]:
        assert ">" in text, f"No path separator in class embed text: {text[:80]}"
