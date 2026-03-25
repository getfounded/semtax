"""
Shared fixtures for the semtax test suite.

fake_embed is a deterministic 16-dim embedding callable — no model download
needed in CI.  All classifier tests use this instead of a real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from semtax._ambiguity import AmbiguityConfig
from semtax._result import LevelResult
from semtax._taxonomy import (
    LEVEL_CLASS,
    LEVEL_COMMODITY,
    LEVEL_FAMILY,
    LEVEL_SEGMENT,
    TaxonomyNode,
    TaxonomyStore,
    build_embed_text,
)


# ---------------------------------------------------------------------------
# Deterministic fake embedding callable
# ---------------------------------------------------------------------------


def fake_embed(texts: list[str]) -> list[list[float]]:
    """
    Returns seeded random 16-dim float vectors.
    Deterministic: same text always gets the same vector.
    """
    rng = np.random.default_rng(seed=42)
    vecs = rng.random((len(texts), 16)).astype(np.float32)
    return vecs.tolist()


# ---------------------------------------------------------------------------
# Minimal TaxonomyStore fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_store() -> TaxonomyStore:
    """
    A small hand-crafted TaxonomyStore with:
      2 Segments  (SEG1, SEG2)
      2 Families  per Segment  (FAM11, FAM12, FAM21, FAM22)
      1 Class     per Family   (CLS111, CLS121, CLS211, CLS221)
      2 Commodities per Class  (8 total)

    All embed texts are short strings; they are NOT semantically meaningful
    (fake_embed ignores content).  This store is used for unit-testing
    classifier logic, reconciliation rules, and range slice correctness.
    """
    seg1 = TaxonomyNode("SEG1", "Segment One", "First segment", LEVEL_SEGMENT, None)
    seg2 = TaxonomyNode("SEG2", "Segment Two", "Second segment", LEVEL_SEGMENT, None)

    fam11 = TaxonomyNode("FAM11", "Family 1-1", "Fam 11 def", LEVEL_FAMILY, "SEG1")
    fam12 = TaxonomyNode("FAM12", "Family 1-2", "Fam 12 def", LEVEL_FAMILY, "SEG1")
    fam21 = TaxonomyNode("FAM21", "Family 2-1", "Fam 21 def", LEVEL_FAMILY, "SEG2")
    fam22 = TaxonomyNode("FAM22", "Family 2-2", "Fam 22 def", LEVEL_FAMILY, "SEG2")

    cls111 = TaxonomyNode("CLS111", "Class 1-1-1", "Cls 111 def", LEVEL_CLASS, "FAM11")
    cls121 = TaxonomyNode("CLS121", "Class 1-2-1", "Cls 121 def", LEVEL_CLASS, "FAM12")
    cls211 = TaxonomyNode("CLS211", "Class 2-1-1", "Cls 211 def", LEVEL_CLASS, "FAM21")
    cls221 = TaxonomyNode("CLS221", "Class 2-2-1", "Cls 221 def", LEVEL_CLASS, "FAM22")

    # Commodities sorted by class code (required by TaxonomyStore contract)
    com_list = [
        TaxonomyNode("COM1111", "Commodity 1", "Com 1 def", LEVEL_COMMODITY, "CLS111"),
        TaxonomyNode("COM1112", "Commodity 2", "Com 2 def", LEVEL_COMMODITY, "CLS111"),
        TaxonomyNode("COM1211", "Commodity 3", "Com 3 def", LEVEL_COMMODITY, "CLS121"),
        TaxonomyNode("COM1212", "Commodity 4", "Com 4 def", LEVEL_COMMODITY, "CLS121"),
        TaxonomyNode("COM2111", "Commodity 5", "Com 5 def", LEVEL_COMMODITY, "CLS211"),
        TaxonomyNode("COM2112", "Commodity 6", "Com 6 def", LEVEL_COMMODITY, "CLS211"),
        TaxonomyNode("COM2211", "Commodity 7", "Com 7 def", LEVEL_COMMODITY, "CLS221"),
        TaxonomyNode("COM2212", "Commodity 8", "Com 8 def", LEVEL_COMMODITY, "CLS221"),
    ]

    segments = [seg1, seg2]
    families = [fam11, fam12, fam21, fam22]
    classes = [cls111, cls121, cls211, cls221]
    commodities = com_list

    segment_embed_texts = [build_embed_text(n, n.label) for n in segments]
    family_embed_texts = [build_embed_text(n, f"Segment One > {n.label}") for n in families]
    class_embed_texts = [build_embed_text(n, f"Segment > Family > {n.label}") for n in classes]
    commodity_embed_texts = [build_embed_text(n, f"Seg > Fam > Cls > {n.label}") for n in commodities]

    segment_by_code = {n.code: n for n in segments}
    family_by_code = {n.code: n for n in families}
    class_by_code = {n.code: n for n in classes}
    commodity_by_code = {n.code: n for n in commodities}

    families_by_segment = {"SEG1": [fam11, fam12], "SEG2": [fam21, fam22]}
    classes_by_family = {
        "FAM11": [cls111], "FAM12": [cls121],
        "FAM21": [cls211], "FAM22": [cls221],
    }
    commodities_by_class = {
        "CLS111": [com_list[0], com_list[1]],
        "CLS121": [com_list[2], com_list[3]],
        "CLS211": [com_list[4], com_list[5]],
        "CLS221": [com_list[6], com_list[7]],
    }
    class_code_to_commodity_range = {
        "CLS111": (0, 2), "CLS121": (2, 4),
        "CLS211": (4, 6), "CLS221": (6, 8),
    }

    return TaxonomyStore(
        segments=segments,
        families=families,
        classes=classes,
        commodities=commodities,
        segment_by_code=segment_by_code,
        family_by_code=family_by_code,
        class_by_code=class_by_code,
        commodity_by_code=commodity_by_code,
        families_by_segment=families_by_segment,
        classes_by_family=classes_by_family,
        commodities_by_class=commodities_by_class,
        segment_embed_texts=segment_embed_texts,
        family_embed_texts=family_embed_texts,
        class_embed_texts=class_embed_texts,
        commodity_embed_texts=commodity_embed_texts,
        class_code_to_commodity_range=class_code_to_commodity_range,
    )


@pytest.fixture
def default_config() -> AmbiguityConfig:
    return AmbiguityConfig()
