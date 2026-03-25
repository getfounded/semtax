"""
Taxonomy data model, UNSPSC CSV loader, and embedding text construction.

The UNSPSC CSV is flat-denormalized: every row contains all four levels
(Segment, Family, Class, Commodity) repeated for each commodity.  This
module parses the 157k-row file into a deduplicated, hierarchical
TaxonomyStore that all other modules operate against.

Key design decisions:
- load_unspsc() is wrapped with lru_cache — process-wide singleton so
  multiple SemTax instances share the same parsed store.
- Commodities are sorted by class code so that commodity_matrix[start:end]
  gives a clean sub-matrix slice for each class (no per-class allocation).
- build_embed_text() guards against empty definition/synonym/acronym fields
  to avoid emitting "Also known as: ." when those UNSPSC columns are blank
  (as they are in the current v260801 release).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

LEVEL_SEGMENT = "segment"
LEVEL_FAMILY = "family"
LEVEL_CLASS = "class"
LEVEL_COMMODITY = "commodity"


@dataclass(frozen=True)
class TaxonomyNode:
    """Immutable node representing one entity at any level of the hierarchy."""

    code: str
    """Taxonomy code string, e.g. '43000000' (Segment), '43190000' (Family)."""

    label: str
    """Human-readable name."""

    definition: str
    """Full definition / description text.  May be empty string."""

    level: str
    """One of LEVEL_SEGMENT | LEVEL_FAMILY | LEVEL_CLASS | LEVEL_COMMODITY."""

    parent_code: Optional[str]
    """Parent node code.  None for Segment nodes."""

    synonym: str = ""
    """Alternate names.  Currently empty in UNSPSC v260801 — handled gracefully."""

    acronym: str = ""
    """Acronyms.  Currently empty in UNSPSC v260801 — handled gracefully."""


def build_embed_text(node: TaxonomyNode, path_prefix: str) -> str:
    """
    Construct the embedding string for a node.

    Rules:
    - path_prefix already contains the full "Segment > Family > Class > Label"
      path, resolved during CSV loading when all ancestor info is on the row.
    - If definition is non-empty, append " — {definition}".
    - For commodity nodes, conditionally append synonym and acronym clauses
      only when those fields are non-empty (guard against emitting
      "Also known as: ." when columns are blank).

    Examples:
        Segment:   "Live Plant and Animal Material — Includes live animals..."
        Family:    "Live Plant and Animal Material > Live animals — ..."
        Class:     "Live Plant and Animal Material > Live animals > Livestock — ..."
        Commodity: "... > Livestock > Cats — Domestic felines. Also known as: cat."
    """
    base = path_prefix
    if node.definition.strip():
        base = f"{path_prefix} — {node.definition.strip()}"
    if node.level == LEVEL_COMMODITY:
        if node.synonym.strip():
            base += f". Also known as: {node.synonym.strip()}"
        if node.acronym.strip():
            base += f". Acronyms: {node.acronym.strip()}"
    return base


# ---------------------------------------------------------------------------
# TaxonomyStore
# ---------------------------------------------------------------------------


@dataclass
class TaxonomyStore:
    """
    Deduplicated, indexed taxonomy hierarchy.

    Do not instantiate directly — use load_unspsc() or load_custom_taxonomy().

    Node lists (segments, families, classes, commodities) are the authoritative
    ordered sequences.  All embedding matrices use the same index alignment.

    Commodities are sorted by class code so that commodity_matrix[start:end]
    gives the sub-matrix for a specific class without allocation.
    """

    # Ordered node lists (index-aligned with embed text lists)
    segments: list[TaxonomyNode]
    families: list[TaxonomyNode]
    classes: list[TaxonomyNode]
    commodities: list[TaxonomyNode]

    # O(1) lookup by code
    segment_by_code: dict[str, TaxonomyNode]
    family_by_code: dict[str, TaxonomyNode]
    class_by_code: dict[str, TaxonomyNode]
    commodity_by_code: dict[str, TaxonomyNode]

    # Parent → children maps for top-down traversal
    families_by_segment: dict[str, list[TaxonomyNode]]   # seg_code  → [Family]
    classes_by_family: dict[str, list[TaxonomyNode]]     # fam_code  → [Class]
    commodities_by_class: dict[str, list[TaxonomyNode]]  # cls_code  → [Commodity]

    # Pre-built embed text lists (parallel to node lists, same index alignment)
    segment_embed_texts: list[str]
    family_embed_texts: list[str]
    class_embed_texts: list[str]
    commodity_embed_texts: list[str]

    # Range-based slice access into the commodity list/matrix by class code.
    # commodities[start:end] gives all commodities for a given class.
    # Requires commodities to be sorted by class code (guaranteed by loader).
    class_code_to_commodity_range: dict[str, tuple[int, int]]


# ---------------------------------------------------------------------------
# Loader — UNSPSC
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"
_UNSPSC_CSV = _DATA_DIR / "unspsc-english-v260801.1.csv"


@lru_cache(maxsize=1)
def load_unspsc() -> TaxonomyStore:
    """
    Parse the bundled UNSPSC CSV and return a TaxonomyStore.

    Wrapped with lru_cache(maxsize=1) — called once per process regardless of
    how many SemTax instances are created.

    Implementation notes:
    - encoding='utf-8-sig' strips the BOM that prefixes the 'Version' column
      header in the UNSPSC v260801 file (appears as '\\ufeffVersion').
    - Single-pass: each row carries full denormalized ancestor data, so path
      prefixes for embed texts are built inline without a second pass.
    - First-seen wins for duplicate codes (a code always has the same label
      and definition, so collision is not an issue — but guard just in case).
    - Commodities are sorted by class code before building the final list so
      that class_code_to_commodity_range slices are contiguous.
    """
    if not _UNSPSC_CSV.exists():
        raise FileNotFoundError(
            f"UNSPSC taxonomy data not found at {_UNSPSC_CSV}.\n"
            "Copy taxonomy_data_sets/unspsc-english-v260801.1.csv into "
            "src/semtax/data/ before using the 'unspsc' preset."
        )

    # Intermediate storage — keyed by code, first-seen wins
    seg_map: dict[str, dict] = {}
    fam_map: dict[str, dict] = {}
    cls_map: dict[str, dict] = {}
    com_map: dict[str, dict] = {}

    # Ancestor labels needed to build path prefixes (populated inline)
    _seg_label: dict[str, str] = {}
    _fam_label: dict[str, str] = {}
    _cls_label: dict[str, str] = {}

    with open(_UNSPSC_CSV, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seg_code = row["Segment"].strip()
            fam_code = row["Family"].strip()
            cls_code = row["Class"].strip()
            com_code = row["Commodity"].strip()

            # Skip rows where any level code is absent (malformed rows)
            if not seg_code or not fam_code or not cls_code or not com_code:
                continue

            if seg_code not in seg_map:
                lbl = row["Segment Title"].strip()
                seg_map[seg_code] = {
                    "label": lbl,
                    "definition": row["Segment Definition"].strip(),
                }
                _seg_label[seg_code] = lbl

            if fam_code not in fam_map:
                lbl = row["Family Title"].strip()
                fam_map[fam_code] = {
                    "label": lbl,
                    "definition": row["Family Definition"].strip(),
                    "parent_code": seg_code,
                }
                _fam_label[fam_code] = lbl

            if cls_code not in cls_map:
                lbl = row["Class Title"].strip()
                cls_map[cls_code] = {
                    "label": lbl,
                    "definition": row["Class Definition"].strip(),
                    "parent_code": fam_code,
                }
                _cls_label[cls_code] = lbl

            if com_code not in com_map:
                com_map[com_code] = {
                    "label": row["Commodity Title"].strip(),
                    "definition": row["Commodity Definition"].strip(),
                    "parent_code": cls_code,
                    "synonym": row.get("Synonym", "").strip(),
                    "acronym": row.get("Acronym", "").strip(),
                    # Store parent labels inline for path prefix construction
                    "_seg_label": _seg_label[seg_code],
                    "_fam_label": _fam_label.get(fam_code, ""),
                    "_cls_label": _cls_label.get(cls_code, ""),
                }

    # Build TaxonomyNode lists and embed texts
    segments, segment_embed_texts = _build_segment_nodes(seg_map)
    families, family_embed_texts = _build_family_nodes(fam_map, _seg_label)
    classes, class_embed_texts = _build_class_nodes(cls_map, _seg_label, _fam_label, fam_map)

    # Sort commodities by class code for contiguous range slicing
    sorted_com_items = sorted(com_map.items(), key=lambda kv: kv[1]["parent_code"])
    commodities, commodity_embed_texts = _build_commodity_nodes(sorted_com_items)

    # Build lookup maps
    segment_by_code = {n.code: n for n in segments}
    family_by_code = {n.code: n for n in families}
    class_by_code = {n.code: n for n in classes}
    commodity_by_code = {n.code: n for n in commodities}

    # Parent → children maps
    families_by_segment: dict[str, list[TaxonomyNode]] = {}
    for fam in families:
        families_by_segment.setdefault(fam.parent_code, []).append(fam)

    classes_by_family: dict[str, list[TaxonomyNode]] = {}
    for cls in classes:
        classes_by_family.setdefault(cls.parent_code, []).append(cls)

    commodities_by_class: dict[str, list[TaxonomyNode]] = {}
    for com in commodities:
        commodities_by_class.setdefault(com.parent_code, []).append(com)

    # Build class → commodity range dict
    class_code_to_commodity_range = _build_commodity_ranges(commodities)

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


# ---------------------------------------------------------------------------
# Node / embed text builders (private helpers)
# ---------------------------------------------------------------------------


def _build_segment_nodes(
    seg_map: dict[str, dict],
) -> tuple[list[TaxonomyNode], list[str]]:
    nodes: list[TaxonomyNode] = []
    texts: list[str] = []
    for code, d in sorted(seg_map.items()):
        node = TaxonomyNode(
            code=code,
            label=d["label"],
            definition=d["definition"],
            level=LEVEL_SEGMENT,
            parent_code=None,
        )
        nodes.append(node)
        texts.append(build_embed_text(node, d["label"]))
    return nodes, texts


def _build_family_nodes(
    fam_map: dict[str, dict],
    seg_label: dict[str, str],
) -> tuple[list[TaxonomyNode], list[str]]:
    nodes: list[TaxonomyNode] = []
    texts: list[str] = []
    for code, d in sorted(fam_map.items()):
        node = TaxonomyNode(
            code=code,
            label=d["label"],
            definition=d["definition"],
            level=LEVEL_FAMILY,
            parent_code=d["parent_code"],
        )
        nodes.append(node)
        seg_lbl = seg_label.get(d["parent_code"], "")
        path_prefix = f"{seg_lbl} > {d['label']}" if seg_lbl else d["label"]
        texts.append(build_embed_text(node, path_prefix))
    return nodes, texts


def _build_class_nodes(
    cls_map: dict[str, dict],
    seg_label: dict[str, str],
    fam_label: dict[str, str],
    fam_map: dict[str, dict],
) -> tuple[list[TaxonomyNode], list[str]]:
    nodes: list[TaxonomyNode] = []
    texts: list[str] = []
    for code, d in sorted(cls_map.items()):
        node = TaxonomyNode(
            code=code,
            label=d["label"],
            definition=d["definition"],
            level=LEVEL_CLASS,
            parent_code=d["parent_code"],
        )
        nodes.append(node)
        fam_code = d["parent_code"]
        fam_lbl = fam_label.get(fam_code, "")
        # Resolve segment label via fam_map parent_code (clean, no heuristics)
        seg_code = fam_map.get(fam_code, {}).get("parent_code", "")
        seg_lbl = seg_label.get(seg_code, "")
        if seg_lbl and fam_lbl:
            path_prefix = f"{seg_lbl} > {fam_lbl} > {d['label']}"
        elif fam_lbl:
            path_prefix = f"{fam_lbl} > {d['label']}"
        else:
            path_prefix = d["label"]
        texts.append(build_embed_text(node, path_prefix))
    return nodes, texts


def _build_commodity_nodes(
    sorted_com_items: list[tuple[str, dict]],
) -> tuple[list[TaxonomyNode], list[str]]:
    nodes: list[TaxonomyNode] = []
    texts: list[str] = []
    for code, d in sorted_com_items:
        node = TaxonomyNode(
            code=code,
            label=d["label"],
            definition=d["definition"],
            level=LEVEL_COMMODITY,
            parent_code=d["parent_code"],
            synonym=d["synonym"],
            acronym=d["acronym"],
        )
        nodes.append(node)
        seg_lbl = d.get("_seg_label", "")
        fam_lbl = d.get("_fam_label", "")
        cls_lbl = d.get("_cls_label", "")
        if seg_lbl and fam_lbl and cls_lbl:
            path_prefix = f"{seg_lbl} > {fam_lbl} > {cls_lbl} > {d['label']}"
        elif cls_lbl:
            path_prefix = f"{cls_lbl} > {d['label']}"
        else:
            path_prefix = d["label"]
        texts.append(build_embed_text(node, path_prefix))
    return nodes, texts


def _build_commodity_ranges(
    commodities: list[TaxonomyNode],
) -> dict[str, tuple[int, int]]:
    """
    Build a dict mapping class_code → (start, end) index range into the
    commodities list.  Assumes commodities are already sorted by parent_code
    (class code).
    """
    ranges: dict[str, tuple[int, int]] = {}
    if not commodities:
        return ranges
    current_cls = commodities[0].parent_code
    start = 0
    for i, com in enumerate(commodities):
        if com.parent_code != current_cls:
            ranges[current_cls] = (start, i)
            current_cls = com.parent_code
            start = i
    # Close last group
    ranges[current_cls] = (start, len(commodities))
    return ranges
