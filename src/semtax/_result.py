"""
Output dataclasses and flag string constants for semtax classification results.

Flag constants are defined here (not in _ambiguity.py) so they can be imported
by both _ambiguity.py and _classifier.py without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Ambiguity flag constants
# ---------------------------------------------------------------------------

FLAG_MARGIN_TOO_SMALL = "margin_too_small"
"""Top-1 and top-2 class cosine scores are within the margin threshold."""

FLAG_MULTI_SEGMENT_SPREAD = "multi_segment_spread"
"""Top-K flat class matches span more than one Segment."""

FLAG_LOW_CONFIDENCE = "low_confidence"
"""Best class cosine score is below the class_confidence_threshold."""

FLAG_COMPOSITE_HEURISTIC = "composite_heuristic"
"""Input description appears to describe multiple distinct items."""

ALL_FLAGS = frozenset(
    {FLAG_MARGIN_TOO_SMALL, FLAG_MULTI_SEGMENT_SPREAD, FLAG_LOW_CONFIDENCE, FLAG_COMPOSITE_HEURISTIC}
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LevelResult:
    """Classification result at a single taxonomy level."""

    code: str
    """Taxonomy code, e.g. '43000000'."""

    label: str
    """Human-readable label, e.g. 'Information Technology Broadcasting and Telecommunications'."""

    confidence: float
    """Raw cosine similarity score (0.0–1.0)."""

    populated: bool = True
    """
    False when this level was not reached.
    Always True for segment/family/class.
    False for commodity when match_level is 'class' or above.
    """

    def __repr__(self) -> str:
        populated_str = "" if self.populated else ", populated=False"
        return f"LevelResult(code={self.code!r}, label={self.label!r}, confidence={self.confidence:.4f}{populated_str})"


@dataclass
class ClassificationResult:
    """
    Full hierarchical classification result for a single input description.

    The ``match_level`` field indicates the deepest level populated with
    confidence above threshold.  ``commodity.populated`` mirrors this:
    it is False whenever ``match_level != 'commodity'``.
    """

    description: str
    """Original input text."""

    segment: LevelResult
    family: LevelResult
    class_: LevelResult
    commodity: LevelResult

    match_level: str
    """Deepest level with confident classification: 'segment' | 'family' | 'class' | 'commodity'."""

    flags: list[str] = field(default_factory=list)
    """Active ambiguity flags. Empty list means a clean, confident classification."""

    # Internal fields — present for debugging/inspection; excluded from to_dict()
    _top_down_path: Optional[list[LevelResult]] = field(default=None, repr=False, compare=False)
    _flat_class_match: Optional[LevelResult] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        """
        Serialise to a nested dict.  Suitable for JSON serialisation or
        constructing a pandas DataFrame row.
        """
        return {
            "description": self.description,
            "segment": {
                "code": self.segment.code,
                "label": self.segment.label,
                "confidence": self.segment.confidence,
            },
            "family": {
                "code": self.family.code,
                "label": self.family.label,
                "confidence": self.family.confidence,
            },
            "class": {
                "code": self.class_.code,
                "label": self.class_.label,
                "confidence": self.class_.confidence,
            },
            "commodity": {
                "code": self.commodity.code,
                "label": self.commodity.label,
                "confidence": self.commodity.confidence,
                "populated": self.commodity.populated,
            },
            "match_level": self.match_level,
            "flags": list(self.flags),
        }

    def to_flat_dict(self) -> dict:
        """
        Serialise to a single-level dict with prefixed keys.
        Useful for building a CSV row or a flat pandas DataFrame.

        Keys: description, segment_code, segment_label, segment_confidence,
              family_code, family_label, family_confidence,
              class_code, class_label, class_confidence,
              commodity_code, commodity_label, commodity_confidence, commodity_populated,
              match_level, flags
        """
        return {
            "description": self.description,
            "segment_code": self.segment.code,
            "segment_label": self.segment.label,
            "segment_confidence": self.segment.confidence,
            "family_code": self.family.code,
            "family_label": self.family.label,
            "family_confidence": self.family.confidence,
            "class_code": self.class_.code,
            "class_label": self.class_.label,
            "class_confidence": self.class_.confidence,
            "commodity_code": self.commodity.code,
            "commodity_label": self.commodity.label,
            "commodity_confidence": self.commodity.confidence,
            "commodity_populated": self.commodity.populated,
            "match_level": self.match_level,
            "flags": "|".join(self.flags) if self.flags else "",
        }
