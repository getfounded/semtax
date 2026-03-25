"""
Ambiguity and composite detection for classification results.

Four independent signals are evaluated per input:

1. margin_too_small   — top-1 and top-2 class cosine scores are within the
                        margin threshold (classifier not sure which class wins).

2. multi_segment_spread — the top-K flat class matches span more than one
                          Segment (input is either ambiguous or cross-category).

3. low_confidence     — the best class cosine score is below
                        class_confidence_threshold (poor semantic match overall).

4. composite_heuristic — the input text appears to describe multiple distinct
                          items (contains conjunctions or has 3+ simple word
                          tokens).  Intentionally simple — no NLP dependency.
                          Can be upgraded to a POS tagger in V2 without
                          changing the public flag name.

All signals are evaluated against class-level scores (not commodity-level)
because the class search is the primary anchor in the hybrid architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ._result import (
    FLAG_COMPOSITE_HEURISTIC,
    FLAG_LOW_CONFIDENCE,
    FLAG_MARGIN_TOO_SMALL,
    FLAG_MULTI_SEGMENT_SPREAD,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AmbiguityConfig:
    """
    Configurable thresholds for ambiguity detection and confidence gating.

    The two threshold fields serve distinct purposes:

    - ``class_confidence_threshold``: controls when to emit FLAG_LOW_CONFIDENCE.
      Below this value the class match is considered unreliable; the result is
      still returned but flagged.

    - ``commodity_confidence_threshold``: controls whether commodity is
      *populated* in the result.  Even a confident class match may not warrant
      drilling to the commodity level if the input is too vague.  When the
      commodity cosine score is below this value, ``match_level`` stops at
      "class" and ``commodity.populated`` is False.
    """

    class_confidence_threshold: float = 0.60
    """Minimum class cosine score before FLAG_LOW_CONFIDENCE is raised."""

    commodity_confidence_threshold: float = 0.72
    """Minimum commodity cosine score before commodity is omitted from output."""

    margin_threshold: float = 0.05
    """
    If top-1 class score minus top-2 class score is below this value,
    FLAG_MARGIN_TOO_SMALL is raised.
    """

    top_k_spread: int = 5
    """
    Number of top flat-class matches whose Segment membership is checked
    for multi-segment spread detection.
    """

    min_noun_count: int = 3
    """
    Minimum number of simple word tokens (all alpha) to trigger the
    composite heuristic when no conjunction is present.
    """


# ---------------------------------------------------------------------------
# Composite heuristic helpers
# ---------------------------------------------------------------------------

_CONJUNCTION_RE = re.compile(r"(?:\band\b|\bplus\b|(?<!\w)&(?!\w))", re.IGNORECASE)
_SIMPLE_WORD_RE = re.compile(r"^[a-zA-Z]+$")


def _is_composite(text: str, min_noun_count: int) -> bool:
    """
    Return True if the text appears to describe multiple distinct items.

    Two independent signals — either alone is sufficient:
      1. Contains a conjunction token (" and ", " & ", " plus ")
      2. Contains >= min_noun_count simple word tokens (alpha-only)
         as a rough proxy for multi-noun descriptions.
    """
    if _CONJUNCTION_RE.search(text):
        return True
    word_count = sum(1 for tok in text.split() if _SIMPLE_WORD_RE.match(tok))
    return word_count >= min_noun_count


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_flags(
    description: str,
    class_top1_score: float,
    class_top2_score: float,
    top_k_class_segment_codes: list[str],
    config: AmbiguityConfig,
) -> list[str]:
    """
    Evaluate all four ambiguity signals and return a list of active flag strings.

    Args:
        description:               The input text.
        class_top1_score:          Cosine score of the best flat-class match.
        class_top2_score:          Cosine score of the second-best flat-class match.
        top_k_class_segment_codes: Segment codes for the top-K flat class matches
                                   (used for multi-segment spread check).
        config:                    AmbiguityConfig instance.

    Returns:
        List of flag strings (subset of FLAG_* constants from _result.py).
        Empty list = clean, confident classification.
    """
    flags: list[str] = []

    # 1. Margin check
    if (class_top1_score - class_top2_score) < config.margin_threshold:
        flags.append(FLAG_MARGIN_TOO_SMALL)

    # 2. Multi-segment spread
    if len(set(top_k_class_segment_codes)) > 1:
        flags.append(FLAG_MULTI_SEGMENT_SPREAD)

    # 3. Absolute low confidence
    if class_top1_score < config.class_confidence_threshold:
        flags.append(FLAG_LOW_CONFIDENCE)

    # 4. Composite heuristic
    if _is_composite(description, config.min_noun_count):
        flags.append(FLAG_COMPOSITE_HEURISTIC)

    return flags
