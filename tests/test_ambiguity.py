"""
Tests for _ambiguity.py — each of the 4 flag signals independently,
threshold boundaries, and composite heuristic details.
"""

from __future__ import annotations

import pytest

from semtax._ambiguity import AmbiguityConfig, _is_composite, detect_flags
from semtax._result import (
    FLAG_COMPOSITE_HEURISTIC,
    FLAG_LOW_CONFIDENCE,
    FLAG_MARGIN_TOO_SMALL,
    FLAG_MULTI_SEGMENT_SPREAD,
)


@pytest.fixture
def config():
    return AmbiguityConfig(
        class_confidence_threshold=0.60,
        commodity_confidence_threshold=0.72,
        margin_threshold=0.05,
        top_k_spread=5,
        min_noun_count=3,
    )


# ---------------------------------------------------------------------------
# FLAG_MARGIN_TOO_SMALL
# ---------------------------------------------------------------------------


def test_margin_flag_raised_when_close(config):
    flags = detect_flags("toner", 0.82, 0.79, ["SEG1", "SEG1", "SEG1", "SEG1", "SEG1"], config)
    assert FLAG_MARGIN_TOO_SMALL in flags


def test_margin_flag_not_raised_when_clear(config):
    flags = detect_flags("toner", 0.82, 0.70, ["SEG1"] * 5, config)
    assert FLAG_MARGIN_TOO_SMALL not in flags


def test_margin_flag_at_exact_threshold_not_raised(config):
    # margin == threshold → NOT flagged (strict less-than)
    flags = detect_flags("toner", 0.80, 0.75, ["SEG1"] * 5, config)
    assert FLAG_MARGIN_TOO_SMALL not in flags  # 0.05 is not < 0.05


def test_margin_flag_just_below_threshold(config):
    flags = detect_flags("toner", 0.80, 0.7501, ["SEG1"] * 5, config)
    assert FLAG_MARGIN_TOO_SMALL in flags  # 0.0499 < 0.05


# ---------------------------------------------------------------------------
# FLAG_MULTI_SEGMENT_SPREAD
# ---------------------------------------------------------------------------


def test_spread_flag_raised_on_multiple_segments(config):
    flags = detect_flags("query", 0.75, 0.65, ["SEG1", "SEG2", "SEG1", "SEG2", "SEG3"], config)
    assert FLAG_MULTI_SEGMENT_SPREAD in flags


def test_spread_flag_not_raised_on_single_segment(config):
    flags = detect_flags("query", 0.75, 0.65, ["SEG1", "SEG1", "SEG1", "SEG1", "SEG1"], config)
    assert FLAG_MULTI_SEGMENT_SPREAD not in flags


def test_spread_flag_two_segments_is_enough(config):
    flags = detect_flags("query", 0.75, 0.65, ["SEG1", "SEG2", "SEG1", "SEG1", "SEG1"], config)
    assert FLAG_MULTI_SEGMENT_SPREAD in flags


# ---------------------------------------------------------------------------
# FLAG_LOW_CONFIDENCE
# ---------------------------------------------------------------------------


def test_low_confidence_raised_below_threshold(config):
    flags = detect_flags("query", 0.45, 0.40, ["SEG1"] * 5, config)
    assert FLAG_LOW_CONFIDENCE in flags


def test_low_confidence_not_raised_above_threshold(config):
    flags = detect_flags("query", 0.65, 0.55, ["SEG1"] * 5, config)
    assert FLAG_LOW_CONFIDENCE not in flags


def test_low_confidence_at_exact_threshold_not_raised(config):
    # score == threshold → NOT flagged (strict less-than)
    flags = detect_flags("query", 0.60, 0.50, ["SEG1"] * 5, config)
    assert FLAG_LOW_CONFIDENCE not in flags


# ---------------------------------------------------------------------------
# FLAG_COMPOSITE_HEURISTIC
# ---------------------------------------------------------------------------


def test_composite_flag_on_and_conjunction(config):
    flags = detect_flags("keyboards and mice", 0.75, 0.65, ["SEG1"] * 5, config)
    assert FLAG_COMPOSITE_HEURISTIC in flags


def test_composite_flag_on_ampersand(config):
    flags = detect_flags("hardware & software", 0.75, 0.65, ["SEG1"] * 5, config)
    assert FLAG_COMPOSITE_HEURISTIC in flags


def test_composite_flag_on_plus(config):
    flags = detect_flags("installation plus configuration", 0.75, 0.65, ["SEG1"] * 5, config)
    assert FLAG_COMPOSITE_HEURISTIC in flags


def test_composite_flag_on_multi_noun(config):
    # "server rack installation service" → 4 words, all alpha → composite
    flags = detect_flags("server rack installation service", 0.75, 0.65, ["SEG1"] * 5, config)
    assert FLAG_COMPOSITE_HEURISTIC in flags


def test_no_composite_flag_on_simple_description(config):
    # "toner" → 1 word → not composite
    flags = detect_flags("toner", 0.85, 0.70, ["SEG1"] * 5, config)
    assert FLAG_COMPOSITE_HEURISTIC not in flags


def test_composite_threshold_exactly_at_boundary(config):
    # min_noun_count=3; "office chair desk" → 3 words → flagged
    flags = detect_flags("office chair desk", 0.80, 0.70, ["SEG1"] * 5, config)
    assert FLAG_COMPOSITE_HEURISTIC in flags


def test_composite_threshold_below_boundary(config):
    # "office chair" → 2 words → NOT flagged
    flags = detect_flags("office chair", 0.80, 0.70, ["SEG1"] * 5, config)
    assert FLAG_COMPOSITE_HEURISTIC not in flags


# ---------------------------------------------------------------------------
# _is_composite helper
# ---------------------------------------------------------------------------


def test_is_composite_conjunction():
    assert _is_composite("keyboards and mice", 3) is True


def test_is_composite_multi_noun():
    assert _is_composite("server rack cable", 3) is True


def test_is_composite_false_single_word():
    assert _is_composite("toner", 3) is False


def test_is_composite_false_two_words():
    assert _is_composite("toner cartridge", 3) is False


# ---------------------------------------------------------------------------
# No flags on clean input
# ---------------------------------------------------------------------------


def test_no_flags_clean_confident_input(config):
    flags = detect_flags(
        "toner",
        class_top1_score=0.88,
        class_top2_score=0.70,
        top_k_class_segment_codes=["SEG1"] * 5,
        config=config,
    )
    assert flags == []


# ---------------------------------------------------------------------------
# AmbiguityConfig defaults
# ---------------------------------------------------------------------------


def test_default_config_values():
    cfg = AmbiguityConfig()
    assert cfg.class_confidence_threshold == 0.60
    assert cfg.commodity_confidence_threshold == 0.72
    assert cfg.margin_threshold == 0.05
    assert cfg.top_k_spread == 5
    assert cfg.min_noun_count == 3
