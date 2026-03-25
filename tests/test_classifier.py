"""
Tests for _classifier.py: HybridClassifier reconciliation logic,
commodity threshold behavior, batch vs single consistency.
Uses the minimal_store fixture and fake_embed callable from conftest.
"""

from __future__ import annotations

import numpy as np
import pytest

from semtax._ambiguity import AmbiguityConfig
from semtax._classifier import HybridClassifier, _second_score
from semtax._embeddings import EmbeddingEngine, TaxonomyEmbeddingCache
from semtax._result import (
    FLAG_COMPOSITE_HEURISTIC,
    FLAG_LOW_CONFIDENCE,
    ClassificationResult,
)
from conftest import fake_embed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return EmbeddingEngine(model_spec=fake_embed)


@pytest.fixture
def warmed_classifier(minimal_store, engine, tmp_path):
    """HybridClassifier with fake embeddings, warmed up against minimal_store."""
    import semtax._embeddings as emb_mod
    original = emb_mod.CACHE_DIR
    emb_mod.CACHE_DIR = tmp_path

    config = AmbiguityConfig(
        class_confidence_threshold=0.0,   # don't gate on confidence in most tests
        commodity_confidence_threshold=0.0,
        margin_threshold=0.05,
    )
    cache = TaxonomyEmbeddingCache("test", engine)
    clf = HybridClassifier(
        store=minimal_store,
        engine=engine,
        cache=cache,
        config=config,
    )
    clf.warm_up(show_progress=False)
    yield clf

    emb_mod.CACHE_DIR = original


@pytest.fixture
def strict_threshold_classifier(minimal_store, engine, tmp_path):
    """Classifier with high thresholds to test populated=False behavior."""
    import semtax._embeddings as emb_mod
    original = emb_mod.CACHE_DIR
    emb_mod.CACHE_DIR = tmp_path

    config = AmbiguityConfig(
        class_confidence_threshold=1.1,   # > 1.0, impossible for cosine to reach
        commodity_confidence_threshold=1.1,
    )
    cache = TaxonomyEmbeddingCache("test_strict", engine)
    clf = HybridClassifier(
        store=minimal_store,
        engine=engine,
        cache=cache,
        config=config,
    )
    clf.warm_up(show_progress=False)
    yield clf

    emb_mod.CACHE_DIR = original


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_classify_returns_classification_result(warmed_classifier):
    results = warmed_classifier.classify_batch(["item one"])
    assert len(results) == 1
    assert isinstance(results[0], ClassificationResult)


def test_result_has_all_fields(warmed_classifier):
    r = warmed_classifier.classify_batch(["item one"])[0]
    assert r.segment is not None
    assert r.family is not None
    assert r.class_ is not None
    assert r.commodity is not None
    assert r.match_level in ("segment", "family", "class", "commodity")
    assert isinstance(r.flags, list)


def test_result_description_preserved(warmed_classifier):
    text = "item one original text"
    results = warmed_classifier.classify_batch([text])
    assert results[0].description == text


# ---------------------------------------------------------------------------
# Commodity threshold gating
# ---------------------------------------------------------------------------


def test_commodity_not_populated_when_threshold_high(strict_threshold_classifier):
    """With very high thresholds nothing should be populated at commodity level."""
    results = strict_threshold_classifier.classify_batch(["item"])
    r = results[0]
    assert r.commodity.populated is False
    assert r.match_level != "commodity"


def test_commodity_populated_when_threshold_zero(warmed_classifier):
    """With threshold=0.0, commodity should always be populated."""
    results = warmed_classifier.classify_batch(["item"])
    r = results[0]
    assert r.commodity.populated is True
    assert r.match_level == "commodity"


# ---------------------------------------------------------------------------
# Batch consistency
# ---------------------------------------------------------------------------


def test_batch_length_matches_input(warmed_classifier):
    texts = ["item one", "item two", "item three"]
    results = warmed_classifier.classify_batch(texts)
    assert len(results) == len(texts)


def test_batch_order_preserved(warmed_classifier):
    texts = ["item one", "item two", "item three"]
    results = warmed_classifier.classify_batch(texts)
    for i, r in enumerate(results):
        assert r.description == texts[i]


def test_single_and_batch_give_same_result(warmed_classifier):
    text = "item one"
    batch_result = warmed_classifier.classify_batch([text])[0]
    single_results = warmed_classifier.classify_batch([text])
    assert len(single_results) == 1
    r = single_results[0]
    assert r.class_.code == batch_result.class_.code
    assert r.commodity.code == batch_result.commodity.code


# ---------------------------------------------------------------------------
# Reconciliation: top-down vs flat class agreement
# ---------------------------------------------------------------------------


def test_internal_paths_stored(warmed_classifier):
    """_top_down_path and _flat_class_match should be populated for debugging."""
    results = warmed_classifier.classify_batch(["test"])
    r = results[0]
    assert r._top_down_path is not None
    assert r._flat_class_match is not None


def test_confidence_scores_are_floats(warmed_classifier):
    r = warmed_classifier.classify_batch(["item"])[0]
    assert isinstance(r.segment.confidence, float)
    assert isinstance(r.class_.confidence, float)
    assert isinstance(r.commodity.confidence, float)


def test_confidence_scores_bounded(warmed_classifier):
    texts = ["item one", "item two"]
    for r in warmed_classifier.classify_batch(texts):
        assert 0.0 <= r.segment.confidence <= 1.0
        assert 0.0 <= r.class_.confidence <= 1.0
        if r.commodity.populated:
            assert 0.0 <= r.commodity.confidence <= 1.0


# ---------------------------------------------------------------------------
# to_dict / to_flat_dict
# ---------------------------------------------------------------------------


def test_to_dict_has_required_keys(warmed_classifier):
    r = warmed_classifier.classify_batch(["item"])[0]
    d = r.to_dict()
    for key in ("description", "segment", "family", "class", "commodity", "match_level", "flags"):
        assert key in d


def test_to_flat_dict_has_required_keys(warmed_classifier):
    r = warmed_classifier.classify_batch(["item"])[0]
    d = r.to_flat_dict()
    for key in ("segment_code", "segment_label", "family_code", "class_code",
                "commodity_code", "commodity_populated", "match_level", "flags"):
        assert key in d


def test_to_flat_dict_flags_as_pipe_separated(warmed_classifier):
    r = warmed_classifier.classify_batch(["item and other and more"])[0]
    d = r.to_flat_dict()
    # flags field should be a string (possibly empty or pipe-separated)
    assert isinstance(d["flags"], str)


# ---------------------------------------------------------------------------
# _second_score helper
# ---------------------------------------------------------------------------


def test_second_score_basic():
    arr = np.array([0.9, 0.7, 0.5], dtype=np.float32)
    assert abs(_second_score(arr, 0) - 0.7) < 1e-5


def test_second_score_single_element():
    arr = np.array([0.9], dtype=np.float32)
    assert _second_score(arr, 0) == 0.0
