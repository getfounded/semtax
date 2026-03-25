"""
Tests for _embeddings.py: cosine similarity, cache round-trip, invalidation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from semtax._embeddings import (
    EmbeddingEngine,
    TaxonomyEmbeddingCache,
    _model_hash,
    cosine_similarity_matrix,
)
from conftest import fake_embed


# ---------------------------------------------------------------------------
# cosine_similarity_matrix
# ---------------------------------------------------------------------------


def test_cosine_self_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    matrix = v.reshape(1, -1)
    score = cosine_similarity_matrix(v, matrix)
    assert abs(float(score[0]) - 1.0) < 1e-5


def test_cosine_orthogonal_is_zero():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    keys = v2.reshape(1, -1)
    score = cosine_similarity_matrix(v1, keys)
    assert abs(float(score[0])) < 1e-5


def test_cosine_opposite_is_minus_one():
    v = np.array([1.0, 0.0], dtype=np.float32)
    keys = np.array([[-1.0, 0.0]], dtype=np.float32)
    score = cosine_similarity_matrix(v, keys)
    assert abs(float(score[0]) - (-1.0)) < 1e-5


def test_cosine_batch_shape():
    queries = np.random.rand(5, 16).astype(np.float32)
    keys = np.random.rand(10, 16).astype(np.float32)
    result = cosine_similarity_matrix(queries, keys)
    assert result.shape == (5, 10)


def test_cosine_single_query_shape():
    query = np.random.rand(16).astype(np.float32)
    keys = np.random.rand(10, 16).astype(np.float32)
    result = cosine_similarity_matrix(query, keys)
    assert result.shape == (10,)


def test_cosine_scores_bounded():
    """Cosine similarity should be in [-1, 1]."""
    queries = np.random.rand(20, 32).astype(np.float32)
    keys = np.random.rand(50, 32).astype(np.float32)
    result = cosine_similarity_matrix(queries, keys)
    assert float(result.min()) >= -1.0 - 1e-5
    assert float(result.max()) <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# _model_hash
# ---------------------------------------------------------------------------


def test_model_hash_is_8_chars():
    h = _model_hash("sentence-transformers/all-MiniLM-L6-v2")
    assert len(h) == 8


def test_model_hash_stable():
    """Same model_id always produces the same hash."""
    h1 = _model_hash("my-model")
    h2 = _model_hash("my-model")
    assert h1 == h2


def test_model_hash_different_models_differ():
    h1 = _model_hash("model-a")
    h2 = _model_hash("model-b")
    assert h1 != h2


# ---------------------------------------------------------------------------
# EmbeddingEngine with fake_embed callable
# ---------------------------------------------------------------------------


def test_engine_callable_model_returns_array():
    engine = EmbeddingEngine(model_spec=fake_embed)
    texts = ["hello world", "toner cartridge"]
    result = engine.embed(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 16)
    assert result.dtype == np.float32


def test_engine_empty_input_returns_empty():
    engine = EmbeddingEngine(model_spec=fake_embed)
    result = engine.embed([])
    assert result.shape[0] == 0


def test_engine_model_id_for_callable():
    engine = EmbeddingEngine(model_spec=fake_embed)
    assert "fake_embed" in engine.model_id


# ---------------------------------------------------------------------------
# TaxonomyEmbeddingCache — round-trip and invalidation
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_engine(tmp_path, monkeypatch):
    """EmbeddingEngine + patched CACHE_DIR pointing to tmp_path."""
    import semtax._embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "CACHE_DIR", tmp_path)
    # Also patch TaxonomyEmbeddingCache to use tmp_path
    orig_init = TaxonomyEmbeddingCache.__init__

    def patched_init(self, taxonomy_name, engine):
        orig_init(self, taxonomy_name, engine)
        self.__class__  # ensure the patch propagates
        # Override _cache_paths to use tmp_path
        import semtax._embeddings as m
        m.CACHE_DIR = tmp_path
        tmp_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(TaxonomyEmbeddingCache, "__init__", patched_init)
    engine = EmbeddingEngine(model_spec=fake_embed)
    return engine, tmp_path


def test_cache_round_trip(tmp_path):
    """Compute embeddings, save, reload, assert arrays are equal."""
    import semtax._embeddings as emb_mod

    engine = EmbeddingEngine(model_spec=fake_embed)
    # Patch CACHE_DIR on the class level for this test
    original_cache_dir = emb_mod.CACHE_DIR
    emb_mod.CACHE_DIR = tmp_path
    try:
        cache = TaxonomyEmbeddingCache("test_taxonomy", engine)
        texts = ["item one", "item two", "item three"]

        first = cache.load_or_compute("segment", texts, show_progress=False)
        second = cache.load_or_compute("segment", texts, show_progress=False)

        np.testing.assert_array_equal(first, second)
        assert first.shape == (3, 16)
    finally:
        emb_mod.CACHE_DIR = original_cache_dir


def test_cache_invalidation_on_count_change(tmp_path):
    """Adding a text should miss the cache and recompute."""
    import semtax._embeddings as emb_mod

    engine = EmbeddingEngine(model_spec=fake_embed)
    original_cache_dir = emb_mod.CACHE_DIR
    emb_mod.CACHE_DIR = tmp_path
    try:
        cache = TaxonomyEmbeddingCache("test_taxonomy", engine)
        texts_3 = ["a", "b", "c"]
        texts_4 = ["a", "b", "c", "d"]

        result_3 = cache.load_or_compute("family", texts_3, show_progress=False)
        result_4 = cache.load_or_compute("family", texts_4, show_progress=False)

        assert result_3.shape == (3, 16)
        assert result_4.shape == (4, 16)
    finally:
        emb_mod.CACHE_DIR = original_cache_dir


def test_cache_writes_meta_json(tmp_path):
    import semtax._embeddings as emb_mod

    engine = EmbeddingEngine(model_spec=fake_embed)
    original_cache_dir = emb_mod.CACHE_DIR
    emb_mod.CACHE_DIR = tmp_path
    try:
        cache = TaxonomyEmbeddingCache("test_taxonomy", engine)
        cache.load_or_compute("class", ["x", "y"], show_progress=False)

        meta_files = list(tmp_path.glob("*.meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["node_count"] == 2
        assert meta["taxonomy_name"] == "test_taxonomy"
        assert meta["level"] == "class"
    finally:
        emb_mod.CACHE_DIR = original_cache_dir
