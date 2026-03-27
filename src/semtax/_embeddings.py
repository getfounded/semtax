"""
Embedding engine, taxonomy embedding cache, and cosine similarity utilities.

Architecture:
- EmbeddingEngine wraps a sentence-transformers model OR any callable with
  signature (list[str]) -> list[list[float]].  Model loading is lazy.
- TaxonomyEmbeddingCache handles disk persistence in ~/.semtax/cache/.
  Cache files are numpy .npy arrays (float32) with companion .meta.json.
  Cache key = taxonomy_name + level + sha256(model_id)[:8].
  Validity is checked by node count — if the count changes (taxonomy update),
  the cache is invalidated and recomputed.

Known V1 limitation:
  Cache does NOT invalidate when embed text CONTENT changes without a count
  change (e.g., future UNSPSC version that populates Synonym/Acronym columns).
  To force recomputation, delete ~/.semtax/cache/ manually.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import redirect_stderr
from pathlib import Path
from typing import Callable, Union

# Must be set before huggingface_hub is imported anywhere — suppresses the
# Windows symlinks warning that fires on every model load from cache.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""
Default embedding model.  all-MiniLM-L6-v2 produces 384-dim vectors,
runs on CPU without a GPU, and is downloaded automatically by
sentence-transformers on first use (~90MB).
"""

CACHE_DIR = Path.home() / ".semtax" / "cache"
EMBED_BATCH_SIZE = 512

ModelSpec = Union[str, Callable[[list[str]], list[list[float]]]]


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------


class EmbeddingEngine:
    """
    Wraps a sentence-transformers model (or any callable) and provides
    batched embedding with optional tqdm progress reporting.

    The underlying model is loaded lazily on the first embed() call to
    avoid startup cost when the taxonomy cache is already warm.

    Args:
        model_spec: sentence-transformers model name (str) or a callable
            with signature (list[str]) -> list[list[float]].
    """

    def __init__(self, model_spec: ModelSpec = DEFAULT_MODEL):
        self._model_spec = model_spec
        self._model = None  # loaded lazily
        self._model_id: str = _resolve_model_id(model_spec)

    @property
    def model_id(self) -> str:
        """Stable string identifier for this model (used as cache key)."""
        return self._model_id

    def embed(
        self,
        texts: list[str],
        show_progress: bool = False,
        desc: str = "Embedding",
    ) -> np.ndarray:
        """
        Embed a list of strings.

        Returns a float32 numpy array of shape (N, D) where N = len(texts)
        and D is the embedding dimension of the model.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_loaded()
        batches = _chunk(texts, EMBED_BATCH_SIZE)
        all_embeddings: list[np.ndarray] = []

        iterator = tqdm(batches, desc=desc, disable=not show_progress, leave=False)
        for batch in iterator:
            vecs = self._embed_batch(batch)
            all_embeddings.append(np.asarray(vecs, dtype=np.float32))

        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if callable(self._model_spec) and not isinstance(self._model_spec, str):
            # Callable IS the model — no loading needed
            self._model = self._model_spec
        else:
            import logging
            import warnings
            # Suppress noisy first-run output from huggingface_hub and
            # sentence-transformers (symlinks on Windows, unauthenticated HF
            # token, LOAD REPORT, unexpected weight keys). All harmless.
            _st_logger = logging.getLogger("sentence_transformers")
            _hf_logger = logging.getLogger("huggingface_hub")
            _prev_st = _st_logger.level
            _prev_hf = _hf_logger.level
            _st_logger.setLevel(logging.ERROR)
            _hf_logger.setLevel(logging.ERROR)
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            with warnings.catch_warnings(), redirect_stderr(io.StringIO()):
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._model = SentenceTransformer(self._model_spec)
            _st_logger.setLevel(_prev_st)
            _hf_logger.setLevel(_prev_hf)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if callable(self._model_spec) and not isinstance(self._model_spec, str):
            return self._model(texts)
        # SentenceTransformer.encode returns ndarray
        return self._model.encode(texts, show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# TaxonomyEmbeddingCache
# ---------------------------------------------------------------------------


class TaxonomyEmbeddingCache:
    """
    Manages loading and saving of pre-computed taxonomy embedding matrices.

    File naming: {taxonomy}_{level}_{model_hash8}.npy
    Companion:   {taxonomy}_{level}_{model_hash8}.meta.json

    One instance per (taxonomy_name, EmbeddingEngine) pair.
    """

    def __init__(self, taxonomy_name: str, engine: EmbeddingEngine):
        self._taxonomy_name = taxonomy_name
        self._engine = engine
        self._hash = _model_hash(engine.model_id)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load_or_compute(
        self,
        level: str,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Return embeddings for the given taxonomy level.

        If a valid cached file exists (matching model hash + node count),
        load and return it silently.  Otherwise compute embeddings, save to
        disk, and return the result.

        Args:
            level:         One of "segment", "family", "class", "commodity"
            texts:         Pre-built embed text strings, index-aligned with
                           the corresponding TaxonomyStore node list.
            show_progress: Show tqdm bar during computation (hidden on cache hit).
        """
        npy_path, meta_path = self._cache_paths(level)
        n = len(texts)

        if self._is_valid(npy_path, meta_path, n):
            return np.load(npy_path)

        # Compute
        desc = f"Building {level} embeddings ({n} nodes)"
        matrix = self._engine.embed(texts, show_progress=show_progress, desc=desc)

        # Persist
        np.save(npy_path, matrix)
        meta = {
            "taxonomy_name": self._taxonomy_name,
            "level": level,
            "model_id": self._engine.model_id,
            "node_count": n,
            "embed_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return matrix

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _cache_paths(self, level: str) -> tuple[Path, Path]:
        stem = f"{self._taxonomy_name}_{level}_{self._hash}"
        return CACHE_DIR / f"{stem}.npy", CACHE_DIR / f"{stem}.meta.json"

    @staticmethod
    def _is_valid(npy_path: Path, meta_path: Path, expected_count: int) -> bool:
        if not npy_path.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            return False
        return meta.get("node_count") == expected_count


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(
    query: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between query vector(s) and all key vectors.

    Pure NumPy — no scipy dependency.

    Args:
        query: shape (D,) for a single query, or (M, D) for a batch of M queries.
        keys:  shape (N, D) — the taxonomy embedding matrix.

    Returns:
        shape (N,) if query is 1D, or (M, N) if query is 2D.
    """
    eps = 1e-10
    if query.ndim == 1:
        q_norm = query / (np.linalg.norm(query) + eps)
        k_norms = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + eps)
        return (k_norms @ q_norm).astype(np.float32)
    else:
        q_norms = query / (np.linalg.norm(query, axis=1, keepdims=True) + eps)
        k_norms = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + eps)
        return (q_norms @ k_norms.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model_id(model_spec: ModelSpec) -> str:
    """Return a stable, printable string identifier for a model spec."""
    if isinstance(model_spec, str):
        return model_spec
    name = getattr(model_spec, "__name__", None)
    if name:
        return name
    return repr(model_spec)[:64]


def _model_hash(model_id: str) -> str:
    """First 8 hex characters of sha256(model_id)."""
    return hashlib.sha256(model_id.encode()).hexdigest()[:8]


def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]
