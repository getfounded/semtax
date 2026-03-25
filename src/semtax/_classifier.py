"""
Hybrid classifier: top-down traversal + flat class search, reconciled at
the class level, with conditional commodity drill-down.

Classification flow per input:
  1. Embed input (batched)
  2. Path 1 — top-down: Segment → Family (within Segment) → Class (within Family)
  3. Path 2 — flat class search: cosine vs all ~900 classes, pick top-K
  4. Reconcile at class level:
       - Paths agree → use that class, proceed to commodity drill-down
       - Paths disagree → use flat class (more reliable), flag disagreement
  5. Commodity drill-down (only if class confidence ≥ threshold):
       - Cosine vs commodities in matched class range (20-50 nodes)
       - Populate commodity only if commodity confidence ≥ threshold
  6. detect_flags() → collect ambiguity signals
  7. Assemble ClassificationResult

Performance notes:
  - Reverse-index dicts (_seg_code_to_row, etc.) are built during warm_up()
    to avoid O(N) list.index() calls in the inner batch loop.
  - Commodity matrix is accessed via pre-computed class range slices
    (no per-class array allocation).
  - Top-down Family/Class steps group inputs by their winning parent to
    enable vectorised sub-batch matrix multiplies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._ambiguity import AmbiguityConfig, detect_flags
from ._embeddings import EmbeddingEngine, TaxonomyEmbeddingCache, cosine_similarity_matrix
from ._result import ClassificationResult, LevelResult
from ._taxonomy import TaxonomyNode, TaxonomyStore


# ---------------------------------------------------------------------------
# Internal score bundle
# ---------------------------------------------------------------------------


@dataclass
class _TopDown:
    """Top-down path result for a single input."""
    seg_node: TaxonomyNode
    fam_node: TaxonomyNode
    cls_node: TaxonomyNode
    seg_score: float
    fam_score: float
    cls_score: float
    seg_score2: float   # runner-up (for margin tracking)
    fam_score2: float
    cls_score2: float


# ---------------------------------------------------------------------------
# HybridClassifier
# ---------------------------------------------------------------------------


class HybridClassifier:
    """
    Runs top-down traversal and flat class search in parallel, reconciles
    at class level, drills to commodity if confidence warrants it.

    Designed to be instantiated once and reused across many classify() calls.
    All heavy state (embedding matrices, store) is held by reference after
    warm_up() loads them.
    """

    def __init__(
        self,
        store: TaxonomyStore,
        engine: EmbeddingEngine,
        cache: TaxonomyEmbeddingCache,
        config: AmbiguityConfig,
    ):
        self._store = store
        self._engine = engine
        self._cache = cache
        self._config = config

        # Embedding matrices — populated by warm_up()
        self._seg_matrix: Optional[np.ndarray] = None
        self._fam_matrix: Optional[np.ndarray] = None
        self._cls_matrix: Optional[np.ndarray] = None
        self._com_matrix: Optional[np.ndarray] = None

        # Reverse-index dicts — populated by warm_up()
        self._seg_code_to_row: dict[str, int] = {}
        self._fam_code_to_row: dict[str, int] = {}
        self._cls_code_to_row: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warm_up(self, show_progress: bool = True) -> None:
        """
        Load (or compute and cache) all taxonomy embedding matrices.

        Called once from SemTax.__init__.  Commodity matrix is ~230MB;
        loading from .npy cache takes ~1s on typical hardware.  First-time
        computation takes longer (~10 min on CPU for all-MiniLM-L6-v2).
        """
        store = self._store
        self._seg_matrix = self._cache.load_or_compute(
            "segment", store.segment_embed_texts, show_progress
        )
        self._fam_matrix = self._cache.load_or_compute(
            "family", store.family_embed_texts, show_progress
        )
        self._cls_matrix = self._cache.load_or_compute(
            "class", store.class_embed_texts, show_progress
        )
        self._com_matrix = self._cache.load_or_compute(
            "commodity", store.commodity_embed_texts, show_progress
        )

        # Build O(1) reverse-index dicts
        self._seg_code_to_row = {n.code: i for i, n in enumerate(store.segments)}
        self._fam_code_to_row = {n.code: i for i, n in enumerate(store.families)}
        self._cls_code_to_row = {n.code: i for i, n in enumerate(store.classes)}

    # ------------------------------------------------------------------
    # Public classify
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[ClassificationResult]:
        """
        Classify a batch of input descriptions.

        Args:
            texts:         Input strings to classify, as-is.
            show_progress: Show tqdm bar while embedding inputs.
        """
        self._ensure_warm()

        M = len(texts)

        # Step 1: Embed all inputs
        query_matrix = self._engine.embed(
            texts,
            show_progress=show_progress,
            desc="Classifying",
        )  # (M, D)

        # Step 2: Top-down path (Segment → Family → Class)
        top_down_results = self._run_top_down(query_matrix)  # list[_TopDown], len M

        # Step 3: Flat class search across all classes
        cls_scores_flat = cosine_similarity_matrix(query_matrix, self._cls_matrix)
        # (M, N_cls)
        K = self._config.top_k_spread
        top_k_cls_indices = np.argsort(cls_scores_flat, axis=1)[:, -K:][:, ::-1]
        # (M, K) — top-K class indices per input, best first

        # Step 4-7: Reconcile, drill to commodity, assemble results
        results: list[ClassificationResult] = []
        for i in range(M):
            td = top_down_results[i]
            flat_cls_idx = int(top_k_cls_indices[i, 0])
            flat_cls_score = float(cls_scores_flat[i, flat_cls_idx])
            flat_cls_score2 = float(cls_scores_flat[i, int(top_k_cls_indices[i, 1])]) if K > 1 else 0.0
            flat_cls_node = self._store.classes[flat_cls_idx]

            # Segment codes for top-K classes (for spread flag)
            top_k_seg_codes = [
                self._resolve_class_segment(self._store.classes[int(top_k_cls_indices[i, k])])
                for k in range(min(K, top_k_cls_indices.shape[1]))
            ]

            result = self._reconcile_and_build(
                text=texts[i],
                td=td,
                flat_cls_node=flat_cls_node,
                flat_cls_score=flat_cls_score,
                flat_cls_score2=flat_cls_score2,
                top_k_seg_codes=top_k_seg_codes,
                query_vec=query_matrix[i],
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Top-down path
    # ------------------------------------------------------------------

    def _run_top_down(self, query_matrix: np.ndarray) -> list[_TopDown]:
        """
        Run the top-down Segment → Family → Class path for all M inputs.

        Groups inputs by their winning parent at each level to enable
        vectorised sub-batch matrix multiplies rather than per-input loops.
        """
        M = query_matrix.shape[0]
        store = self._store

        # --- Segment step ---
        seg_scores = cosine_similarity_matrix(query_matrix, self._seg_matrix)
        # (M, N_seg)
        top_seg_idx = np.argmax(seg_scores, axis=1)  # (M,)

        results: list[Optional[_TopDown]] = [None] * M

        # Group inputs by their winning Segment
        seg_groups: dict[str, list[int]] = {}
        for i in range(M):
            seg_code = store.segments[int(top_seg_idx[i])].code
            seg_groups.setdefault(seg_code, []).append(i)

        for seg_code, seg_input_indices in seg_groups.items():
            fam_nodes = store.families_by_segment.get(seg_code, [])
            if not fam_nodes:
                # Segment has no families — fall back to segment-only result
                for i in seg_input_indices:
                    seg_node = store.segments[int(top_seg_idx[i])]
                    results[i] = _TopDown(
                        seg_node=seg_node, fam_node=seg_node, cls_node=seg_node,
                        seg_score=float(seg_scores[i, int(top_seg_idx[i])]),
                        fam_score=0.0, cls_score=0.0,
                        seg_score2=_second_score(seg_scores[i], int(top_seg_idx[i])),
                        fam_score2=0.0, cls_score2=0.0,
                    )
                continue

            fam_row_indices = [self._fam_code_to_row[n.code] for n in fam_nodes]
            fam_sub_matrix = self._fam_matrix[fam_row_indices]  # (F, D)
            sub_q = query_matrix[seg_input_indices]              # (G, D)
            fam_scores_sub = cosine_similarity_matrix(sub_q, fam_sub_matrix)
            # (G, F)
            top_fam_local = np.argmax(fam_scores_sub, axis=1)   # (G,)

            # Group by winning Family
            fam_groups: dict[str, list[tuple[int, int]]] = {}
            for g, i in enumerate(seg_input_indices):
                fam_code = fam_nodes[int(top_fam_local[g])].code
                fam_groups.setdefault(fam_code, []).append((g, i))

            for fam_code, gf_pairs in fam_groups.items():
                cls_nodes = store.classes_by_family.get(fam_code, [])
                gf_g_indices = [p[0] for p in gf_pairs]
                gf_i_indices = [p[1] for p in gf_pairs]

                if not cls_nodes:
                    for g, i in gf_pairs:
                        seg_node = store.segments[int(top_seg_idx[i])]
                        fam_node = fam_nodes[int(top_fam_local[g])]
                        results[i] = _TopDown(
                            seg_node=seg_node, fam_node=fam_node, cls_node=fam_node,
                            seg_score=float(seg_scores[i, int(top_seg_idx[i])]),
                            fam_score=float(fam_scores_sub[g, int(top_fam_local[g])]),
                            cls_score=0.0,
                            seg_score2=_second_score(seg_scores[i], int(top_seg_idx[i])),
                            fam_score2=_second_score(fam_scores_sub[g], int(top_fam_local[g])),
                            cls_score2=0.0,
                        )
                    continue

                cls_row_indices = [self._cls_code_to_row[n.code] for n in cls_nodes]
                cls_sub_matrix = self._cls_matrix[cls_row_indices]  # (C, D)
                sub_q2 = query_matrix[gf_i_indices]                 # (GF, D)
                cls_scores_sub = cosine_similarity_matrix(sub_q2, cls_sub_matrix)
                # (GF, C)
                top_cls_local = np.argmax(cls_scores_sub, axis=1)   # (GF,)

                for gf_pos, (g, i) in enumerate(gf_pairs):
                    seg_node = store.segments[int(top_seg_idx[i])]
                    fam_node = fam_nodes[int(top_fam_local[g])]
                    cls_node = cls_nodes[int(top_cls_local[gf_pos])]

                    results[i] = _TopDown(
                        seg_node=seg_node,
                        fam_node=fam_node,
                        cls_node=cls_node,
                        seg_score=float(seg_scores[i, int(top_seg_idx[i])]),
                        fam_score=float(fam_scores_sub[g, int(top_fam_local[g])]),
                        cls_score=float(cls_scores_sub[gf_pos, int(top_cls_local[gf_pos])]),
                        seg_score2=_second_score(seg_scores[i], int(top_seg_idx[i])),
                        fam_score2=_second_score(fam_scores_sub[g], int(top_fam_local[g])),
                        cls_score2=_second_score(cls_scores_sub[gf_pos], int(top_cls_local[gf_pos])),
                    )

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Reconciliation and result assembly
    # ------------------------------------------------------------------

    def _reconcile_and_build(
        self,
        text: str,
        td: _TopDown,
        flat_cls_node: TaxonomyNode,
        flat_cls_score: float,
        flat_cls_score2: float,
        top_k_seg_codes: list[str],
        query_vec: np.ndarray,
    ) -> ClassificationResult:
        """
        Reconcile top-down and flat class results, drill to commodity if
        warranted, then assemble a ClassificationResult.

        Reconciliation rules:
        - If both paths agree on class code → use that class, take higher
          of the two class confidence scores.
        - If they disagree → use the flat class result as the anchor (more
          reliable since it searches all classes), use flat score.

        Confidence source for parent levels:
        - Segment and Family confidence always come from the top-down path
          (reliable at upper levels regardless of class agreement).
        """
        paths_agree = (td.cls_node.code == flat_cls_node.code)

        if paths_agree:
            winning_cls = flat_cls_node
            cls_score = max(td.cls_score, flat_cls_score)
        else:
            winning_cls = flat_cls_node
            cls_score = flat_cls_score

        # Resolve Segment and Family from the flat class's ancestry
        # (even when top-down disagrees, we use flat class as anchor)
        fam_node = self._store.family_by_code.get(winning_cls.parent_code)
        seg_node: Optional[TaxonomyNode] = None
        if fam_node:
            seg_node = self._store.segment_by_code.get(fam_node.parent_code)

        # Fall back to top-down ancestors if lookup fails
        if fam_node is None:
            fam_node = td.fam_node
        if seg_node is None:
            seg_node = td.seg_node

        # Commodity drill-down (only if class is confident enough)
        com_node: Optional[TaxonomyNode] = None
        com_score = 0.0
        com_populated = False

        if cls_score >= self._config.class_confidence_threshold:
            com_node, com_score = self._drill_commodity(winning_cls.code, query_vec)
            if com_score >= self._config.commodity_confidence_threshold:
                com_populated = True

        # Determine match level
        if com_populated:
            match_level = "commodity"
        elif cls_score >= self._config.class_confidence_threshold:
            match_level = "class"
        elif td.fam_score >= self._config.class_confidence_threshold:
            match_level = "family"
        else:
            match_level = "segment"

        # Detect flags (using class scores as the primary signal)
        flags = detect_flags(
            description=text,
            class_top1_score=cls_score,
            class_top2_score=flat_cls_score2,
            top_k_class_segment_codes=top_k_seg_codes,
            config=self._config,
        )

        # Placeholder commodity when not populated
        if com_node is None:
            com_node = winning_cls  # use class as placeholder (code/label shown)

        return ClassificationResult(
            description=text,
            segment=LevelResult(
                code=seg_node.code,
                label=seg_node.label,
                confidence=round(td.seg_score, 4),
            ),
            family=LevelResult(
                code=fam_node.code,
                label=fam_node.label,
                confidence=round(td.fam_score, 4),
            ),
            class_=LevelResult(
                code=winning_cls.code,
                label=winning_cls.label,
                confidence=round(cls_score, 4),
            ),
            commodity=LevelResult(
                code=com_node.code,
                label=com_node.label,
                confidence=round(com_score, 4),
                populated=com_populated,
            ),
            match_level=match_level,
            flags=flags,
            _top_down_path=[
                LevelResult(td.seg_node.code, td.seg_node.label, round(td.seg_score, 4)),
                LevelResult(td.fam_node.code, td.fam_node.label, round(td.fam_score, 4)),
                LevelResult(td.cls_node.code, td.cls_node.label, round(td.cls_score, 4)),
            ],
            _flat_class_match=LevelResult(
                flat_cls_node.code, flat_cls_node.label, round(flat_cls_score, 4)
            ),
        )

    # ------------------------------------------------------------------
    # Commodity drill-down
    # ------------------------------------------------------------------

    def _drill_commodity(
        self, class_code: str, query_vec: np.ndarray
    ) -> tuple[Optional[TaxonomyNode], float]:
        """
        Search commodities within a specific class.

        Uses the pre-computed class_code_to_commodity_range slice to avoid
        allocating separate per-class arrays.

        Returns (best_commodity_node, best_cosine_score).
        Returns (None, 0.0) if the class has no commodities in the store.
        """
        range_ = self._store.class_code_to_commodity_range.get(class_code)
        if range_ is None:
            return None, 0.0

        start, end = range_
        if start >= end:
            return None, 0.0

        com_sub_matrix = self._com_matrix[start:end]  # (K, D) — no copy
        scores = cosine_similarity_matrix(query_vec, com_sub_matrix)  # (K,)
        best_local = int(np.argmax(scores))
        best_score = float(scores[best_local])
        best_node = self._store.commodities[start + best_local]
        return best_node, best_score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_class_segment(self, cls_node: TaxonomyNode) -> str:
        """Return the Segment code for a Class node via parent chain."""
        fam = self._store.family_by_code.get(cls_node.parent_code)
        if fam:
            return fam.parent_code
        return ""

    def _ensure_warm(self) -> None:
        if self._com_matrix is None:
            self.warm_up()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _second_score(scores: np.ndarray, top_idx: int) -> float:
    """Return the score of the runner-up (second-highest value)."""
    if len(scores) <= 1:
        return 0.0
    masked = scores.copy()
    masked[top_idx] = -1.0
    return float(np.max(masked))
