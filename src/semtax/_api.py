"""
Public SemTax class — the single entry point for library users.

Usage:
    from semtax import SemTax

    classifier = SemTax()
    result = classifier.classify("toner cartridges for laser printer")
    results = classifier.classify(["laptop repair", "office paper", "server"])
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Callable, Optional, Union

from ._ambiguity import AmbiguityConfig
from ._classifier import HybridClassifier
from ._embeddings import DEFAULT_MODEL, EmbeddingEngine, TaxonomyEmbeddingCache

from ._result import ClassificationResult
from ._taxonomy import TaxonomyStore, load_unspsc
from ._telemetry import capture

ModelSpec = Union[str, Callable[[list[str]], list[list[float]]]]

_COLUMN_CANDIDATES: list[str] = [
    "description", "item_description", "desc", "item", "name", "text"
]

_FLAT_DICT_FIELDS: list[str] = [
    "description",
    "segment_code", "segment_label", "segment_confidence",
    "family_code", "family_label", "family_confidence",
    "class_code", "class_label", "class_confidence",
    "commodity_code", "commodity_label", "commodity_confidence",
    "commodity_populated", "match_level", "flags",
]


class SemTax:
    """
    Zero-shot hierarchical taxonomy classifier.

    Classifies free-text descriptions against a hierarchical taxonomy using
    semantic embeddings.  No training data required — works immediately on
    any text.

    Args:
        taxonomy:
            Which taxonomy to classify against.  Built-in preset: ``"unspsc"``.
            Pass a file path (str) to use a custom CSV taxonomy.
            Custom CSV format: columns [code, parent_code, label, description].

        embedding_model:
            sentence-transformers model name string, or any callable with
            signature ``(list[str]) -> list[list[float]]``.
            Default: ``"sentence-transformers/all-MiniLM-L6-v2"`` (downloaded
            automatically on first use, ~90 MB, CPU-compatible).

        class_confidence_threshold:
            Minimum cosine score for the flat class search result before
            ``FLAG_LOW_CONFIDENCE`` is raised.  Also gates commodity drill-down:
            below this threshold, the result stops at the class level.
            Default: ``0.60``.

        commodity_confidence_threshold:
            Minimum cosine score for the commodity drill-down before commodity
            is populated in the result.  Below this, ``match_level`` is
            ``"class"`` and ``commodity.populated`` is False.
            Default: ``0.72``.

        config:
            Full ``AmbiguityConfig`` instance for fine-grained threshold control.
            If provided, ``class_confidence_threshold`` and
            ``commodity_confidence_threshold`` kwargs override the config values.

        telemetry:
            Set ``False`` to disable anonymous usage telemetry.  Can also be
            disabled process-wide via ``SEMTAX_DISABLE_TELEMETRY=1``.

        verbose:
            Set ``False`` to suppress warm-up progress bars.

    Examples::

        # Default — UNSPSC, all-MiniLM-L6-v2
        c = SemTax()
        r = c.classify("laptop battery replacement")

        # Custom model
        c = SemTax(embedding_model="sentence-transformers/all-mpnet-base-v2")

        # OpenAI embeddings
        def openai_embed(texts):
            import openai
            resp = openai.embeddings.create(input=texts, model="text-embedding-3-small")
            return [r.embedding for r in resp.data]
        c = SemTax(embedding_model=openai_embed)

        # Adjust thresholds
        c = SemTax(class_confidence_threshold=0.55, commodity_confidence_threshold=0.68)

        # Batch
        results = c.classify(["toner cartridges", "office chairs", "server racks"])
    """

    def __init__(
        self,
        taxonomy: str = "unspsc",
        embedding_model: ModelSpec = DEFAULT_MODEL,
        class_confidence_threshold: float = 0.60,
        commodity_confidence_threshold: float = 0.72,
        config: Optional[AmbiguityConfig] = None,
        telemetry: bool = True,
        verbose: bool = True,
    ):
        self._telemetry = telemetry
        self._verbose = verbose

        # Build AmbiguityConfig — explicit kwargs take precedence over config
        if config is None:
            self._config = AmbiguityConfig(
                class_confidence_threshold=class_confidence_threshold,
                commodity_confidence_threshold=commodity_confidence_threshold,
            )
        else:
            self._config = AmbiguityConfig(
                class_confidence_threshold=class_confidence_threshold,
                commodity_confidence_threshold=commodity_confidence_threshold,
                margin_threshold=config.margin_threshold,
                top_k_spread=config.top_k_spread,
                min_noun_count=config.min_noun_count,
            )

        # Load taxonomy store
        if taxonomy == "unspsc":
            self._taxonomy_name = "unspsc"
            self._store: TaxonomyStore = load_unspsc()
        else:
            self._taxonomy_name = "custom"
            from ._taxonomy import load_custom_taxonomy  # type: ignore[attr-defined]
            self._store = load_custom_taxonomy(taxonomy)

        # Build embedding engine and cache
        self._engine = EmbeddingEngine(model_spec=embedding_model)
        self._cache = TaxonomyEmbeddingCache(
            taxonomy_name=self._taxonomy_name,
            engine=self._engine,
        )

        # Build classifier
        self._classifier = HybridClassifier(
            store=self._store,
            engine=self._engine,
            cache=self._cache,
            config=self._config,
        )

        # Warm up — loads / computes taxonomy embedding matrices
        self._classifier.warm_up(show_progress=verbose)

        capture(
            "classifier_initialized",
            {
                "taxonomy": self._taxonomy_name,
                "model_id": self._engine.model_id,
                "python_version": sys.version.split()[0],
            },
            opt_out=not telemetry,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        descriptions: Union[str, list[str]],
        show_progress: bool = False,
    ) -> Union[ClassificationResult, list[ClassificationResult]]:
        """
        Classify one or more text descriptions.

        Args:
            descriptions:
                A single string, or a list of strings.
                Single string → returns a single ``ClassificationResult``.
                List of strings → returns a list, same length and order.

            show_progress:
                Show a tqdm progress bar while embedding the input batch.
                Useful for large batches (>1000 items).

        Returns:
            ``ClassificationResult`` (single) or ``list[ClassificationResult]`` (batch).

        Raises:
            ValueError: If any input string is empty or whitespace-only.
        """
        single = isinstance(descriptions, str)
        texts: list[str] = [descriptions] if single else list(descriptions)

        if not texts:
            return [] if not single else None  # type: ignore[return-value]

        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(
                    f"Empty description at index {i}. "
                    "All inputs must be non-empty strings."
                )

        results = self._classifier.classify_batch(
            texts=texts,
            show_progress=show_progress,
        )

        capture(
            "classify_called",
            {"batch_size": len(texts), "taxonomy": self._taxonomy_name},
            opt_out=not self._telemetry,
        )

        return results[0] if single else results

    # ------------------------------------------------------------------
    # Import methods
    # ------------------------------------------------------------------

    def classify_csv(
        self,
        path: str,
        column: Optional[str] = None,
        show_progress: bool = True,
    ) -> "pd.DataFrame":
        """
        Classify descriptions from a CSV file and return a DataFrame with
        the original columns plus classification results appended.

        Args:
            path:
                Path to the CSV file.

            column:
                Name of the column containing descriptions to classify.
                If omitted, semtax looks for a column named one of:
                ``description``, ``item_description``, ``desc``, ``item``,
                ``name``, ``text`` (case-insensitive, first match wins).
                Pass ``column`` explicitly if your column has a different name.

            show_progress:
                Show a tqdm progress bar while classifying. Default ``True``
                since CSV files tend to be large.

        Returns:
            A pandas DataFrame with all original columns plus classification
            columns appended.

        Raises:
            ImportError:  If pandas is not installed.
            ValueError:   If the description column cannot be found.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for classify_csv. "
                "Install it with: pip install pandas"
            )

        df = pd.read_csv(path)
        col = self._detect_column(list(df.columns), column, path)
        results = self.classify(df[col].tolist(), show_progress=show_progress)
        classified = pd.DataFrame([r.to_flat_dict() for r in results])
        return pd.concat([df, classified.drop(columns=["description"])], axis=1)

    def classify_json(
        self,
        path: str,
        column: Optional[str] = None,
        show_progress: bool = True,
    ) -> "pd.DataFrame":
        """
        Classify descriptions from a JSON file and return a DataFrame with
        the original data plus classification results appended.

        The JSON file must contain a top-level list. Two formats are accepted:

        - **List of strings** — each string is treated as a description directly.
          The ``column`` parameter is ignored in this case.
        - **List of dicts** — semtax auto-detects the description column using
          the same logic as ``classify_csv`` (or use ``column`` explicitly).

        Args:
            path:
                Path to the JSON file.

            column:
                Name of the key containing descriptions (list-of-dicts only).
                Auto-detected if omitted.

            show_progress:
                Show a tqdm progress bar while classifying.

        Returns:
            A pandas DataFrame with original data plus classification columns.

        Raises:
            ImportError:  If pandas is not installed.
            ValueError:   If the file does not contain a top-level list, or if
                          the description column cannot be found.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for classify_json. "
                "Install it with: pip install pandas"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"JSON file must contain a top-level list, "
                f"but got {type(data).__name__}. "
                "Expected a list of strings or a list of dicts."
            )

        if len(data) == 0:
            return pd.DataFrame(columns=_FLAT_DICT_FIELDS)

        if isinstance(data[0], str):
            df = pd.DataFrame({"description": data})
            col = "description"
        elif isinstance(data[0], dict):
            df = pd.DataFrame(data)
            col = self._detect_column(list(df.columns), column, path)
        else:
            raise ValueError(
                f"JSON list elements must be strings or dicts, "
                f"but got {type(data[0]).__name__}."
            )

        results = self.classify(df[col].tolist(), show_progress=show_progress)
        classified = pd.DataFrame([r.to_flat_dict() for r in results])
        return pd.concat([df, classified.drop(columns=["description"])], axis=1)

    def classify_excel(
        self,
        path: str,
        column: Optional[str] = None,
        sheet_name: Union[str, int] = 0,
        show_progress: bool = True,
    ) -> "pd.DataFrame":
        """
        Classify descriptions from an Excel file (.xlsx) and return a DataFrame
        with the original columns plus classification results appended.

        Args:
            path:
                Path to the .xlsx file.

            column:
                Name of the column containing descriptions. Auto-detected if
                omitted (same logic as ``classify_csv``).

            sheet_name:
                Sheet to read. Accepts a sheet name (str) or zero-based index
                (int). Default ``0`` reads the first sheet.

            show_progress:
                Show a tqdm progress bar while classifying.

        Returns:
            A pandas DataFrame with original columns plus classification columns.

        Raises:
            ImportError:  If pandas or openpyxl is not installed.
            ValueError:   If the description column cannot be found.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for classify_excel. "
                "Install it with: pip install pandas openpyxl"
            )
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "openpyxl is required for classify_excel. "
                "Install it with: pip install openpyxl"
            )

        df = pd.read_excel(path, sheet_name=sheet_name)
        col = self._detect_column(list(df.columns), column, path)
        results = self.classify(df[col].tolist(), show_progress=show_progress)
        classified = pd.DataFrame([r.to_flat_dict() for r in results])
        return pd.concat([df, classified.drop(columns=["description"])], axis=1)

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_csv(
        self,
        results: list[ClassificationResult],
        path: str,
    ) -> None:
        """
        Write classification results to a CSV file.

        Does not require pandas — uses the Python standard library ``csv``
        module.

        Args:
            results: List of ``ClassificationResult`` objects from ``classify()``.
            path:    Destination file path.
        """
        rows = [r.to_flat_dict() for r in results]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FLAT_DICT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def to_dataframe(
        self,
        results: list[ClassificationResult],
    ) -> "pd.DataFrame":
        """
        Convert classification results to a pandas DataFrame.

        Args:
            results: List of ``ClassificationResult`` objects from ``classify()``.

        Returns:
            A flat pandas DataFrame with one row per result.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe. "
                "Install it with: pip install pandas"
            )
        if not results:
            return pd.DataFrame(columns=_FLAT_DICT_FIELDS)
        return pd.DataFrame([r.to_flat_dict() for r in results])

    def to_json(
        self,
        results: list[ClassificationResult],
        path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export classification results as JSON (list of flat dicts).

        Args:
            results:
                List of ``ClassificationResult`` objects from ``classify()``.
            path:
                If given, writes to this file path and returns ``None``.
                If omitted, returns the JSON as a string.

        Returns:
            JSON string if ``path`` is ``None``, otherwise ``None``.
        """
        data = [r.to_flat_dict() for r in results]
        if path is None:
            return json.dumps(data, indent=2, ensure_ascii=False)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return None

    def to_excel(
        self,
        results: list[ClassificationResult],
        path: str,
    ) -> None:
        """
        Write classification results to an Excel file (.xlsx).

        Args:
            results: List of ``ClassificationResult`` objects from ``classify()``.
            path:    Destination file path (should end in ``.xlsx``).

        Raises:
            ImportError: If pandas or openpyxl is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_excel. "
                "Install it with: pip install pandas openpyxl"
            )
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "openpyxl is required for to_excel. "
                "Install it with: pip install openpyxl"
            )
        df = (
            pd.DataFrame([r.to_flat_dict() for r in results])
            if results
            else pd.DataFrame(columns=_FLAT_DICT_FIELDS)
        )
        df.to_excel(path, index=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_column(
        columns: list[str],
        explicit: Optional[str],
        source_label: str,
    ) -> str:
        """Resolve the description column name from a list of column names."""
        if explicit is not None:
            if explicit not in columns:
                raise ValueError(
                    f"Column '{explicit}' not found in {source_label}. "
                    f"Columns found: {columns}"
                )
            return explicit
        col_lower = {c.lower(): c for c in columns}
        for candidate in _COLUMN_CANDIDATES:
            if candidate in col_lower:
                return col_lower[candidate]
        raise ValueError(
            f"Could not find a description column in {source_label}. "
            f"Columns found: {columns}. "
            "Pass column='your_column_name' explicitly."
        )

    def __repr__(self) -> str:
        return (
            f"SemTax(taxonomy={self._taxonomy_name!r}, "
            f"model={self._engine.model_id!r}, "
            f"class_threshold={self._config.class_confidence_threshold}, "
            f"commodity_threshold={self._config.commodity_confidence_threshold})"
        )
