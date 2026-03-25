"""
semtax — Zero-shot hierarchical taxonomy classification using semantic embeddings.

    pip install semtax

Quick start::

    from semtax import SemTax

    classifier = SemTax()
    result = classifier.classify("toner cartridges for laser printer")
    print(result.class_.label, result.class_.confidence)
    print(result.commodity.label, result.commodity.populated)

    # Batch
    results = classifier.classify(["laptop repair", "office chairs"])
"""

from ._api import SemTax
from ._ambiguity import AmbiguityConfig
from ._embeddings import DEFAULT_MODEL
from ._result import (
    ClassificationResult,
    FLAG_COMPOSITE_HEURISTIC,
    FLAG_LOW_CONFIDENCE,
    FLAG_MARGIN_TOO_SMALL,
    FLAG_MULTI_SEGMENT_SPREAD,
    LevelResult,
)

__version__ = "0.1.0"

__all__ = [
    "SemTax",
    "ClassificationResult",
    "LevelResult",
    "AmbiguityConfig",
    "DEFAULT_MODEL",
    "FLAG_MARGIN_TOO_SMALL",
    "FLAG_MULTI_SEGMENT_SPREAD",
    "FLAG_LOW_CONFIDENCE",
    "FLAG_COMPOSITE_HEURISTIC",
    "__version__",
]
