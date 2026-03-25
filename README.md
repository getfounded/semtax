# semtax

**Zero-shot hierarchical taxonomy classification using semantic embeddings.**

```bash
pip install semtax
```

No training data. No API keys. No labeled examples. Point it at text, get back a taxonomy match.

---

## What is this?

`semtax` classifies free-text descriptions against a hierarchical taxonomy — instantly, locally, without any setup beyond the install.

The core idea: taxonomy nodes already contain rich semantic information in their labels and definitions. By embedding both the input text and the taxonomy nodes into the same vector space, we can find the best match using cosine similarity. No supervised learning required.

**The gap this fills:** most existing tools either require labeled training data (HiClass, scikit-learn pipelines) or charge per API call (Qvalia, Classifast). `semtax` requires neither.

---

## Quickstart

```python
from semtax import SemTax

classifier = SemTax()

result = classifier.classify("toner cartridges for laser printer")

print(result.segment.label)      # Office Equipment and Accessories and Supplies
print(result.class_.label)       # Toner cartridges and supplies
print(result.class_.confidence)  # 0.8341
print(result.commodity.label)    # Laser toner cartridges
print(result.match_level)        # commodity
```

### Batch classification

```python
descriptions = [
    "laptop battery replacement",
    "janitorial cleaning services",
    "annual software license renewal",
    "server rack unit 2U",
]

results = classifier.classify(descriptions)

for r in results:
    print(f"{r.description:<40} → {r.class_.label} ({r.class_.confidence:.2f})")
```

### Classify from a CSV file

```python
from semtax import SemTax

classifier = SemTax()

# Auto-detects a column named "description", "item", "name", "text", etc.
output = classifier.classify_csv("spend_data.csv")
output.to_csv("spend_data_classified.csv", index=False)
```

If your column has a non-standard name, pass it explicitly:

```python
output = classifier.classify_csv("spend_data.csv", column="line_item")
```

The output is a DataFrame with all your original columns plus `segment_code`, `segment_label`, `segment_confidence`, `family_code`, `family_label`, `class_code`, `class_label`, `class_confidence`, `commodity_code`, `commodity_label`, `commodity_populated`, `match_level`, and `flags`.

### Export a list to CSV

```python
import pandas as pd

results = classifier.classify(descriptions)
df = pd.DataFrame([r.to_flat_dict() for r in results])
df.to_csv("classified.csv", index=False)
```

---

## How it works

Classification runs two paths in parallel and reconciles them:

**Path 1 — top-down:** Segment → Family → Class, drilling down the hierarchy at each level.

**Path 2 — flat class search:** cosine similarity against all ~900 classes directly, giving a strong semantic anchor without the noise of 157k commodity-level comparisons.

Both paths are reconciled at the class level. If they agree, confidence is high. If they disagree, the result is flagged. Once a class is matched above the confidence threshold, commodity drill-down searches only the 20-50 commodities within that class — a tractable scope where fine-grained distinctions are reliable.

**Taxonomy embeddings are cached** on first use at `~/.semtax/cache/`. Subsequent runs load from disk in ~1 second.

---

## Confidence and ambiguity flags

Every result includes a confidence score at each level. Results that are uncertain are flagged:

```python
result = classifier.classify("IT hardware and software maintenance services")

print(result.flags)
# ['composite_heuristic', 'multi_segment_spread']

print(result.class_.confidence)   # 0.61
print(result.commodity.populated) # False — stopped at class level
```

| Flag | Meaning |
|------|---------|
| `low_confidence` | Best class match scored below threshold |
| `margin_too_small` | Top-1 and top-2 class scores are too close to call |
| `multi_segment_spread` | Top matches span multiple segments — ambiguous input |
| `composite_heuristic` | Input likely describes multiple distinct items |

### Configuring thresholds

```python
classifier = SemTax(
    class_confidence_threshold=0.55,     # flag low_confidence below this
    commodity_confidence_threshold=0.68, # stop at class level below this
)
```

---

## Custom embedding models

```python
# Higher accuracy, slower — still local
classifier = SemTax(
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# OpenAI embeddings
import openai

def openai_embed(texts):
    resp = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    return [r.embedding for r in resp.data]

classifier = SemTax(embedding_model=openai_embed)
```

Any callable with signature `(list[str]) -> list[list[float]]` works.

---

## What's available to import

```python
from semtax import (
    SemTax,               # the classifier
    AmbiguityConfig,      # fine-grained threshold configuration
    ClassificationResult, # return type from classify() — useful for type hints
    LevelResult,          # segment / family / class / commodity result type
    FLAG_LOW_CONFIDENCE,        # filter results by flag
    FLAG_MARGIN_TOO_SMALL,
    FLAG_MULTI_SEGMENT_SPREAD,
    FLAG_COMPOSITE_HEURISTIC,
)
```

### AmbiguityConfig

For control beyond the two threshold kwargs:

```python
from semtax import SemTax, AmbiguityConfig

config = AmbiguityConfig(
    class_confidence_threshold=0.55,
    commodity_confidence_threshold=0.68,
    margin_threshold=0.08,   # stricter margin requirement
    top_k_spread=3,          # check top-3 classes for segment spread (default 5)
)

classifier = SemTax(config=config)
```

### Filtering by flag

```python
results = classifier.classify(descriptions)

# Only keep clean, high-confidence results
clean = [r for r in results if not r.flags]

# Find everything that looks composite
composite = [r for r in results if FLAG_COMPOSITE_HEURISTIC in r.flags]
```

---

## Telemetry

`semtax` collects anonymous usage data (batch sizes, taxonomy matched, model used — never description text) via PostHog. Opt out any time:

```python
SemTax(telemetry=False)
```

```bash
SEMTAX_DISABLE_TELEMETRY=1 python your_script.py
```

---

## Roadmap

| Version | Feature |
|---------|---------|
| **V1** | UNSPSC classification, hybrid search, confidence scoring, ambiguity flags |
| **V2** | CWE (cybersecurity weakness classification) |
| **V2** | NAICS (industry/vendor classification) |
| **V2** | LLM enrichment layer for low-confidence items |
| **V2** | Custom taxonomy support (bring your own CSV) |
| **V3** | arXiv subject categories |
| **V3** | CPV (EU public procurement) |

---

## Why not just use an LLM?

- **Cost at scale:** Classifying 50k rows through an LLM API is expensive. Local embeddings cost nothing.
- **Speed:** Batch embedding classification is orders of magnitude faster than LLM inference.
- **No data leaving your environment:** Sensitive procurement or financial data often can't touch external APIs.
- **Deterministic output:** LLMs hallucinate codes and format output inconsistently. `semtax` returns clean, structured results every time.

*LLMs are reasoning engines, not classification infrastructure.*

---

## License

MIT
