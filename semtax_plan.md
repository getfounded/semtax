# semtax

**pip install semtax** — Open source Python library for zero-shot hierarchical taxonomy classification using semantic embeddings.

---

## Positioning
**Zero-shot hierarchical taxonomy classification using semantic embeddings. No training data required.**

This is not a spend tool. It is a general-purpose library that ships with procurement taxonomies as built-in presets. The same engine works for CWE (cybersecurity), arXiv (research), NAICS (industry), CPV (EU procurement), and any custom taxonomy a user provides.

## Vision
A lightweight, pip-installable library that classifies text against any hierarchical taxonomy using semantic similarity. Runs locally — no API keys, no data leaving the user's environment. The gap it fills: no maintained open source Python library does this as a zero-shot approach. Commercial alternatives (Qvalia, Classifast) require paid API calls. HiClass (closest real competitor) requires labeled training data before it can classify anything.

---

## Name
`semtax` — semantic + taxonomy. Short, technically accurate, domain-agnostic. Confirm on pypi.org before publishing.

The web demo (spend-focused CSV upload tool) can still be branded `spendcat` separately. Library name and demo name don't need to match.

---

## Embedding Strategy

### Two Separate Embedding Operations

**1. Taxonomy Embeddings — Pre-computed, cached, ships with library**
- Computed once on first use, stored locally, never recomputed unless model changes
- Cache key is taxonomy name + model name — switching models invalidates cache
- What gets embedded per node: label + full hierarchy path + description (see design decision below)

**2. Input Embeddings — Computed at classification time**
- User's raw spend/purchasing descriptions get embedded at runtime
- Same model as taxonomy embeddings — must match for cosine similarity to be valid
- Preprocessed before embedding (strip part numbers, normalize abbreviations, etc.)

### Full Classification Flow
```
User input descriptions
        ↓
Preprocess (strip noise, normalize)
        ↓
Embed input descriptions [runtime]
        ↓
Compare against cached taxonomy embeddings via cosine similarity
        ↓
Return confidence-scored hierarchical match (Segment → Family → Class → Commodity)
```

### Taxonomy Node Embedding Design Decision
What text gets embedded for each UNSPSC node matters significantly for accuracy. Options in increasing richness:

| Option | Example | Notes |
|--------|---------|-------|
| Label only | `"Toner cartridges"` | Weakest — too short, loses context |
| Label + parent | `"Office supplies > Toner cartridges"` | Better — adds hierarchy context |
| Label + description | `"Toner cartridges — Replaceable ink containers for laser printers"` | Good |
| Full path + description | `"Office Equipment > Office Supplies > Toner cartridges — Replaceable ink containers"` | Best accuracy |

**Decision: embed full hierarchy path + description for each node.** The pre-computation cost is irrelevant since it's cached. The accuracy improvement from richer context is meaningful. This is a conscious architectural decision — be able to explain it.

## Model Flexibility
```python
# Default — works out of the box, no setup
classifier = SemTax(taxonomy="unspsc")

# Swap in any sentence-transformers model for higher accuracy
classifier = SemTax(
    taxonomy="unspsc",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# Or pass a callable — covers OpenAI, Cohere, any provider
import openai
def openai_embed(texts):
    response = openai.embeddings.create(
        input=texts, model="text-embedding-3-small"
    )
    return [r.embedding for r in response.data]

classifier = SemTax(taxonomy="unspsc", embedding_model=openai_embed)
```

**Caching note:** taxonomy embeddings are cached per model. Switching models invalidates the cache and triggers re-embedding. This needs to be handled in the cache key design before Claude Code generates it.

## Custom Taxonomy (Bring Your Own Dataset)
Most organizations don't report against UNSPSC internally — they have their own GL chart of accounts, internal spend category tree, or custom product hierarchy. Custom taxonomy support lets users classify against *their* categories instead of a standard they don't actually use.

**Expected file format (CSV):**
```
code,parent_code,label,description
1000,,Technology,All technology spend
1100,1000,Software,Software licenses and subscriptions
1110,1100,SaaS,Cloud-hosted software services
```

**Interface:**
```python
# Built-in preset
classifier = SemTax(taxonomy="unspsc")

# Custom taxonomy from file
classifier = SemTax(taxonomy="path/to/my_taxonomy.csv")

# Custom taxonomy from DataFrame
classifier = SemTax(taxonomy=my_dataframe)
```

**Required columns:** `code`, `parent_code` (empty for root nodes), `label`
**Optional column:** `description` (improves embedding quality if provided — label + description gets embedded together)

**Validation rules to implement:**
- All parent_codes must reference a code that exists in the file
- No circular references
- At least one root node (empty parent_code)
- Codes must be unique

**Caching:** custom taxonomies are cached by file hash, not filename. Same file = cache hit even if path changes.

This is a V1 or early V2 feature. Do not include in GitHub description until shipped.

## LLM Enrichment (Optional, V2)
Default embeddings get you 70-80% accuracy instantly. For low-confidence items, an optional LLM enrichment layer handles what the embedding model isn't confident about — without sending every row through an API.

```python
results = classifier.classify(descriptions)

# Only send low-confidence items to LLM — cheaper and faster than full LLM pipeline
uncertain = [r for r in results if r.confidence < 0.65]
enriched = classifier.enrich_with_llm(uncertain, api_key="...")
```

## Defending Against "Just Use an LLM"
- **Cost at scale:** Classifying 50k rows through an LLM API costs real money. Local embeddings cost nothing.
- **Speed:** Batch embedding classification is orders of magnitude faster than LLM inference.
- **No internet required:** Sensitive financial/procurement data often can't leave the corporate environment.
- **Structured deterministic output:** LLMs hallucinate codes and format inconsistently. semtax returns clean confidence-scored output every time.
- **No API key:** `pip install semtax` works out of the box.
- One-liner: *"LLMs are reasoning engines, not classification infrastructure."*

---

## V1 Features (Library)
- Hierarchical classification: Segment → Family → Class → Commodity (conditional)
- Confidence score at each level in structured output
- Configurable confidence threshold for commodity population
- Description preprocessing pipeline (strip part numbers, abbreviations, noise)
- Batch processing with tqdm progress bar
- Anonymous opt-out telemetry via PostHog (usage patterns only — never description text)

## V1 Features (Web Demo)
- CSV upload with column mapping UI
- Returns classified results as downloadable CSV
- Spend distribution visualization by UNSPSC Segment (if spend amount column provided)
- Stack: FastAPI backend + HTMX + Tailwind/DaisyUI
- Hosted on Railway or Render
- "Your data never leaves your browser" is NOT accurate for server-side — don't claim it

---

## Output Structure
```python
{
  "description": "laptop repair service",
  "segment": {"code": "43000000", "label": "IT Equipment", "confidence": 0.91},
  "family": {"code": "43190000", "label": "...", "confidence": 0.84},
  "class": {"code": "43191500", "label": "...", "confidence": 0.79},
  "commodity": {"code": "43191501", "label": "...", "confidence": 0.71, "populated": true},
  "match_level": "commodity"
}
```

---

## Telemetry (PostHog)
Capture anonymously, opt-out with single flag:
- Descriptions classified per call
- Confidence score distributions
- UNSPSC segments matched (usage patterns)
- Library version, Python version
- Web demo vs library usage

**Never capture:** actual description text or spend amounts.

---

## Download Tracking
- **pypistats.org** — daily/weekly/monthly download counts (automatic, no setup)
- **pepy.tech** — cumulative badge for README
- **PostHog** — actual usage telemetry (active installs vs. passive downloads)

---

## Built-in Taxonomy Presets (Roadmap)
| Version | Taxonomy | Audience |
|---------|----------|----------|
| V1 | UNSPSC | Procurement, finance, FP&A |
| V2 | CWE (Common Weakness Enumeration) | Security engineers, bug bounty, DevSecOps |
| V2 | NAICS | Industry classification, vendor analysis |
| V3 | arXiv subject categories | ML researchers, RAG over research corpora |
| V3 | CPV | European public procurement |
| Any | Custom (bring your own CSV/JSON) | General developer use |

## Taxonomy Data Sources & Bundling Strategy

### UNSPSC
- **Source:** `undp.org/unspsc` — free download, requires clicking through on site (no programmatic URL)
- **Scale:** ~157,000 codes across 4 levels (Segment, Family, Class, Commodity)
- **Format:** Excel download, convert to CSV for bundling
- **Licensing:** Check redistribution terms on UNDP download page before bundling. If redistribution not permitted, implement fetch-on-first-use to `undp.org/unspsc` instead
- **Avoid:** UNGM export (only 13k codes, incomplete), O*NET reference (only 4k codes, subset)
- **Strategy:** Bundle static CSV in package for V1. No download command — zero friction.

### CWE (V2)
- **Source:** `https://cwe.mitre.org/data/xml/cwec_latest.xml.zip` — stable public URL, no auth required
- **Scale:** ~900 weaknesses in Software Development view
- **Format:** Dense XML — NOT directly usable. Requires one-time parsing script to extract clean hierarchy
- **Relevant views:** Software Development (primary use case), Hardware Design (niche, skip V1), Research Concepts (academic, skip)
- **External mappings (CWE Top 25, OWASP, etc.):** Not taxonomies — skip
- **Parsing approach:** Use ElementTree to extract `ID`, `Name`, `Description`, and `Related_Weaknesses` (parent-child relationships). Reference implementations:
  - `github.com/lirantal/cwe-sdk` — pre-built `cwe-dictionary.json` and `cwe-hierarchy.json`
  - `github.com/dsto97/CWEParser` — simple Python ElementTree parser
- **Key challenge:** CWE is a graph, not a clean tree. Weaknesses can have multiple parents. Need to pick a specific view (Software Development) and handle graph-to-tree flattening
- **Strategy:** Write a one-time `/scripts/parse_cwe.py` data prep script, commit output CSV to repo, never run again unless updating taxonomy version. Script is not user-facing.

### No Download Commands in V1
Bundling taxonomies directly in the package is non-negotiable for V1. `pip install semtax` must work immediately with zero setup. Any required download step kills adoption. The optional `semtax update cwe` CLI command is a V2 quality-of-life feature only.

---

## Competitive Landscape
| Tool | Type | Gap |
|------|------|-----|
| HiClass (scikit-learn-contrib) | Real, maintained pip library | Requires labeled training data — supervised, not zero-shot |
| ClassiCore | GitHub (1 star) | Docker app, not a library, no adoption |
| Qvalia / Classifast | Paid APIs | Data leaves environment, per-transaction cost |
| CVE-CWE (GitHub) | Single-purpose script | No stars, not pip-installable, narrow use case |
| Old GitHub repos | 2019 BiLSTM/Random Forest | Unmaintained, require training data |

## The scikit-learn / HiClass Answer
When asked "why not just use scikit-learn?":

HiClass and sklearn-based hierarchical classifiers require labeled training data. You need hundreds of pre-labeled examples per class before the model can classify anything. Most people don't have that — they have a messy export and a deadline.

This library requires nothing except the text to classify. The taxonomy itself is the training signal. It works on day one with zero labeled examples because it understands language semantically rather than learning from statistical patterns in labeled data.

One sentence version: *"scikit-learn is great if you have training data. If you don't, you'd spend weeks labeling before classifying a single item. This works immediately."*

---

## What NOT to Do
- Don't mention municipal procurement anywhere — keep it generic
- Don't add NAICS to V1 — delays release, save for V2
- Don't use Transformers.js for V1 demo — server-side first, port later if traction warrants it
- Don't let Claude Code make architectural decisions you can't explain
- The web demo can be branded `spendcat` — library and demo names don't need to match