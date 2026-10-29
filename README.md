# Contexting Strategies

An engineering lab for a practical RAG question that sits **after retrieval** and **before the LLM call**: **how should you assemble evidence into a prompt under a token budget** without leaking unauthorized text, repeating the same facts, or starving critical passages?

The project retrieves candidates with BM25, runs several **context assembly** strategies inspired by the [context algorithm](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-algorithm.md), and reports assembly-quality metrics from [evaluate assembling](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-evaluate-assembling.md). It is for learning and controlled experiments—not a hosted chat service or a substitute for end-to-end answer evaluation.

## Start here

### Results from one local run (synthetic corpus)

Seven labeled queries, 10 retrieved candidates each, **120-token** context budget (including rendered citation headers). Python **3.14.7** on macOS arm64, 2026-10-29. Corpus and labels are synthetic; do not treat these numbers as production forecasts.

| Strategy | Evidence coverage | Redundancy rate | Context utilization | Unauthorized leak rate | Assembly p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Naive retrieval order** (baseline) | 0.81 | 0.10 | 0.57 | **0.14** (1/7 queries) | 0.04 |
| **Weighted score** \(S(d)\) | 0.67 | 0.12 | 0.57 | 0.00 | 0.09 |
| **Full pipeline** (ACL → dedupe → \(S(d)\) → diversity → budget) | 0.67 | 0.12 | 0.57 | 0.00 | 0.10 |

**How to read this:** the naive baseline keeps retrieval order and **does not enforce tenant ACL**, so a billing-scoped query can still pack a platform-only secret chunk if retrieval returns it—matching the [assembling trade-offs](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-assembling-trade-offs.md) anti-pattern row *“include every retrieved chunk.”* The pipeline removes unauthorized candidates, deduplicates near-duplicate bodies, caps one chunk per document, and selects by the weighted score from [context scoring math](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-scoring-math.md). On this tiny set, that safety costs some **evidence coverage** versus naive packing because duplicates and extra platform chunks no longer consume budget.

These metrics judge **assembled prompts**, not generated answers. Pair this lab with retrieval/reranking benchmarks and downstream groundedness checks when you change models.

### Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m contexting_strategies.cli evaluate
```

Save a JSON report:

```bash
python -m contexting_strategies.cli evaluate \
  --candidate-k 10 --budget-tokens 120 \
  --output reports/run.json
```

## Architecture

```
User query + ACL tenants
        |
        v
   BM25 retrieval (demo corpus)
        |
        v
 +------+------+------+
 |      |      |      |
 v      v      v      ...
naive  weighted  pipeline assemblers
        |
        v
 Rendered evidence blocks + citation IDs
        |
        v
 Assembly metrics (coverage, redundancy, leaks, ...)
```

### Assembly strategies

| Strategy | Implements | Use it when |
| --- | --- | --- |
| `naive_retrieval_order` | Fill budget in retriever order; **no ACL** | You need a deliberately unsafe baseline to show tenant leaks and duplicate waste. |
| `weighted_score` | Sort by \(S(d) = w_r R + w_f F + w_a A + w_t T\) then pack | You want better selection than raw similarity without full dedupe/diversity rules. |
| `pipeline_acl_dedupe_score_diverse` | Steps 1–7 of the [context algorithm](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-algorithm.md) | You need traceable citations, tenant safety, dedupe, and one chunk per document under budget. |

Default weights (`0.45 / 0.20 / 0.20 / 0.15`) are **illustrative**. Calibrate on your own golden set; the docs explicitly warn against copying defaults from another product.

### Metrics

| Metric | Meaning |
| --- | --- |
| Evidence coverage | Share of required relevant document IDs (label ≥ 2) present in the assembled context |
| Redundancy rate | Average token Jaccard overlap between included chunks |
| Context utilization | Estimated tokens used ÷ budget |
| Relevant evidence per token | Count of label ≥ 1 chunks ÷ tokens |
| Citation traceability | Fraction of included chunks whose citation ID appears in the rendered prompt |
| Unauthorized leak rate | Fraction of queries where a chunk from a disallowed tenant appears (should be 0 after ACL) |
| Assembly latency | Time to filter, score, dedupe, and render (local CPU, sub-millisecond here) |

Token estimates use the same regex tokenizer as the demo retriever—consistent for budgeting comparisons, not a vendor-exact token count.

## Data

- `data/documents.json` — small multi-team corpus with `tenant`, `authority_tier`, `updated_at`, and citation IDs; includes a near-duplicate draft chunk and a platform-only secret doc.
- `data/golden.json` — seven queries with graded relevance labels; `q-007` restricts `allowed_tenants` to `billing` for ACL testing.

## Configuration flags

```bash
python -m contexting_strategies.cli evaluate \
  --candidate-k 10 \
  --budget-tokens 120 \
  --weight-relevance 0.45 \
  --weight-freshness 0.20 \
  --weight-authority 0.20 \
  --weight-task 0.15
```

## Tests

```bash
python -m unittest discover -s tests -v
```

CI runs the same suite on Ubuntu with Python 3.11 (see `.github/workflows/tests.yml`).

## Limitations

- Retrieval is BM25-only; no conversation history, summarization, or structured SQL blocks yet ([context info for output](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-info-for-output.md)).
- No LLM calls; groundedness and abstention behavior are out of scope for this repo.
- Seven synthetic queries cannot calibrate real weights or predict production coverage.

## Related documentation

- [Contexting overview](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/tree/main/contexting)
- [Simple document question shape](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-simple-document-question.md)
- Companion code lab: [reranking-strategies](https://github.com/CaioHAndradeLima/reranking-strategies) (candidate ordering before assembly)

Documentation source repo: [AI-System-Architecturing-Engineering](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/tree/main/contexting).
