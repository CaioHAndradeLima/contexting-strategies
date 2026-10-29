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

Extended scenarios (multi-hop assembly, conversation history budgeting, insufficient-evidence gate):

```bash
python -m contexting_strategies.cli evaluate-scenarios \
  --output reports/scenarios.json
```

### Scenario results (same environment, 2026-10-29)

Guides: [multi-hop question](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-multi-hop-question.md), [conversation strategies](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-conversation-strategies.md), [evaluate assembling — no-answer handling](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-evaluate-assembling.md).

| Scenario | Baseline | Improved approach | Measured result |
| --- | --- | --- | --- |
| **Multi-hop timeline** (`mh-001`, 180-token budget) | One retrieval on the vague question *“What is the timeline?”* | Per-hop sub-queries with fair budget split + dated sections | Hop coverage **0.17** → **0.83** |
| **Follow-up thread** (“Did we fix it?” + 8-turn history, 200-token total budget) | `full_history` fills **65** history tokens | `last_n` (2 turns) uses **26** history tokens | Evidence coverage **1.0** in both; `last_n` lowers utilization (**0.49** vs **0.69**) leaving headroom for more evidence on tighter budgets |
| **Abstention gate** (pipeline + sufficiency check, 200-token budget) | Answer when coverage is thin | `INSUFFICIENT_EVIDENCE` when coverage &lt; threshold or corpus has no labels | **100%** abstain decision accuracy on the seven-query golden set (including unanswerable `q-006`) |

Multi-hop assembly does **not** call an LLM for decomposition—the golden file lists explicit hop sub-queries, matching the doc pattern “one query per facet.” Production systems often learn or infer those sub-queries; here they are labeled for reproducible measurement.

The sufficiency gate prefix is a **prompt-level abstain signal** for the demo. Wire it to your product’s UX (clarifying question, ticket escalation, etc.) in a real service.

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
| Hop coverage | For multi-hop cases, average per-hop evidence coverage (label ≥ 2) |
| Abstain decision accuracy | Whether abstain/abstain-not matches answerable labels and coverage rules |
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

- Retrieval is BM25-only; no LLM query rewriting, summarization model, or structured SQL blocks ([context info for output](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-info-for-output.md)).
- Multi-hop sub-queries are **authored in fixtures**, not generated at runtime.
- Conversation “summary” is deterministic bullet compression, not an LLM summary—real summaries need faithfulness checks.
- No LLM answer generation; downstream groundedness is out of scope.
- Synthetic queries cannot calibrate real weights or predict production coverage.

## Related documentation

- [Contexting overview](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/tree/main/contexting)
- [Simple document question shape](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-simple-document-question.md)
- Companion code lab: [reranking-strategies](https://github.com/CaioHAndradeLima/reranking-strategies) (candidate ordering before assembly)

Documentation source repo: [AI-System-Architecturing-Engineering](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/tree/main/contexting).
