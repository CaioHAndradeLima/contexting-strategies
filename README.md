# Contexting Strategies

An engineering lab for a practical RAG question that sits **after retrieval** and **before the LLM call**: **how should you assemble evidence into a prompt under a token budget** without leaking unauthorized text, repeating the same facts, or starving critical passages?

The project implements the [context algorithm](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-algorithm.md), extended scenarios from the contexting docs, a **rerank → assemble** bridge to [reranking-strategies](https://github.com/CaioHAndradeLima/reranking-strategies), and a **production-style measurement pipeline** with optional **LLM judges** and cost accounting. Offline commands remain available for CI; live measurement requires an API key.

## Production measurement pipeline (`measure`)

Use this when you want to compare **assembly strategies** the way a production eval job would: same retrieval (optional heuristic rerank), assemble, offline metrics, then—when enabled—**generate an answer**, **judge context quality**, and **judge answer groundedness** with structured JSON rubrics.

```bash
cp .env.example .env   # set OPENAI_API_KEY
pip install -e '.[llm-judge]'

# Offline only (CI-safe): offline metrics + abstention, no API spend
python -m contexting_strategies.cli measure \
  --strategies pipeline_acl_dedupe_score_diverse,weighted_score \
  --budget-tokens 200 --output reports/measure-offline.json

# Live LLM measurement (answer + dual judges, hard call budget)
python -m contexting_strategies.cli measure \
  --live-llm \
  --strategies pipeline_acl_dedupe_score_diverse,weighted_score,naive_retrieval_order \
  --budget-tokens 200 \
  --answer-model gpt-6-luna \
  --judge-model gpt-6-luna \
  --max-llm-calls 150 \
  --output reports/measure-live.json
```

| Stage | Component | Output |
| --- | --- | --- |
| Retrieve | BM25 (+ optional heuristic rerank) | Candidate set shared across strategies |
| Assemble | Selected assemblers | Prompt + offline metrics (coverage, leaks, redundancy) |
| Sufficiency | Gate | Skip generation when evidence is thin |
| Generate | `OpenAIAnswerGenerator` | Answer from assembled context only |
| Judge context | `OpenAIContextJudge` | sufficiency / citations / safety (0–4) |
| Judge answer | `OpenAIAnswerJudge` | correctness / groundedness / calibration (0–4) |
| Report | `measurement_pipeline.py` | Per-strategy summary, token/cost totals, recommendation |

Environment variables (`CONTEXT_LAB_*`) and flags are documented in [.env.example](.env.example). **Do not** treat demo corpus scores as production SLAs—swap in your corpus, golden labels, and models, then rerun `measure`.

### Profiles and quality gates (CI / Docker)

Strategy sets and pass/fail thresholds live in [config/measure_profiles.json](config/measure_profiles.json)—similar to a production eval config repo:

| Profile | Strategies | Typical use |
| --- | --- | --- |
| `ci-all-strategies` | naive, weighted, pipeline | PR gate: compare all assemblers |
| `ci-pipeline-only` | pipeline | Fast smoke on production assembler |
| `ci-tight-budget` | weighted, pipeline | Stress small token budget |
| `nightly-live` | pipeline, weighted + LLM judges | Manual/nightly with API key |

```bash
# Local production-like run (offline)
./scripts/docker-ci.sh ci-all-strategies

# Or compose
docker compose run --rm measure-offline   # writes reports/measure-offline.json

# Profile-based CLI
context-lab measure --profile ci-tight-budget --output reports/m.json
context-lab check-measure --profile ci-tight-budget --report reports/m.json
```

**GitHub Actions**

- [`measure-gate.yml`](.github/workflows/measure-gate.yml) — on every push/PR: Docker image build, matrix over profiles (`ci-all-strategies`, `ci-tight-budget`, `ci-pipeline-only`), measure + quality gate, artifact upload.
- [`measure-live.yml`](.github/workflows/measure-live.yml) — manual `workflow_dispatch`: live LLM measure with repository secret `OPENAI_API_KEY` (profile `nightly-live`).

Add `OPENAI_API_KEY` under **Settings → Secrets and variables → Actions** before running the live workflow.

## Start here

### Single-hop assembly (120-token budget, local run 2026-10-29)

| Strategy | Evidence coverage | Unauthorized leak rate |
| --- | ---: | ---: |
| Naive retrieval order | 0.81 | 0.14 |
| Weighted \(S(d)\) | 0.67 | 0.00 |
| Full pipeline | 0.67 | 0.00 |

### Full lab (`evaluate-full`)

One command runs assembly, scenarios, rerank bridge, analytical, groundedness, and workflow stages:

```bash
python -m contexting_strategies.cli evaluate-full --output reports/full.json
```

| Track | Doc guide | What the lab measures |
| --- | --- | --- |
| **A — Rerank → assemble** | [Reranking](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/retrieval/search/retrieval-strategy-reranking.md) · [assembly flow](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-assembly-flow.md) | Evidence coverage after BM25 order vs heuristic rerank vs imported [rerank report](data/rerank_report_sample.json) |
| **B — Analytical context** | [Analytical question](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-analytical-question.md) | Structured SQL block + doc evidence; numeric support vs reference answer |
| **C — Groundedness** | [Evaluate assembling](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-evaluate-assembling.md) | Lexical groundedness + correctness proxy on [answer_samples.json](data/answer_samples.json) (no live LLM required) |
| **Multi-hop / history / abstain** | [Multi-hop](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-multi-hop-question.md) · [Conversation strategies](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-conversation-strategies.md) | `evaluate-scenarios` (hop coverage 0.17→0.83 on `mh-001`; history token savings) |
| **Reliability** | [Reliable architecture](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-reliable-architecture.md) | Injection phrase filtering in [reliability.py](src/contexting_strategies/reliability.py); workflow flags |
| **Five prompt inputs** | [Context info for output](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-info-for-output.md) | [full_prompt.py](src/contexting_strategies/full_prompt.py) render order |
| **Workflow nodes** | [LangGraph assembly](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-assembly-langchain-langgraph.md) | [workflow.py](src/contexting_strategies/workflow.py) stage timings (no LangGraph dependency) |

**Sample `evaluate-full` metrics (synthetic, same date):** rerank-bridge mean coverage ≈ **0.90** (retrieval/heuristic) and **0.94** (imported rerank report); analytical structured numeric support **1.0** on `an-001`; groundedness mean lexical score **0.67** with correctness proxy **1.0** on bundled samples.

### Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m contexting_strategies.cli evaluate
python -m contexting_strategies.cli evaluate-scenarios
python -m contexting_strategies.cli evaluate-full
```

### Rerank report import

From **reranking-strategies**:

```bash
python scripts/export_context_candidates.py --output rerank_report.json
```

Copy or point contexting at that file (default fixture: `data/rerank_report_sample.json`).

## Architecture

```
Query + ACL
   → retrieve → rerank (optional / bridge) → assemble → sufficiency gate
   → [analytical structured block] → full prompt render → (optional) answer + groundedness
```

## Commands

| Command | Purpose |
| --- | --- |
| `measure` | **Production pipeline** — strategy comparison + optional LLM generate/judge |
| `check-measure` | Apply profile quality gates to a measure report JSON (exit 1 on failure) |
| `evaluate` | Compare naive / weighted / pipeline assemblers (offline) |
| `evaluate-scenarios` | Multi-hop, conversation history, abstention |
| `evaluate-full` | All offline tracks + rerank bridge + analytical + lexical groundedness + workflow |

## Data files

| File | Role |
| --- | --- |
| `data/documents.json` | Corpus with tenant, authority, citations, injection sample `doc-016` |
| `data/golden.json` | Single-hop labeled queries |
| `data/multihop_golden.json` | Per-hop sub-queries |
| `data/threads.json` | Follow-up history scenarios |
| `data/structured_results.json` | SQL-style blocks for analytical cases |
| `data/analytical_golden.json` | Analytics questions |
| `data/answer_samples.json` | Answers for offline groundedness |
| `data/rerank_report_sample.json` | Rankings exported from reranking-strategies |

## Tests

```bash
python -m unittest discover -s tests -v
```

## Honest limitations

- **`measure --live-llm`** uses OpenAI APIs; costs depend on prompt size and query count. Cap calls with `--max-llm-calls`.
- Offline **`evaluate-full` groundedness** remains lexical; live **`measure`** uses LLM judges when enabled.
- **Multi-hop sub-queries are fixture-authored**, not model-generated.
- **LangGraph** is represented as an explicit Python workflow, not a LangChain dependency.
- **Analytical** data is toy SQL rows, not a real warehouse connector.
- Synthetic corpora **do not predict** production metrics.

## Related documentation

- [Contexting overview](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/tree/main/contexting)
- [reranking-strategies](https://github.com/CaioHAndradeLima/reranking-strategies)
- [AI-System-Architecturing-Engineering](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering)
