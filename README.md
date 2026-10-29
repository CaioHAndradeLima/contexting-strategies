# Contexting Strategies

An engineering lab for a practical RAG question that sits **after retrieval** and **before the LLM call**: **how should you assemble evidence into a prompt under a token budget** without leaking unauthorized text, repeating the same facts, or starving critical passages?

The project implements the [context algorithm](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-algorithm.md), extended scenarios from the contexting docs, a **rerank → assemble** bridge to [reranking-strategies](https://github.com/CaioHAndradeLima/reranking-strategies), analytical SQL+doc prompts, offline groundedness checks, and a linear **workflow** trace (retrieve → rerank → assemble → sufficiency). It is for learning and controlled experiments—not a hosted chat service.

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
| `evaluate` | Compare naive / weighted / pipeline assemblers |
| `evaluate-scenarios` | Multi-hop, conversation history, abstention |
| `evaluate-full` | All tracks above + rerank bridge + analytical + groundedness + workflow |

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

- **No live LLM** for assembly, summarization, or judging unless you add your own integration; groundedness is a lexical baseline, not GPT-as-judge.
- **Multi-hop sub-queries are fixture-authored**, not model-generated.
- **LangGraph** is represented as an explicit Python workflow, not a LangChain dependency.
- **Analytical** data is toy SQL rows, not a real warehouse connector.
- Synthetic corpora **do not predict** production metrics.

## Related documentation

- [Contexting overview](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/tree/main/contexting)
- [reranking-strategies](https://github.com/CaioHAndradeLima/reranking-strategies)
- [AI-System-Architecturing-Engineering](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering)
