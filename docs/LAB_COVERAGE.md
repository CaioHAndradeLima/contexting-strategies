# Contexting docs → lab coverage

Maps each guide under [AI-System-Architecturing-Engineering/contexting](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/tree/main/contexting) to code and CLI output.

| Guide | Lab module / command | `evaluate-full` key |
| --- | --- | --- |
| [context-algorithm.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-algorithm.md) | `assemblers.py` (`pipeline_acl_dedupe_score_diverse`) | `assembly` |
| [context-scoring-math.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-scoring-math.md) | `scoring.py`, `scoring_breakdown.py` | `scoring_math` |
| [context-assembling-trade-offs.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-assembling-trade-offs.md) | `tradeoffs.py` | `tradeoffs` |
| [context-simple-document-question.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-simple-document-question.md) | `SimpleDocumentAssembler`, `simple_document.py` | `simple_document` |
| [context-multi-hop-question.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-multi-hop-question.md) | `multihop.py` | `scenarios.multihop` |
| [context-conversation-strategies.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-conversation-strategies.md) | `history_assembly.py` | `scenarios.conversation` |
| [context-analytical-question.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-analytical-question.md) | `analytical.py` | `analytical` |
| [context-info-for-output.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-info-for-output.md) | `full_prompt.py` | (used in analytical / workflow paths) |
| [context-reliable-architecture.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-reliable-architecture.md) | `reliability.py`, ACL in assemblers | `workflow.injection_detected`, leak metrics in `assembly` |
| [context-evaluate-assembling.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-evaluate-assembling.md) | `metrics.py`, `measure` + judges | `assembly`, `groundedness`, `measure` |
| [context-assembly-flow.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-assembly-flow.md) | `assembly_flow.py` | `assembly_flow` |
| [context-assembly-langchain-langgraph.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-assembly-langchain-langgraph.md) | `workflow.py` | `workflow` |
| [context-architecturing-decision-based.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-architecturing-decision-based.md) | `maturity.py` | `maturity` |
| Query resolution (scoring-math section) | `query_resolution.py`, `threads.json` | `query_resolution` |
| Rerank → assemble | `rerank_bridge.py` | `rerank_bridge` |
| Algorithm steps 1–8 | `context_pipeline.py`, `algorithm_eval.py` | `algorithm_steps` |
| Compacting (truncate / summarize) | `compacting.py`, `compacting_eval.py` | `compacting` |
| Query rewriting | `query_rewrite.py`, `query_rewrite_eval.py` | `query_rewrite` |
| LangGraph node roles | `langgraph_workflow.py` | `langgraph` |
| Five prompt inputs (explicit) | `context_info_eval.py` | `context_info_for_output` |
| Validated analytical SQL | `analytical_sql.py`, `data/analytical_queries.json` | `analytical.validated_sql` |
| Multi-agent task context | `task_context.py` | `task_context` |

**Coverage:** 14/14 guides — `evaluate-full` → `coverage.lab_coverage_percent` is **100** (lab syllabus; production deployment wiring is listed under `maturity.production_deployment_only`).

Run everything:

```bash
python -m contexting_strategies.cli evaluate-full --output reports/full.json
```
