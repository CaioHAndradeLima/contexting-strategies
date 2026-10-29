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

Run everything:

```bash
python -m contexting_strategies.cli evaluate-full --output reports/full.json
```
