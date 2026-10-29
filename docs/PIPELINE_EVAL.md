# Contexting pipeline evaluation

Scorecard aligned with [context-evaluate-assembling.md](https://github.com/CaioHAndradeLima/AI-System-Architecturing-Engineering/blob/main/contexting/context-evaluate-assembling.md).

## Command

```bash
context-lab evaluate-pipeline --profile ci-contexting-eval --output reports/pipeline-eval.json
context-lab check-measure --profile ci-contexting-eval --report reports/pipeline-eval.json
```

## What runs

| Layer | Source | Metrics |
| --- | --- | --- |
| Assembly | `measure` on golden set (pipeline assembler) | coverage, redundancy, citation traceability, leak rate, assembly p95, abstain correctness |
| Abstention | `evaluate_abstention_on_queries` | `abstain_correct_rate` |
| Multi-hop | `evaluate_multihop` | hop coverage + lift vs single-query baseline |

Report includes `scorecard` (human-readable rollup) and `scenarios` (raw scenario blocks).

## Gates (`ci-contexting-eval`)

**Per strategy (`gates.pipeline_acl_dedupe_score_diverse`):**

- `min_evidence_coverage`, `max_unauthorized_leak`, `max_redundancy_rate`
- `min_citation_traceability`, `max_assembly_ms_p95`, `min_abstain_correct`

**Scenarios (`scenario_gates`):**

- `min_abstain_correct_rate`
- `min_multihop_hop_coverage`
- `min_multihop_lift_over_baseline`

Live LLM judges remain on `measure --profile nightly-live` / `./scripts/measure-live-local.sh`.
