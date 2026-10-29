from collections import defaultdict
from collections.abc import Callable
from statistics import mean
from time import perf_counter

from .assemblers import PipelineAssembler
from .history_assembly import assemble_thread_scenario
from .metrics import score_context
from .models import Hit, MultiHopCase, ScoreWeights, ThreadScenario
from .multihop import assemble_multihop, assemble_single_query_baseline
from .sufficiency import apply_sufficiency_gate


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def abstain_correct(context, case) -> float:
    from .metrics import evidence_coverage

    if not case.answerable:
        return 1.0 if context.should_abstain else 0.0
    coverage = evidence_coverage(context, case)
    if coverage >= 0.5:
        return 1.0 if not context.should_abstain else 0.0
    return 1.0 if context.should_abstain else 0.0


def evaluate_multihop(
    cases: list[MultiHopCase],
    retriever: Callable[[str, int], list[Hit]],
    budget_tokens: int = 180,
    weights: ScoreWeights | None = None,
    candidate_k: int = 6,
) -> dict:
    weights = weights or ScoreWeights()
    weights.validate()
    rows = []
    summaries: dict[str, list[float]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        row = {"query_id": case.id, "query": case.query, "strategies": {}}
        for label, assembler in (
            ("single_query_baseline", assemble_single_query_baseline),
            ("multihop_fair_budget_pipeline", assemble_multihop),
        ):
            started = perf_counter()
            context = assembler(case, retriever, budget_tokens, weights, candidate_k=candidate_k)
            latency_ms = (perf_counter() - started) * 1000
            hop_cov = context.hop_coverage if context.hop_coverage is not None else 0.0
            row["strategies"][label] = {
                "hop_coverage": hop_cov,
                "chunk_ids": [chunk.document.id for chunk in context.chunks],
                "latency_ms": latency_ms,
            }
            summaries[label].append(hop_cov)
            latencies[label].append(latency_ms)
        rows.append(row)
    summary = {
        name: {
            "hop_coverage": mean(values),
            "latency_ms": mean(latencies[name]),
            "latency_ms_p95": _percentile(latencies[name], 0.95),
        }
        for name, values in summaries.items()
    }
    return {"budget_tokens": budget_tokens, "cases": rows, "summary": summary}


def evaluate_threads(
    scenarios: list[ThreadScenario],
    retriever: Callable[[str, int], list[Hit]],
    weights: ScoreWeights | None = None,
    candidate_k: int = 8,
) -> dict:
    weights = weights or ScoreWeights()
    weights.validate()
    rows = []
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    for scenario in scenarios:
        started = perf_counter()
        hits = retriever(scenario.resolved_query, candidate_k)
        retrieval_ms = (perf_counter() - started) * 1000
        context = assemble_thread_scenario(scenario, hits, weights)
        latency_ms = (perf_counter() - started) * 1000
        pseudo_case = _thread_as_case(scenario)
        metrics = score_context(context, pseudo_case)
        metrics["abstain_correct"] = abstain_correct(context, pseudo_case)
        metrics["history_token_count"] = float(context.history_token_count)
        name = context.strategy
        by_strategy[name].append(metrics)
        latencies[name].append(latency_ms)
        rows.append(
            {
                "scenario_id": scenario.id,
                "history_strategy": scenario.history_strategy,
                "retrieval_ms": retrieval_ms,
                "resolved_query": scenario.resolved_query,
                **metrics,
                "chunk_ids": [chunk.document.id for chunk in context.chunks],
                "should_abstain": context.should_abstain,
            }
        )
    summary = {}
    for name, values in by_strategy.items():
        summary[name] = {
            **{metric: mean(row[metric] for row in values) for metric in values[0].keys()},
            "latency_ms_p95": _percentile(latencies[name], 0.95),
        }
    return {"cases": rows, "summary": summary}


def evaluate_abstention_on_queries(
    queries,
    retriever: Callable[[str, int], list[Hit]],
    candidate_k: int,
    budget_tokens: int,
    weights: ScoreWeights | None = None,
    *,
    min_coverage: float = 0.5,
) -> dict:
    weights = weights or ScoreWeights()
    pipeline = PipelineAssembler()
    rows = []
    scores = []
    for case in queries:
        hits = retriever(case.query, candidate_k)
        context = pipeline.assemble(case, hits, budget_tokens, weights)
        gated = apply_sufficiency_gate(context, case, min_coverage=min_coverage)
        correct = abstain_correct(gated, case)
        scores.append(correct)
        rows.append(
            {
                "query_id": case.id,
                "answerable": case.answerable,
                "should_abstain": gated.should_abstain,
                "abstain_correct": correct,
                "evidence_coverage": gated.evidence_coverage,
                "abstain_reason": gated.abstain_reason,
            }
        )
    return {
        "summary": {"abstain_correct_rate": mean(scores) if scores else 0.0},
        "queries": rows,
    }


def _thread_as_case(scenario: ThreadScenario):
    from .models import QueryCase

    return QueryCase(
        id=scenario.id,
        query=scenario.query,
        relevance=scenario.relevance,
        tags=scenario.tags,
        answerable=scenario.answerable,
    )
