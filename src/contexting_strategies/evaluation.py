from collections import defaultdict
from collections.abc import Callable, Iterable
from statistics import mean
from time import perf_counter

from .assemblers import ContextAssembler
from .metrics import score_context
from .models import Hit, QueryCase, ScoreWeights
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


def recommend_best_strategy(summary: dict[str, dict], metric: str = "evidence_coverage") -> dict:
    allowed = {
        "evidence_coverage",
        "relevant_evidence_per_token",
        "redundancy_rate",
        "context_utilization",
    }
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")
    if not summary:
        return {"metric": metric, "strategy": None}

    def sort_key(item):
        name, values = item
        latency = values.get("latency_ms_p95", values.get("latency_ms", float("inf")))
        if metric == "redundancy_rate":
            return (values[metric], -values["evidence_coverage"], latency, name)
        return (-values[metric], values["redundancy_rate"], latency, name)

    name, values = min(summary.items(), key=sort_key)
    return {"metric": metric, "strategy": name, **values}


def evaluate(
    queries: Iterable[QueryCase],
    retriever: Callable[[str, int], list[Hit]],
    assemblers: Iterable[ContextAssembler],
    candidate_k: int = 10,
    budget_tokens: int = 120,
    weights: ScoreWeights | None = None,
) -> dict:
    if type(candidate_k) is not int or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer")
    if type(budget_tokens) is not int or budget_tokens <= 0:
        raise ValueError("budget_tokens must be a positive integer")
    weights = weights or ScoreWeights()
    weights.validate()
    assembler_list = list(assemblers)
    metric_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    rows = []
    for case in queries:
        started = perf_counter()
        candidates = retriever(case.query, candidate_k)
        retrieval_ms = (perf_counter() - started) * 1000
        row = {
            "query_id": case.id,
            "query": case.query,
            "answerable": case.answerable,
            "candidate_ids": [hit.document.id for hit in candidates],
            "retrieval_ms": retrieval_ms,
            "strategies": {},
        }
        for assembler in assembler_list:
            started = perf_counter()
            context = apply_sufficiency_gate(assembler.assemble(case, candidates, budget_tokens, weights), case)
            latency_ms = (perf_counter() - started) * 1000
            metrics = score_context(context, case)
            from .scenario_evaluation import abstain_correct

            metrics["abstain_correct"] = abstain_correct(context, case)
            latencies[assembler.name].append(latency_ms)
            metric_rows[assembler.name].append(metrics)
            row["strategies"][assembler.name] = {
                **metrics,
                "latency_ms": latency_ms,
                "chunk_ids": [chunk.document.id for chunk in context.chunks],
                "dropped_unauthorized": context.dropped_unauthorized,
                "dropped_duplicate": context.dropped_duplicate,
            }
        rows.append(row)
    summary = {}
    for name, values in metric_rows.items():
        summary[name] = {
            **{metric: mean(row[metric] for row in values) for metric in values[0].keys()},
            "latency_ms": mean(latencies[name]),
            "latency_ms_p50": _percentile(latencies[name], 0.50),
            "latency_ms_p95": _percentile(latencies[name], 0.95),
        }
    return {
        "candidate_k": candidate_k,
        "budget_tokens": budget_tokens,
        "weights": {
            "relevance": weights.relevance,
            "freshness": weights.freshness,
            "authority": weights.authority,
            "task": weights.task,
        },
        "queries": rows,
        "summary": summary,
        "recommendation": recommend_best_strategy(summary),
    }
