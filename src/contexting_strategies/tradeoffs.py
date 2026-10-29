"""Map assembler metrics to context-assembling-trade-offs.md anti-patterns."""

from statistics import mean


def evaluate_tradeoffs(assembly_report: dict) -> dict:
    summary = assembly_report.get("summary", {})
    rows = []
    for strategy, metrics in summary.items():
        flags = []
        if metrics.get("unauthorized_leak", 0) > 0:
            flags.append("trust_source_as_instructions_or_skip_acl")
        if metrics.get("redundancy_rate", 0) > 0.35:
            flags.append("include_every_retrieved_chunk_noise")
        if metrics.get("evidence_coverage", 0) < 0.5 and metrics.get("context_utilization", 0) > 0.85:
            flags.append("truncate_or_fill_without_sufficiency")
        if strategy == "naive_retrieval_order":
            flags.append("similarity_only_baseline")
        rows.append(
            {
                "strategy": strategy,
                "anti_patterns_observed": flags,
                "metrics": metrics,
            }
        )
    best_coverage = max(summary.items(), key=lambda item: item[1].get("evidence_coverage", 0))[0] if summary else None
    return {
        "guide": "context-assembling-trade-offs.md",
        "strategies": rows,
        "recommended_for_coverage": best_coverage,
    }
