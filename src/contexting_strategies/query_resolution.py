from collections.abc import Callable
from statistics import mean

from .assemblers import PipelineAssembler
from .metrics import evidence_coverage
from .models import DEFAULT_TENANTS, Hit, QueryCase, ScoreWeights, ThreadScenario


def evaluate_query_resolution(
    scenarios: list[ThreadScenario],
    retriever: Callable[[str, int], list[Hit]],
    *,
    candidate_k: int = 8,
    budget_tokens: int = 160,
    weights: ScoreWeights | None = None,
) -> dict:
    """Compare ambiguous follow-up text vs resolved retrieval query (scoring-math section)."""
    weights = weights or ScoreWeights()
    assembler = PipelineAssembler()
    rows = []
    ambiguous_coverages = []
    resolved_coverages = []
    for scenario in scenarios:
        ambiguous = scenario.query
        tenants = DEFAULT_TENANTS
        pseudo = QueryCase(
            id=scenario.id,
            query=ambiguous,
            relevance=scenario.relevance,
            tags=("query-resolution",),
            answerable=True,
            allowed_tenants=tenants,
        )
        resolved_case = QueryCase(
            id=scenario.id,
            query=scenario.resolved_query,
            relevance=scenario.relevance,
            tags=("query-resolution",),
            answerable=True,
            allowed_tenants=tenants,
        )
        amb_ctx = assembler.assemble(pseudo, retriever(ambiguous, candidate_k), budget_tokens, weights)
        res_ctx = assembler.assemble(
            resolved_case,
            retriever(scenario.resolved_query, candidate_k),
            budget_tokens,
            weights,
        )
        amb_cov = evidence_coverage(amb_ctx, pseudo)
        res_cov = evidence_coverage(res_ctx, resolved_case)
        ambiguous_coverages.append(amb_cov)
        resolved_coverages.append(res_cov)
        rows.append(
            {
                "scenario_id": scenario.id,
                "ambiguous_query": ambiguous,
                "resolved_query": scenario.resolved_query,
                "coverage_ambiguous": amb_cov,
                "coverage_resolved": res_cov,
                "lift": res_cov - amb_cov,
            }
        )
    return {
        "cases": rows,
        "summary": {
            "mean_coverage_ambiguous": mean(ambiguous_coverages) if ambiguous_coverages else 0.0,
            "mean_coverage_resolved": mean(resolved_coverages) if resolved_coverages else 0.0,
        },
    }
