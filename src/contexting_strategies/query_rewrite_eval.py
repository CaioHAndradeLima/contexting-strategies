from .data import ThreadScenario
from .query_rewrite import rewrite_follow_up_query, rewrite_quality


def evaluate_query_rewrite(scenarios: list[ThreadScenario]) -> dict:
    rows = []
    overlaps = []
    for scenario in scenarios:
        rewritten = rewrite_follow_up_query(scenario.turns, scenario.query)
        quality = rewrite_quality(rewritten, scenario.resolved_query)
        overlaps.append(quality["token_overlap_with_reference"])
        rows.append({"scenario_id": scenario.id, **quality})
    return {
        "guide": "context-scoring-math.md#contextual-query-resolution-before-retrieval",
        "cases": rows,
        "mean_overlap_with_fixture_reference": sum(overlaps) / len(overlaps) if overlaps else 0.0,
    }
