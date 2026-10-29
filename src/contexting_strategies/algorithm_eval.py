from collections.abc import Callable

from .assemblers import PipelineAssembler
from .models import Hit, QueryCase, ScoreWeights


def evaluate_algorithm_steps(
    queries: list[QueryCase],
    retriever: Callable[[str, int], list[Hit]],
    *,
    candidate_k: int = 10,
    budget_tokens: int = 180,
    weights: ScoreWeights | None = None,
    sample_ids: list[str] | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    ids = set(sample_ids or [])
    cases = [q for q in queries if not ids or q.id in ids]
    assembler = PipelineAssembler()
    rows = []
    for case in cases:
        hits = retriever(case.query, candidate_k)
        _, trace = assembler.assemble_with_trace(case, hits, budget_tokens, weights)
        rows.append({"query_id": case.id, "steps": trace.steps, "merged_parent_child": trace.merged_parent_child})
    return {"algorithm_doc": "context-algorithm.md", "cases": rows, "steps_implemented": list(range(1, 9))}
