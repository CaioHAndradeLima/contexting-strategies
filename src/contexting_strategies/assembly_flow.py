from collections.abc import Callable

from .models import Hit, QueryCase, ScoreWeights
from .workflow import run_context_workflow


def summarize_rag_pipeline(
    case: QueryCase,
    retriever: Callable[[str, int], list[Hit]],
    *,
    candidate_k: int = 10,
    budget_tokens: int = 180,
    weights: ScoreWeights | None = None,
) -> dict:
    """Stage list aligned with context-assembly-flow.md / context-algorithm.md."""
    trace = run_context_workflow(
        case,
        retriever,
        candidate_k=candidate_k,
        budget_tokens=budget_tokens,
        weights=weights,
    )
    return {
        "query_id": case.id,
        "stages": [
            {"name": "retrieval", "ms": trace.stages_ms.get("retrieve"), "top_ids": trace.retrieval_ids},
            {"name": "optional_rerank", "ms": trace.stages_ms.get("rerank"), "top_ids": trace.rerank_ids},
            {"name": "context_assembly", "ms": trace.stages_ms.get("assemble"), "chunk_ids": trace.assembled_ids},
            {"name": "sufficiency_gate", "ms": trace.stages_ms.get("sufficiency"), "should_abstain": trace.should_abstain},
            {"name": "safety_scan", "injection_detected": trace.injection_detected},
        ],
        "doc_guides": [
            "context-assembly-flow.md",
            "context-algorithm.md",
            "context-assembly-langchain-langgraph.md",
        ],
    }
