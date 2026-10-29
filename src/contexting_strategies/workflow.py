from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

from .assemblers import PipelineAssembler
from .heuristic_rerank import HeuristicReranker
from .models import Hit, QueryCase, ScoreWeights
from .reliability import any_injection_in_chunks
from .sufficiency import apply_sufficiency_gate


@dataclass
class WorkflowTrace:
    query_id: str
    stages_ms: dict[str, float] = field(default_factory=dict)
    retrieval_ids: list[str] = field(default_factory=list)
    rerank_ids: list[str] = field(default_factory=list)
    assembled_ids: list[str] = field(default_factory=list)
    should_abstain: bool = False
    injection_detected: bool = False


def run_context_workflow(
    case: QueryCase,
    retriever: Callable[[str, int], list[Hit]],
    *,
    candidate_k: int = 10,
    metric_k: int = 5,
    budget_tokens: int = 180,
    weights: ScoreWeights | None = None,
) -> WorkflowTrace:
    """LangGraph-style linear pipeline: retrieve → rerank → assemble → sufficiency (+ safety scan)."""
    weights = weights or ScoreWeights()
    trace = WorkflowTrace(query_id=case.id)
    started = perf_counter()
    candidates = retriever(case.query, candidate_k)
    trace.stages_ms["retrieve"] = (perf_counter() - started) * 1000
    trace.retrieval_ids = [hit.document.id for hit in candidates[:metric_k]]

    started = perf_counter()
    reranked = HeuristicReranker().rerank(case.query, candidates, metric_k)
    trace.stages_ms["rerank"] = (perf_counter() - started) * 1000
    trace.rerank_ids = [hit.document.id for hit in reranked]

    started = perf_counter()
    assembled = PipelineAssembler().assemble(case, reranked, budget_tokens, weights)
    trace.stages_ms["assemble"] = (perf_counter() - started) * 1000
    trace.assembled_ids = [chunk.document.id for chunk in assembled.chunks]
    trace.injection_detected = any_injection_in_chunks(assembled.chunks)

    started = perf_counter()
    gated = apply_sufficiency_gate(assembled, case, min_coverage=0.34)
    trace.stages_ms["sufficiency"] = (perf_counter() - started) * 1000
    trace.should_abstain = gated.should_abstain
    return trace
