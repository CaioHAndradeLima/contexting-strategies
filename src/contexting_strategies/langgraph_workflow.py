from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

from .assemblers import PipelineAssembler
from .full_prompt import render_full_prompt
from .groundedness import lexical_groundedness
from .heuristic_rerank import HeuristicReranker
from .models import ApplicationContext, Hit, QueryCase, ScoreWeights
from .reliability import any_injection_in_chunks
from .sufficiency import apply_sufficiency_gate

LANGGRAPH_SPEC = {
    "nodes": [
        {"id": "retrieve_node", "role": "retrieve"},
        {"id": "assemble_context_node", "role": "assemble"},
        {"id": "generate_answer_node", "role": "generate"},
        {"id": "validate_answer_node", "role": "validate"},
    ],
    "edges": [
        ["retrieve_node", "assemble_context_node"],
        ["assemble_context_node", "generate_answer_node"],
        ["generate_answer_node", "validate_answer_node"],
    ],
}


@dataclass
class GraphState:
    query_id: str
    query: str
    tenant: str = "billing"
    rewritten_query: str | None = None
    candidates: list[Hit] = field(default_factory=list)
    assembled_rendered: str = ""
    assembled_ids: list[str] = field(default_factory=list)
    draft_answer: str = ""
    should_abstain: bool = False
    injection_detected: bool = False
    validation_passed: bool = False
    validation_score: float = 0.0
    node_timings_ms: dict[str, float] = field(default_factory=dict)


def retrieve_node(state: GraphState, retriever: Callable[[str, int], list[Hit]], *, k: int = 10) -> GraphState:
    started = perf_counter()
    q = state.rewritten_query or state.query
    state.candidates = retriever(q, k)
    state.node_timings_ms["retrieve_node"] = (perf_counter() - started) * 1000
    return state


def assemble_context_node(state: GraphState, case: QueryCase, *, budget: int, weights: ScoreWeights) -> GraphState:
    started = perf_counter()
    query = state.rewritten_query or state.query
    reranked = HeuristicReranker().rerank(query, state.candidates, 8)
    context = apply_sufficiency_gate(
        PipelineAssembler().assemble(case, reranked, budget, weights),
        case,
        min_coverage=0.34,
    )
    state.assembled_rendered = context.rendered
    state.assembled_ids = [c.document.id for c in context.chunks]
    state.should_abstain = context.should_abstain
    state.injection_detected = any_injection_in_chunks(context.chunks)
    state.node_timings_ms["assemble_context_node"] = (perf_counter() - started) * 1000
    return state


def generate_answer_node(state: GraphState, *, reference_answer: str | None) -> GraphState:
    started = perf_counter()
    if state.should_abstain or state.injection_detected:
        state.draft_answer = "I do not have sufficient evidence in the provided context to answer reliably."
    elif reference_answer:
        state.draft_answer = reference_answer
    else:
        state.draft_answer = "Evidence assembled; generation stub for offline lab."
    state.node_timings_ms["generate_answer_node"] = (perf_counter() - started) * 1000
    return state


def validate_answer_node(state: GraphState, case: QueryCase, documents_by_id: dict) -> GraphState:
    started = perf_counter()
    if state.should_abstain:
        state.validation_passed = not case.answerable
        state.validation_score = 1.0 if state.validation_passed else 0.0
    else:
        evidence = [documents_by_id[i] for i in state.assembled_ids if i in documents_by_id]
        state.validation_score = lexical_groundedness(state.draft_answer, evidence)
        state.validation_passed = state.validation_score >= 0.5
    state.node_timings_ms["validate_answer_node"] = (perf_counter() - started) * 1000
    return state


def run_langgraph_pipeline(
    case: QueryCase,
    retriever: Callable[[str, int], list[Hit]],
    documents_by_id: dict,
    *,
    budget_tokens: int = 180,
    weights: ScoreWeights | None = None,
    rewritten_query: str | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    state = GraphState(query_id=case.id, query=case.query, rewritten_query=rewritten_query)
    state = retrieve_node(state, retriever)
    state = assemble_context_node(state, case, budget=budget_tokens, weights=weights)
    state = generate_answer_node(state, reference_answer=case.reference_answer)
    state = validate_answer_node(state, case, documents_by_id)
    return {
        "graph": LANGGRAPH_SPEC,
        "query_id": case.id,
        "should_abstain": state.should_abstain,
        "validation_passed": state.validation_passed,
        "validation_score": state.validation_score,
        "injection_detected": state.injection_detected,
        "node_timings_ms": state.node_timings_ms,
        "assembled_ids": state.assembled_ids,
    }
