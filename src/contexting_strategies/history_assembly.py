from .assemblers import PipelineAssembler
from .conversation import HistoryStrategy, select_history
from .models import DEFAULT_TENANTS, AssembledContext, Hit, QueryCase, ScoreWeights, ThreadScenario
from .sufficiency import apply_sufficiency_gate
from .tokens import estimate_tokens


def assemble_thread_scenario(
    scenario: ThreadScenario,
    hits: list[Hit],
    weights: ScoreWeights,
) -> AssembledContext:
    strategy = HistoryStrategy(scenario.history_strategy)
    history_text, history_tokens = select_history(
        scenario.turns,
        strategy,
        scenario.history_budget_tokens,
        last_n=scenario.history_n,
    )
    evidence_case = QueryCase(
        id=scenario.id,
        query=scenario.query,
        relevance=scenario.relevance,
        tags=scenario.tags,
        answerable=scenario.answerable,
        allowed_tenants=DEFAULT_TENANTS,
    )
    pipeline = PipelineAssembler()
    evidence = pipeline.assemble(
        evidence_case,
        hits,
        scenario.evidence_budget_tokens,
        weights,
    )
    rendered = (
        f"Question: {scenario.query}\n"
        f"Resolved retrieval query: {scenario.resolved_query}\n"
        f"Conversation ({strategy.value}):\n{history_text or '(none)'}\n\n"
        f"Evidence:\n"
        + "\n\n".join(
            f"[{chunk.citation_id}] {chunk.document.title}\n{chunk.document.text}"
            for chunk in evidence.chunks
        )
    )
    context = AssembledContext(
        strategy=f"pipeline_with_{strategy.value}",
        chunks=evidence.chunks,
        rendered=rendered,
        token_count=estimate_tokens(rendered),
        dropped_unauthorized=evidence.dropped_unauthorized,
        dropped_duplicate=evidence.dropped_duplicate,
        budget_tokens=scenario.history_budget_tokens + scenario.evidence_budget_tokens,
        history_token_count=history_tokens,
    )
    return apply_sufficiency_gate(context, evidence_case)
