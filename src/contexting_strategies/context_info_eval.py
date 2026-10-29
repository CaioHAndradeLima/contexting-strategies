from collections.abc import Callable

from .assemblers import PipelineAssembler
from .full_prompt import render_full_prompt, token_count
from .conversation import HistoryStrategy, select_history
from .models import ApplicationContext, Hit, QueryCase, ScoreWeights, StructuredResult, ThreadScenario


def evaluate_five_input_render(
    case: QueryCase,
    scenario: ThreadScenario,
    structured: StructuredResult,
    retriever: Callable[[str, int], list[Hit]],
    *,
    candidate_k: int = 8,
    weights: ScoreWeights | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    hits = retriever(scenario.resolved_query, candidate_k)
    strategy = HistoryStrategy(scenario.history_strategy)
    history_text, _ = select_history(
        scenario.turns,
        strategy,
        scenario.history_budget_tokens,
        last_n=scenario.history_n,
    )
    evidence = PipelineAssembler().assemble(case, hits, 120, weights)
    rendered = render_full_prompt(
        system_policy="Cite evidence IDs; refuse when context is insufficient.",
        evidence=evidence,
        structured=structured,
        history_text=history_text,
        application=ApplicationContext(question=case.query, locale="en-US", output_format="markdown"),
    )
    return {
        "guide": "context-info-for-output.md",
        "sections_present": {
            "system_policy": "System policy:" in rendered,
            "retrieved_evidence": "Retrieved evidence:" in rendered,
            "structured_results": structured.label in rendered,
            "conversation_history": "Conversation history:" in rendered,
            "current_request": "Current request:" in rendered,
        },
        "token_count": token_count(rendered),
        "render_order_valid": rendered.index("System policy:")
        < rendered.index("Retrieved evidence:")
        < rendered.index("Structured results")
        < rendered.index("Conversation history:")
        < rendered.index("Current request:"),
    }
