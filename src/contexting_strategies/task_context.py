"""Task-specific context slices for multi-agent patterns (architecturing decision table)."""

from .assemblers import PipelineAssembler
from .models import Hit, QueryCase, ScoreWeights


def filter_hits_for_agent(hits: list[Hit], agent_team: str) -> list[Hit]:
    return [hit for hit in hits if hit.document.metadata.get("team") == agent_team]


def assemble_for_agent(
    case: QueryCase,
    hits: list[Hit],
    *,
    agent_team: str,
    budget_tokens: int,
    weights: ScoreWeights | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    scoped = filter_hits_for_agent(hits, agent_team)
    context = PipelineAssembler().assemble(case, scoped, budget_tokens, weights)
    return {
        "agent_team": agent_team,
        "candidate_count": len(scoped),
        "chunk_ids": [c.document.id for c in context.chunks],
        "token_count": context.token_count,
    }
