"""Minimal doc Q&A prompt shape (context-simple-document-question.md)."""

from collections.abc import Callable

from .assemblers import SimpleDocumentAssembler
from .metrics import score_context
from .models import Hit, QueryCase, ScoreWeights


def evaluate_simple_document(
    queries: list[QueryCase],
    retriever: Callable[[str, int], list[Hit]],
    *,
    candidate_k: int = 10,
    budget_tokens: int = 120,
    weights: ScoreWeights | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    assembler = SimpleDocumentAssembler()
    rows = []
    for case in queries:
        hits = retriever(case.query, candidate_k)
        context = assembler.assemble(case, hits, budget_tokens, weights)
        rows.append(
            {
                "query_id": case.id,
                "strategy": assembler.name,
                "prompt_shape": "question_plus_citations_only",
                "metrics": score_context(context, case),
                "chunk_ids": [c.document.id for c in context.chunks],
            }
        )
    return {"cases": rows}
