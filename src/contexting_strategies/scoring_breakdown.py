from datetime import datetime, timezone

from .models import Hit, QueryCase, ScoreWeights
from .scoring import (
    _normalize_scores,
    authority_score,
    combined_score,
    freshness_score,
    task_score,
)


def breakdown_scoring_signals(
    hits: list[Hit],
    case: QueryCase,
    weights: ScoreWeights,
    *,
    reference: datetime | None = None,
    limit: int = 5,
) -> list[dict]:
    """Expose R, F, A, T and S(d) for context-scoring-math.md."""
    reference = reference or datetime(2026, 10, 29, tzinfo=timezone.utc)
    normalized = _normalize_scores(hits)
    rows = []
    for hit in hits[:limit]:
        doc = hit.document
        r = normalized[doc.id]
        f = freshness_score(doc, reference)
        a = authority_score(doc)
        t = task_score(doc, case.query)
        s = combined_score(hit, case, r, reference, weights)
        rows.append(
            {
                "doc_id": doc.id,
                "R_relevance_norm": round(r, 4),
                "F_freshness": round(f, 4),
                "A_authority": round(a, 4),
                "T_task": round(t, 4),
                "S_combined": round(s, 4),
                "weights": {
                    "relevance": weights.relevance,
                    "freshness": weights.freshness,
                    "authority": weights.authority,
                    "task": weights.task,
                },
            }
        )
    rows.sort(key=lambda row: row["S_combined"], reverse=True)
    return rows


def evaluate_scoring_math(
    queries: list[QueryCase],
    retriever,
    *,
    candidate_k: int = 8,
    weights: ScoreWeights | None = None,
    sample_query_ids: list[str] | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    ids = set(sample_query_ids or [])
    cases = [q for q in queries if not ids or q.id in ids]
    return {
        "formula": "S(d) = w_r R + w_f F + w_a A + w_t T",
        "cases": [
            {
                "query_id": case.id,
                "query": case.query,
                "top_signals": breakdown_scoring_signals(
                    retriever(case.query, candidate_k),
                    case,
                    weights,
                ),
            }
            for case in cases
        ],
    }
