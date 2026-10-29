import re
from datetime import datetime, timezone

from .models import Document, Hit, QueryCase, ScoreWeights
from .retrieval import tokenize

ENTITY_RE = re.compile(r"\bINC-\d+\b", re.I)


def _normalize_scores(hits: list[Hit]) -> dict[str, float]:
    if not hits:
        return {}
    values = [hit.score for hit in hits]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {hit.document.id: 1.0 for hit in hits}
    return {hit.document.id: (hit.score - lo) / (hi - lo) for hit in hits}


def freshness_score(document: Document, reference: datetime) -> float:
    raw = document.metadata.get("updated_at")
    if not raw:
        return 0.5
    updated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (reference - updated).total_seconds() / 86400.0)
    return max(0.0, 1.0 - min(age_days / 365.0, 1.0))


def authority_score(document: Document) -> float:
    tier = document.metadata.get("authority_tier", "official")
    return {"official": 1.0, "runbook": 0.85, "draft": 0.4}.get(str(tier), 0.5)


def task_score(document: Document, query: str) -> float:
    query_terms = set(tokenize(query))
    doc_terms = set(tokenize(document.searchable_text))
    overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
    entity_bonus = 0.0
    for entity in ENTITY_RE.findall(query):
        if entity.lower() in document.searchable_text.lower():
            entity_bonus = 0.55
            break
    return min(1.0, overlap + entity_bonus)


def combined_score(
    hit: Hit,
    case: QueryCase,
    relevance_norm: float,
    reference: datetime,
    weights: ScoreWeights,
) -> float:
    weights.validate()
    freshness = freshness_score(hit.document, reference)
    authority = authority_score(hit.document)
    task = task_score(hit.document, case.query)
    return (
        weights.relevance * relevance_norm
        + weights.freshness * freshness
        + weights.authority * authority
        + weights.task * task
    )


def score_hits(
    hits: list[Hit],
    case: QueryCase,
    weights: ScoreWeights,
    reference: datetime | None = None,
) -> list[tuple[Hit, float]]:
    reference = reference or datetime(2026, 10, 29, tzinfo=timezone.utc)
    normalized = _normalize_scores(hits)
    scored = []
    for hit in hits:
        assembly = combined_score(hit, case, normalized[hit.document.id], reference, weights)
        scored.append((hit, assembly))
    scored.sort(key=lambda pair: (-pair[1], pair[0].document.id))
    return scored
