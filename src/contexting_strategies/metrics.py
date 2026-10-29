from .models import AssembledContext, QueryCase
from .retrieval import tokenize


def required_evidence_ids(case: QueryCase, min_label: int = 2) -> set[str]:
    return {doc_id for doc_id, label in case.relevance.items() if label >= min_label}


def evidence_coverage(context: AssembledContext, case: QueryCase) -> float:
    required = required_evidence_ids(case)
    if not required:
        return 1.0 if not case.answerable else 0.0
    included = {chunk.document.id for chunk in context.chunks}
    return len(required & included) / len(required)


def redundancy_rate(context: AssembledContext) -> float:
    if len(context.chunks) <= 1:
        return 0.0
    term_sets = [set(tokenize(chunk.document.searchable_text)) for chunk in context.chunks]
    overlaps = []
    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            union = term_sets[i] | term_sets[j]
            if not union:
                continue
            overlaps.append(len(term_sets[i] & term_sets[j]) / len(union))
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def context_utilization(context: AssembledContext) -> float:
    if context.budget_tokens <= 0:
        return 0.0
    return min(1.0, context.token_count / context.budget_tokens)


def relevant_evidence_per_token(context: AssembledContext, case: QueryCase) -> float:
    if context.token_count <= 0:
        return 0.0
    relevant = {doc_id for doc_id, label in case.relevance.items() if label >= 1}
    count = sum(1 for chunk in context.chunks if chunk.document.id in relevant)
    return count / context.token_count


def citation_traceability(context: AssembledContext) -> float:
    if not context.chunks:
        return 1.0
    ok = 0
    for chunk in context.chunks:
        if chunk.citation_id and chunk.citation_id in context.rendered:
            ok += 1
    return ok / len(context.chunks)


def unauthorized_leak(context: AssembledContext, case: QueryCase) -> bool:
    allowed = set(case.allowed_tenants)
    for chunk in context.chunks:
        tenant = chunk.document.metadata.get("tenant")
        if tenant is not None and tenant not in allowed:
            return True
    return False


def score_context(context: AssembledContext, case: QueryCase) -> dict[str, float]:
    return {
        "evidence_coverage": evidence_coverage(context, case),
        "redundancy_rate": redundancy_rate(context),
        "context_utilization": context_utilization(context),
        "relevant_evidence_per_token": relevant_evidence_per_token(context, case),
        "citation_traceability": citation_traceability(context),
        "unauthorized_leak": 1.0 if unauthorized_leak(context, case) else 0.0,
        "token_count": float(context.token_count),
    }
