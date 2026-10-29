from .metrics import evidence_coverage, required_evidence_ids
from .models import AssembledContext, QueryCase


def apply_sufficiency_gate(
    context: AssembledContext,
    case: QueryCase,
    *,
    min_coverage: float = 0.5,
) -> AssembledContext:
    coverage = evidence_coverage(context, case)
    if not case.answerable:
        return _with_abstain(
            context,
            should_abstain=True,
            reason="question is not answerable from the indexed corpus",
            coverage=coverage,
        )

    required = required_evidence_ids(case)
    if not required:
        return _with_abstain(context, should_abstain=False, reason=None, coverage=coverage)

    if coverage < min_coverage:
        return _with_abstain(
            context,
            should_abstain=True,
            reason=f"evidence coverage {coverage:.2f} below threshold {min_coverage:.2f}",
            coverage=coverage,
        )
    return _with_abstain(context, should_abstain=False, reason=None, coverage=coverage)


def _with_abstain(
    context: AssembledContext,
    *,
    should_abstain: bool,
    reason: str | None,
    coverage: float,
) -> AssembledContext:
    rendered = context.rendered
    if should_abstain:
        rendered = f"INSUFFICIENT_EVIDENCE: {reason}\n\n{rendered}"
    return AssembledContext(
        strategy=context.strategy,
        chunks=context.chunks,
        rendered=rendered,
        token_count=context.token_count,
        dropped_unauthorized=context.dropped_unauthorized,
        dropped_duplicate=context.dropped_duplicate,
        budget_tokens=context.budget_tokens,
        should_abstain=should_abstain,
        abstain_reason=reason,
        history_token_count=context.history_token_count,
        hop_coverage=context.hop_coverage,
        evidence_coverage=coverage,
    )
