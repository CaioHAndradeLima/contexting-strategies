import re

from .models import ConversationTurn

INC_RE = re.compile(r"\bINC-\d+\b", re.I)


def rewrite_follow_up_query(turns: tuple[ConversationTurn, ...], ambiguous_query: str) -> str:
    """
    Contextual query resolution before retrieval (context-scoring-math.md).
    Lab uses deterministic rules; production may swap in an LLM rewriter.
    """
    normalized = ambiguous_query.strip().lower()
    if normalized not in {"did we fix it?", "did we fix it", "was it fixed?"}:
        return ambiguous_query.strip()

    incident: str | None = None
    for turn in reversed(turns):
        match = INC_RE.search(turn.content)
        if match:
            incident = match.group(0).upper()
            break
    incident = incident or "INC-4821"
    return (
        f"Was incident {incident} resolved and what mitigation was applied? "
        "Include remediation date if documented."
    )


def rewrite_quality(rewritten: str, reference: str) -> dict:
    ref_terms = set(re.findall(r"[a-z0-9]+", reference.lower()))
    out_terms = set(re.findall(r"[a-z0-9]+", rewritten.lower()))
    overlap = len(ref_terms & out_terms) / max(1, len(ref_terms))
    return {"token_overlap_with_reference": round(overlap, 4), "rewritten": rewritten, "reference": reference}
