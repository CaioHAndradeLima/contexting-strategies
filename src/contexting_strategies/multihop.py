from collections.abc import Callable

from .assemblers import _pack_chunks, _render_block, _tenant_allowed
from .models import AssembledContext, ContextChunk, Hit, MultiHopCase, ScoreWeights
from .scoring import score_hits
from .tokens import estimate_tokens


def assemble_multihop(
    case: MultiHopCase,
    retriever: Callable[[str, int], list[Hit]],
    budget_tokens: int,
    weights: ScoreWeights,
    *,
    candidate_k: int = 6,
) -> AssembledContext:
    if not case.hops:
        raise ValueError("multi-hop case requires at least one hop")
    hop_budget = max(20, budget_tokens // len(case.hops))
    sections: list[str] = []
    all_chunks: list[ContextChunk] = []
    dropped_unauthorized = 0
    dropped_duplicate = 0
    hop_coverages: list[float] = []

    for hop in case.hops:
        hits = retriever(hop.query, candidate_k)
        filtered: list[Hit] = []
        for hit in hits:
            if not _tenant_allowed(hit.document, case.allowed_tenants):
                dropped_unauthorized += 1
                continue
            filtered.append(hit)
        scored = score_hits(filtered, _hop_as_query_case(case, hop.id, hop.relevance), weights)
        packed = _pack_chunks(
            "multihop_hop",
            scored,
            _hop_as_query_case(case, hop.id, hop.relevance),
            hop_budget,
            max_per_document=1,
            reserve_instruction_tokens=20,
        )
        dropped_duplicate += packed.dropped_duplicate
        required = {doc_id for doc_id, label in hop.relevance.items() if label >= 2}
        included = {chunk.document.id for chunk in packed.chunks}
        hop_coverages.append(len(required & included) / len(required) if required else 1.0)
        event_lines = []
        for chunk in packed.chunks:
            event_at = chunk.document.metadata.get("event_at", "unknown-date")
            event_lines.append(f"  event_at={event_at} {_render_block(chunk)}")
        section = f"## Hop: {hop.label} ({hop.id})\nSub-query: {hop.query}\n" + "\n\n".join(event_lines)
        sections.append(section)
        all_chunks.extend(packed.chunks)

    header = f"Multi-hop question: {case.query}\nOrganized evidence (by hop, chronological hints):\n"
    rendered = header + "\n\n".join(sections)
    hop_coverage = sum(hop_coverages) / len(hop_coverages)
    return AssembledContext(
        strategy="multihop_fair_budget_pipeline",
        chunks=tuple(all_chunks),
        rendered=rendered,
        token_count=estimate_tokens(rendered),
        dropped_unauthorized=dropped_unauthorized,
        dropped_duplicate=dropped_duplicate,
        budget_tokens=budget_tokens,
        hop_coverage=hop_coverage,
    )


def assemble_single_query_baseline(
    case: MultiHopCase,
    retriever: Callable[[str, int], list[Hit]],
    budget_tokens: int,
    weights: ScoreWeights,
    *,
    candidate_k: int = 10,
) -> AssembledContext:
    """Pack using only the top-level query (typical failure mode for multi-hop)."""
    hits = retriever(case.query, candidate_k)
    merged_relevance = {}
    for hop in case.hops:
        merged_relevance.update(hop.relevance)
    pseudo = _hop_as_query_case(case, "merged", merged_relevance)
    scored = score_hits(hits, pseudo, weights)
    packed = _pack_chunks(
        "single_query_baseline",
        scored,
        pseudo,
        budget_tokens,
        max_per_document=1,
    )
    required_by_hop = []
    for hop in case.hops:
        req = {doc_id for doc_id, label in hop.relevance.items() if label >= 2}
        included = {chunk.document.id for chunk in packed.chunks}
        required_by_hop.append(len(req & included) / len(req) if req else 1.0)
    rendered = f"Question: {case.query}\nEvidence:\n" + "\n\n".join(_render_block(c) for c in packed.chunks)
    return AssembledContext(
        strategy="single_query_baseline",
        chunks=packed.chunks,
        rendered=rendered,
        token_count=estimate_tokens(rendered),
        dropped_unauthorized=packed.dropped_unauthorized,
        dropped_duplicate=packed.dropped_duplicate,
        budget_tokens=budget_tokens,
        hop_coverage=sum(required_by_hop) / len(required_by_hop),
    )


def _hop_as_query_case(case: MultiHopCase, hop_id: str, relevance: dict[str, int]):
    from .models import QueryCase

    return QueryCase(
        id=f"{case.id}:{hop_id}",
        query=case.query,
        relevance=relevance,
        tags=case.tags,
        answerable=True,
        allowed_tenants=case.allowed_tenants,
    )
