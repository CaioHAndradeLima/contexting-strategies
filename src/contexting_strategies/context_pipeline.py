"""Explicit context-algorithm steps 1–8 (context-algorithm.md)."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from .compacting import compact_document_text
from .models import Document, Hit, QueryCase, ScoreWeights
from .scoring import score_hits
from .tokens import estimate_document_tokens


def _tenant_allowed(document: Document, allowed_tenants: Iterable[str]) -> bool:
    tenant = document.metadata.get("tenant")
    if tenant is None:
        return True
    return tenant in set(allowed_tenants)


def step1_validate_access(hits: list[Hit], case: QueryCase) -> tuple[list[Hit], int]:
    allowed: list[Hit] = []
    dropped = 0
    for hit in hits:
        if _tenant_allowed(hit.document, case.allowed_tenants):
            allowed.append(hit)
        else:
            dropped += 1
    return allowed, dropped


def step2_remove_invalid_stale_duplicate(hits: list[Hit]) -> tuple[list[Hit], dict]:
    kept: list[Hit] = []
    stats = {"tombstone": 0, "superseded": 0, "duplicate_hash": 0}
    seen_hashes: set[str] = set()
    for hit in hits:
        meta = hit.document.metadata
        if meta.get("tombstone"):
            stats["tombstone"] += 1
            continue
        if meta.get("superseded_by"):
            stats["superseded"] += 1
            continue
        fingerprint = meta.get("content_hash") or hit.document.text.strip().lower()
        if fingerprint in seen_hashes:
            stats["duplicate_hash"] += 1
            continue
        seen_hashes.add(fingerprint)
        kept.append(hit)
    return kept, stats


def step3_merge_parent_child(hits: list[Hit]) -> tuple[list[Hit], int]:
    by_id = {hit.document.id: hit for hit in hits}
    merged = 0
    for hit in list(hits):
        parent_id = hit.document.metadata.get("parent_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        combined = Document(
            parent.document.id,
            parent.document.title,
            f"{parent.document.text}\n{hit.document.text}",
            parent.document.metadata,
        )
        by_id[parent_id] = Hit(combined, max(parent.score, hit.score), parent.source)
        if hit.document.id in by_id:
            del by_id[hit.document.id]
            merged += 1
    return list(by_id.values()), merged


def step4_score(hits: list[Hit], case: QueryCase, weights: ScoreWeights) -> list[tuple[Hit, float]]:
    return score_hits(hits, case, weights)


def step5_apply_diversity(scored: list[tuple[Hit, float]], max_per_document: int = 1) -> list[tuple[Hit, float]]:
    per_doc: dict[str, int] = {}
    selected: list[tuple[Hit, float]] = []
    for hit, score in scored:
        count = per_doc.get(hit.document.id, 0)
        if count >= max_per_document:
            continue
        per_doc[hit.document.id] = count + 1
        selected.append((hit, score))
    return selected


def step6_allocate_budget(
    pairs: list[tuple[Hit, float]],
    budget_tokens: int,
    *,
    compact_mode: str = "truncate",
    reserve_instruction_tokens: int = 40,
) -> tuple[list[tuple[Hit, float, str]], dict]:
    effective = max(0, budget_tokens - reserve_instruction_tokens)
    packed: list[tuple[Hit, float, str]] = []
    used = 0
    compact_stats = {"chunks_compacted": 0, "mode": compact_mode}
    for hit, score in pairs:
        text = hit.document.text
        tokens = estimate_document_tokens(hit.document)
        if tokens > effective // 2 and effective > 30:
            compact = compact_document_text(text, max(20, effective // 3), mode=compact_mode)
            text = compact["text"]
            tokens = compact["compact_tokens"]
            compact_stats["chunks_compacted"] += 1
        if used + tokens > effective and packed:
            break
        if tokens > effective and not packed:
            compact = compact_document_text(text, effective, mode=compact_mode)
            text = compact["text"]
            tokens = compact["compact_tokens"]
            compact_stats["chunks_compacted"] += 1
        if tokens > effective:
            continue
        packed.append((hit, score, text))
        used += int(tokens)
    compact_stats["used_tokens"] = used
    compact_stats["budget_tokens"] = budget_tokens
    return packed, compact_stats


@dataclass
class AlgorithmTrace:
    query_id: str
    steps: list[dict] = field(default_factory=list)
    dropped_unauthorized: int = 0
    merged_parent_child: int = 0


def preprocess_candidates(hits: list[Hit], case: QueryCase) -> tuple[list[Hit], AlgorithmTrace]:
    trace = AlgorithmTrace(query_id=case.id)
    step1, dropped = step1_validate_access(hits, case)
    trace.dropped_unauthorized = dropped
    trace.steps.append({"step": 1, "name": "validate_access", "remaining": len(step1), "dropped_unauthorized": dropped})

    step2, dup_stats = step2_remove_invalid_stale_duplicate(step1)
    trace.steps.append({"step": 2, "name": "remove_invalid_stale_duplicate", "remaining": len(step2), **dup_stats})

    step3, merged = step3_merge_parent_child(step2)
    trace.merged_parent_child = merged
    trace.steps.append({"step": 3, "name": "merge_parent_child", "remaining": len(step3), "merged": merged})
    return step3, trace


def continue_algorithm_trace(
    trace: AlgorithmTrace,
    case: QueryCase,
    hits: list[Hit],
    weights: ScoreWeights,
    budget_tokens: int,
) -> tuple[list[tuple[Hit, float, str]], AlgorithmTrace]:
    scored = step4_score(hits, case, weights)
    trace.steps.append({"step": 4, "name": "score_and_select", "candidates_scored": len(scored)})

    diverse = step5_apply_diversity(scored, max_per_document=1)
    trace.steps.append({"step": 5, "name": "diversity_and_coverage", "selected": len(diverse)})

    packed, compact_stats = step6_allocate_budget(diverse, budget_tokens)
    trace.steps.append({"step": 6, "name": "allocate_token_budget", **compact_stats})
    trace.steps.append({"step": 7, "name": "order_and_citation_ids", "chunk_count": len(packed)})
    trace.steps.append({"step": 8, "name": "render_final_context", "chunk_count": len(packed)})
    return packed, trace
