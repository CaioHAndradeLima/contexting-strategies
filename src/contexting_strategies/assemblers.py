import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable

from .models import AssembledContext, ContextChunk, Hit, QueryCase, ScoreWeights
from .scoring import score_hits
from .tokens import estimate_document_tokens, estimate_tokens


def _content_fingerprint(document) -> str:
    payload = document.text.strip().lower()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tenant_allowed(document, allowed_tenants: Iterable[str]) -> bool:
    tenant = document.metadata.get("tenant")
    if tenant is None:
        return True
    return tenant in set(allowed_tenants)


def _render_block(chunk: ContextChunk) -> str:
    meta = chunk.document.metadata
    source = meta.get("source_label", chunk.document.title)
    return (
        f"[{chunk.citation_id}] source={source} doc_id={chunk.document.id}\n"
        f"{chunk.document.title}\n{chunk.document.text}"
    )


def _pack_chunks(
    strategy: str,
    pairs: list[tuple[Hit, float]],
    case: QueryCase,
    budget_tokens: int,
    *,
    max_per_document: int | None = None,
    reserve_instruction_tokens: int = 40,
    enforce_tenant_acl: bool = True,
) -> AssembledContext:
    effective_budget = max(0, budget_tokens - reserve_instruction_tokens)
    selected: list[ContextChunk] = []
    used_tokens = 0
    per_doc: dict[str, int] = {}
    dropped_duplicate = 0
    seen_hashes: set[str] = set()

    for hit, assembly_score in pairs:
        doc = hit.document
        if enforce_tenant_acl and not _tenant_allowed(doc, case.allowed_tenants):
            continue
        fingerprint = doc.metadata.get("content_hash") or _content_fingerprint(doc)
        if fingerprint in seen_hashes:
            dropped_duplicate += 1
            continue
        if max_per_document is not None:
            count = per_doc.get(doc.id, 0)
            if count >= max_per_document:
                continue
        chunk_tokens = estimate_document_tokens(doc)
        if used_tokens + chunk_tokens > effective_budget and selected:
            break
        if chunk_tokens > effective_budget and not selected:
            continue
        citation_id = doc.metadata.get("citation_id", doc.id)
        selected.append(
            ContextChunk(
                document=doc,
                retrieval_score=hit.score,
                assembly_score=assembly_score,
                citation_id=str(citation_id),
            )
        )
        seen_hashes.add(fingerprint)
        per_doc[doc.id] = per_doc.get(doc.id, 0) + 1
        used_tokens += chunk_tokens

    blocks = [_render_block(chunk) for chunk in selected]
    header = f"Question: {case.query}\nEvidence:\n"
    rendered = header + "\n\n".join(blocks)
    return AssembledContext(
        strategy=strategy,
        chunks=tuple(selected),
        rendered=rendered,
        token_count=estimate_tokens(rendered),
        dropped_unauthorized=0,
        dropped_duplicate=dropped_duplicate,
        budget_tokens=budget_tokens,
    )


class ContextAssembler(ABC):
    name: str

    @abstractmethod
    def assemble(
        self,
        case: QueryCase,
        hits: list[Hit],
        budget_tokens: int,
        weights: ScoreWeights,
    ) -> AssembledContext:
        raise NotImplementedError


class SimpleDocumentAssembler(ContextAssembler):
    """MVP single-corpus shape: question, top chunks, citation IDs (tenant-safe)."""

    name = "simple_document_mvp"

    def assemble(
        self,
        case: QueryCase,
        hits: list[Hit],
        budget_tokens: int,
        weights: ScoreWeights,
    ) -> AssembledContext:
        allowed = [hit for hit in hits if _tenant_allowed(hit.document, case.allowed_tenants)]
        pairs = [(hit, hit.score) for hit in allowed]
        context = _pack_chunks(self.name, pairs, case, budget_tokens, max_per_document=2)
        blocks = []
        for chunk in context.chunks:
            blocks.append(
                f"[{chunk.citation_id}] doc_id={chunk.document.id} title={chunk.document.title}\n"
                f"{chunk.document.text}"
            )
        rendered = f"Question:\n{case.query}\n\nSources:\n" + "\n\n".join(blocks)
        return AssembledContext(
            strategy=context.strategy,
            chunks=context.chunks,
            rendered=rendered,
            token_count=estimate_tokens(rendered),
            dropped_unauthorized=context.dropped_unauthorized,
            dropped_duplicate=context.dropped_duplicate,
            budget_tokens=context.budget_tokens,
        )


class NaiveRetrievalOrderAssembler(ContextAssembler):
    """Baseline: keep retrieval order and fill the budget (include-every-chunk anti-pattern)."""

    name = "naive_retrieval_order"

    def assemble(
        self,
        case: QueryCase,
        hits: list[Hit],
        budget_tokens: int,
        weights: ScoreWeights,
    ) -> AssembledContext:
        pairs = [(hit, hit.score) for hit in hits]
        return _pack_chunks(
            self.name,
            pairs,
            case,
            budget_tokens,
            max_per_document=None,
            enforce_tenant_acl=False,
        )


class WeightedScoreAssembler(ContextAssembler):
    """Select by S(d) from context-scoring-math, then pack by budget."""

    name = "weighted_score"

    def assemble(
        self,
        case: QueryCase,
        hits: list[Hit],
        budget_tokens: int,
        weights: ScoreWeights,
    ) -> AssembledContext:
        scored = score_hits(hits, case, weights)
        return _pack_chunks(self.name, scored, case, budget_tokens, max_per_document=None)


class PipelineAssembler(ContextAssembler):
    """ACL filter, dedupe, weighted score, diversity cap, budget, citation render."""

    name = "pipeline_acl_dedupe_score_diverse"

    def assemble(
        self,
        case: QueryCase,
        hits: list[Hit],
        budget_tokens: int,
        weights: ScoreWeights,
    ) -> AssembledContext:
        allowed: list[Hit] = []
        dropped_unauthorized = 0
        for hit in hits:
            if _tenant_allowed(hit.document, case.allowed_tenants):
                allowed.append(hit)
            else:
                dropped_unauthorized += 1
        scored = score_hits(allowed, case, weights)
        context = _pack_chunks(
            self.name,
            scored,
            case,
            budget_tokens,
            max_per_document=1,
        )
        return AssembledContext(
            strategy=context.strategy,
            chunks=context.chunks,
            rendered=context.rendered,
            token_count=context.token_count,
            dropped_unauthorized=dropped_unauthorized,
            dropped_duplicate=context.dropped_duplicate,
            budget_tokens=context.budget_tokens,
        )


def default_assemblers() -> list[ContextAssembler]:
    return [
        NaiveRetrievalOrderAssembler(),
        WeightedScoreAssembler(),
        PipelineAssembler(),
    ]
