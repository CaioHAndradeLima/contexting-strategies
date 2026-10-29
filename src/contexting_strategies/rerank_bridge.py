import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from .assemblers import PipelineAssembler
from .heuristic_rerank import HeuristicReranker
from .metrics import evidence_coverage
from .models import Document, Hit, QueryCase, ScoreWeights
from .sufficiency import apply_sufficiency_gate


def hits_from_ranked_ids(ranked_ids: list[str], candidates: list[Hit]) -> list[Hit]:
    by_id = {hit.document.id: hit for hit in candidates}
    ordered: list[Hit] = []
    for doc_id in ranked_ids:
        if doc_id in by_id and doc_id not in {hit.document.id for hit in ordered}:
            ordered.append(by_id[doc_id])
    return ordered


def load_rerank_report(path: Path) -> dict:
    return json.loads(path.read_text())


def evaluate_rerank_to_assembly(
    queries: list[QueryCase],
    retriever: Callable[[str, int], list[Hit]],
    *,
    candidate_k: int = 10,
    metric_k: int = 5,
    budget_tokens: int = 180,
    weights: ScoreWeights | None = None,
    external_report: dict | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    reranker = HeuristicReranker()
    pipeline = PipelineAssembler()
    rows = []
    retrieval_coverages: list[float] = []
    rerank_coverages: list[float] = []
    external_coverages: list[float] = []

    report_by_query: dict[str, dict] = {}
    if external_report:
        rows = external_report.get("results", external_report.get("queries", []))
        if not rows and "retrieval_modes" in external_report:
            mode = external_report["retrieval_modes"]
            first = next(iter(mode.values()), {})
            rows = first.get("results", [])
        for row in rows:
            qid = row.get("query_id")
            if qid:
                report_by_query[qid] = row

    for case in queries:
        candidates = retriever(case.query, candidate_k)
        retrieval_order = candidates[:metric_k]
        reranked = reranker.rerank(case.query, candidates, metric_k)

        ctx_retrieval = apply_sufficiency_gate(
            pipeline.assemble(case, retrieval_order, budget_tokens, weights), case, min_coverage=0.34
        )
        ctx_rerank = apply_sufficiency_gate(
            pipeline.assemble(case, reranked, budget_tokens, weights), case, min_coverage=0.34
        )
        cov_retrieval = evidence_coverage(ctx_retrieval, case)
        cov_rerank = evidence_coverage(ctx_rerank, case)
        retrieval_coverages.append(cov_retrieval)
        rerank_coverages.append(cov_rerank)

        external_cov = None
        external_ids: list[str] = []
        ext_row = report_by_query.get(case.id)
        if ext_row and "strategies" in ext_row:
            heuristic = ext_row["strategies"].get("heuristic", {})
            external_ids = heuristic.get("ranked_ids", [])
        elif ext_row:
            external_ids = ext_row.get("ranked_ids", [])
        if external_ids:
            external_hits = hits_from_ranked_ids(external_ids, candidates)
            ctx_external = apply_sufficiency_gate(
                pipeline.assemble(case, external_hits, budget_tokens, weights), case, min_coverage=0.34
            )
            external_cov = evidence_coverage(ctx_external, case)
            external_coverages.append(external_cov)

        rows.append(
            {
                "query_id": case.id,
                "retrieval_coverage": cov_retrieval,
                "heuristic_rerank_coverage": cov_rerank,
                "external_report_coverage": external_cov,
                "retrieval_ranked_ids": [hit.document.id for hit in retrieval_order],
                "heuristic_ranked_ids": [hit.document.id for hit in reranked],
                "external_ranked_ids": external_ids or None,
            }
        )

    summary = {
        "mean_retrieval_coverage": mean(retrieval_coverages) if retrieval_coverages else 0.0,
        "mean_heuristic_rerank_coverage": mean(rerank_coverages) if rerank_coverages else 0.0,
    }
    if external_coverages:
        summary["mean_external_report_coverage"] = mean(external_coverages)
    return {"candidate_k": candidate_k, "metric_k": metric_k, "budget_tokens": budget_tokens, "rows": rows, "summary": summary}
