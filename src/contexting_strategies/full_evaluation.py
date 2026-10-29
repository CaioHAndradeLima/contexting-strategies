from pathlib import Path

from .analytical import assemble_analytical, structured_numeric_support
from .assemblers import default_assemblers
from .data import (
    load_analytical_cases,
    load_answer_samples,
    load_documents,
    load_multihop_cases,
    load_queries,
    load_structured_results,
    load_thread_scenarios,
    validate_analytical_dataset,
    validate_dataset,
    validate_multihop_dataset,
)
from .evaluation import evaluate
from .groundedness import evaluate_answer_samples
from .models import QueryCase, ScoreWeights
from .rerank_bridge import evaluate_rerank_to_assembly, load_rerank_report
from .retrieval import BM25Retriever
from .scenario_evaluation import evaluate_abstention_on_queries, evaluate_multihop, evaluate_threads
from .workflow import run_context_workflow


def run_full_evaluation(
    root: Path,
    *,
    candidate_k: int = 10,
    budget_tokens: int = 120,
    multihop_budget_tokens: int = 180,
    weights: ScoreWeights | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    documents = load_documents(root / "data/documents.json")
    queries = load_queries(root / "data/golden.json")
    validate_dataset(documents, queries)
    multihop = load_multihop_cases(root / "data/multihop_golden.json")
    validate_multihop_dataset(documents, multihop)
    retriever = BM25Retriever(documents)
    search = lambda query, limit: retriever.search(query, limit)

    report = {
        "assembly": evaluate(
            queries,
            search,
            default_assemblers(),
            candidate_k=candidate_k,
            budget_tokens=budget_tokens,
            weights=weights,
        ),
        "scenarios": {
            "abstention": evaluate_abstention_on_queries(
                queries,
                search,
                candidate_k=candidate_k,
                budget_tokens=max(budget_tokens, 200),
                weights=weights,
                min_coverage=0.34,
            ),
            "multihop": evaluate_multihop(
                multihop,
                search,
                budget_tokens=multihop_budget_tokens,
                weights=weights,
                candidate_k=candidate_k,
            ),
            "conversation": evaluate_threads(
                load_thread_scenarios(root / "data/threads.json"),
                search,
                weights=weights,
                candidate_k=candidate_k,
            ),
        },
    }

    rerank_path = root / "data/rerank_report_sample.json"
    external = load_rerank_report(rerank_path) if rerank_path.exists() else None
    report["rerank_bridge"] = evaluate_rerank_to_assembly(
        queries,
        search,
        candidate_k=candidate_k,
        budget_tokens=max(budget_tokens, 180),
        weights=weights,
        external_report=external,
    )

    analytical_cases = load_analytical_cases(root / "data/analytical_golden.json")
    structured = load_structured_results(root / "data/structured_results.json")
    validate_analytical_dataset(documents, analytical_cases, structured)
    analytical_rows = []
    for case in analytical_cases:
        hits = search(case.query, candidate_k)
        row = assemble_analytical(case, hits, structured[case.structured_id], weights)
        sample_answer = (case.reference_answer or "").strip()
        if sample_answer:
            row["structured_numeric_support"] = structured_numeric_support(sample_answer, structured[case.structured_id])
        analytical_rows.append({"query_id": case.id, **row})
    report["analytical"] = {"cases": analytical_rows}

    samples = load_answer_samples(root / "data/answer_samples.json")
    analytical_as_queries = [
        QueryCase(
            id=case.id,
            query=case.query,
            relevance=case.relevance,
            tags=case.tags,
            answerable=case.answerable,
            reference_answer=case.reference_answer,
        )
        for case in analytical_cases
    ]
    report["groundedness"] = evaluate_answer_samples(queries + analytical_as_queries, samples, documents)

    workflow_rows = [run_context_workflow(case, search, candidate_k=candidate_k, budget_tokens=max(budget_tokens, 180), weights=weights) for case in queries]
    report["workflow"] = {
        "cases": [
            {
                "query_id": trace.query_id,
                "stages_ms": trace.stages_ms,
                "should_abstain": trace.should_abstain,
                "injection_detected": trace.injection_detected,
                "assembled_ids": trace.assembled_ids,
            }
            for trace in workflow_rows
        ]
    }
    return report
