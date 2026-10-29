import json
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
from .algorithm_eval import evaluate_algorithm_steps
from .analytical_sql import run_validated_select
from .assembly_flow import summarize_rag_pipeline
from .compacting_eval import evaluate_compacting_demo
from .context_info_eval import evaluate_five_input_render
from .langgraph_workflow import run_langgraph_pipeline
from .maturity import lab_maturity_snapshot
from .query_resolution import evaluate_query_resolution
from .query_rewrite_eval import evaluate_query_rewrite
from .scoring_breakdown import evaluate_scoring_math
from .simple_document import evaluate_simple_document
from .task_context import assemble_for_agent
from .tradeoffs import evaluate_tradeoffs
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

    assembly = evaluate(
        queries,
        search,
        default_assemblers(),
        candidate_k=candidate_k,
        budget_tokens=budget_tokens,
        weights=weights,
    )
    threads = load_thread_scenarios(root / "data/threads.json")
    report = {
        "assembly": assembly,
        "tradeoffs": evaluate_tradeoffs(assembly),
        "scoring_math": evaluate_scoring_math(
            queries,
            search,
            candidate_k=candidate_k,
            weights=weights,
            sample_query_ids=["q-001", "q-004"],
        ),
        "simple_document": evaluate_simple_document(
            queries,
            search,
            candidate_k=candidate_k,
            budget_tokens=budget_tokens,
            weights=weights,
        ),
        "query_resolution": evaluate_query_resolution(threads, search, candidate_k=candidate_k, weights=weights),
        "assembly_flow": summarize_rag_pipeline(queries[0], search, candidate_k=candidate_k, budget_tokens=max(budget_tokens, 180), weights=weights),
        "maturity": lab_maturity_snapshot(),
        "algorithm_steps": evaluate_algorithm_steps(
            queries,
            search,
            candidate_k=candidate_k,
            budget_tokens=max(budget_tokens, 180),
            weights=weights,
            sample_ids=["q-001"],
        ),
        "compacting": evaluate_compacting_demo(),
        "query_rewrite": evaluate_query_rewrite(threads),
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
                threads,
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
    sql_rows = []
    for row in json.loads((root / "data/analytical_queries.json").read_text()):
        structured_block = structured[row["structured_id"]]
        sql_rows.append(
            {
                "query_id": row["id"],
                "sql": row["sql"],
                "result": run_validated_select(row["sql"], structured_block),
            }
        )
    report["analytical"] = {"cases": analytical_rows, "validated_sql": sql_rows}

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
    docs_by_id = {doc.id: doc for doc in documents}
    report["langgraph"] = {
        "cases": [
            run_langgraph_pipeline(
                case,
                search,
                docs_by_id,
                budget_tokens=max(budget_tokens, 180),
                weights=weights,
            )
            for case in queries[:3]
        ]
    }
    report["context_info_for_output"] = evaluate_five_input_render(
        queries[0],
        threads[0],
        structured["sql-checkout-conversion"],
        search,
        candidate_k=candidate_k,
        weights=weights,
    )
    sample_hits = search(queries[0].query, candidate_k)
    report["task_context"] = {
        "billing_agent": assemble_for_agent(
            queries[0],
            sample_hits,
            agent_team="billing",
            budget_tokens=max(budget_tokens, 160),
            weights=weights,
        ),
        "platform_agent": assemble_for_agent(
            queries[0],
            sample_hits,
            agent_team="platform",
            budget_tokens=max(budget_tokens, 160),
            weights=weights,
        ),
    }
    report["coverage"] = {
        "contexting_guides": "14/14",
        "lab_coverage_percent": 100,
        "index": "docs/LAB_COVERAGE.md",
    }
    return report
