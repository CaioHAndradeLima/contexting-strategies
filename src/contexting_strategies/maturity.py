"""Lab capability snapshot vs context-architecturing-decision-based.md rows."""

from importlib.metadata import version


def lab_maturity_snapshot() -> dict:
    pkg = "contexting-strategies"
    try:
        pkg_version = version(pkg)
    except Exception:
        pkg_version = "unknown"

    capabilities = {
        "context_assembly": "Steps 1–8 in context_pipeline.py + pipeline assembler",
        "evidence_selection": "S(d), tombstone/stale filter, parent-child merge, rerank bridge",
        "token_budget": "Budget allocation + compacting (truncate + extractive summarize)",
        "conversation_history": "full_history, last_n, summary_plus_recent + query_rewrite",
        "source_metadata": "citation_id, doc_id, title; five-input render order in full_prompt.py",
        "security": "ACL, injection scan, sanitize_chunk_text, leak metrics",
        "prompt_injection": "Untrusted boundaries in render + reliability filters",
        "citation_handling": "citation_traceability + validate_answer_node in langgraph_workflow",
        "evaluation": "evaluate-full covers all contexting/*.md guides (see LAB_COVERAGE.md)",
        "observability": "Algorithm step trace, LangGraph node timings, measure LLM usage",
        "failure_handling": "Sufficiency gate, abstention, validate_answer routing",
        "cost_optimization": "Token budgets, redundancy, measure call caps",
        "analytical_sql": "Allowlisted SELECT against structured_results (analytical_sql.py)",
        "langgraph_nodes": "retrieve → assemble → generate → validate with exported graph spec",
        "multi_agent_task_context": "task_context.py team-scoped evidence slices",
    }
    maturity_column = {key: "growing_production" for key in capabilities}
    return {
        "package_version": pkg_version,
        "reference_doc": "context-architecturing-decision-based.md",
        "contexting_guides_covered": "14/14 (see docs/LAB_COVERAGE.md)",
        "lab_coverage_percent": 100,
        "capabilities": capabilities,
        "maturity_column_hint": maturity_column,
        "production_deployment_only": [
            "Hosted LangGraph/Fargate deployment wiring",
            "Live warehouse credentials and arbitrary SQL",
            "Organization-wide distributed tracing backends",
        ],
    }
