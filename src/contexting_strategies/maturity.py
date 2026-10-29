"""Lab capability snapshot vs context-architecturing-decision-based.md rows."""

from importlib.metadata import version


def lab_maturity_snapshot() -> dict:
    pkg = "contexting-strategies"
    try:
        pkg_version = version(pkg)
    except Exception:
        pkg_version = "unknown"

    capabilities = {
        "context_assembly": "Dedicated Python module with multiple assemblers + pipeline",
        "evidence_selection": "S(d) scoring, dedupe, diversity, rerank bridge",
        "token_budget": "Fixed budget per command; profiles for CI stress",
        "conversation_history": "full_history, last_n, summary_plus_recent fixtures",
        "source_metadata": "citation_id, doc_id, title in rendered blocks",
        "security": "ACL, injection phrase scan, unauthorized leak metrics",
        "prompt_injection": "sanitize_chunk_text + reliability filters",
        "citation_handling": "citation_traceability metric",
        "evaluation": "evaluate, evaluate-scenarios, evaluate-full, measure, check-measure",
        "observability": "latency ms, workflow stage timings, LLM usage totals in measure",
        "failure_handling": "sufficiency gate + abstention metrics",
        "cost_optimization": "token budgets, redundancy metrics, measure call caps",
    }
    # Rough mapping to doc columns (lab = growing production tier for most rows)
    maturity_column = {
        key: "growing_production" if key not in {"observability", "cost_optimization"} else "startup_to_growing"
        for key in capabilities
    }
    return {
        "package_version": pkg_version,
        "reference_doc": "context-architecturing-decision-based.md",
        "capabilities": capabilities,
        "maturity_column_hint": maturity_column,
        "not_in_lab": [
            "enterprise multi-agent configurable policies per agent",
            "distributed traces and per-tenant production dashboards",
            "real warehouse SQL connector (analytical uses toy structured_results.json)",
        ],
    }
