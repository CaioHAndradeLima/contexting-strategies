"""Contexting pipeline evaluation scorecard (assembly + abstention + multi-hop)."""

from pathlib import Path

from .data import load_documents, load_multihop_cases, load_queries, validate_dataset, validate_multihop_dataset
from .measure_profiles import MeasureProfile
from .measurement_pipeline import run_measurement_pipeline, select_assemblers
from .models import ScoreWeights
from .pipeline_config import PipelineConfig
from .retrieval import BM25Retriever
from .scenario_evaluation import evaluate_abstention_on_queries, evaluate_multihop


def build_scorecard(
    measure_report: dict,
    abstention: dict,
    multihop: dict,
    *,
    primary_strategy: str = "pipeline_acl_dedupe_score_diverse",
) -> dict:
    assembly = measure_report.get("summary", {}).get(primary_strategy, {})
    mh = multihop.get("summary", {})
    baseline_cov = mh.get("single_query_baseline", {}).get("hop_coverage", 0.0)
    multihop_cov = mh.get("multihop_fair_budget_pipeline", {}).get("hop_coverage", 0.0)
    return {
        "primary_strategy": primary_strategy,
        "assembly": {
            "evidence_coverage": assembly.get("evidence_coverage"),
            "redundancy_rate": assembly.get("redundancy_rate"),
            "citation_traceability": assembly.get("citation_traceability"),
            "relevant_evidence_per_token": assembly.get("relevant_evidence_per_token"),
            "unauthorized_leak": assembly.get("unauthorized_leak"),
            "assembly_ms_p95": assembly.get("assembly_ms_p95"),
            "abstain_correct_rate": assembly.get("abstain_correct"),
        },
        "abstention": abstention.get("summary", {}),
        "multihop": {
            "baseline_hop_coverage": baseline_cov,
            "multihop_hop_coverage": multihop_cov,
            "lift_over_baseline": multihop_cov - baseline_cov,
        },
        "reference_doc": "context-evaluate-assembling.md",
    }


def run_contexting_pipeline_eval(
    root: Path,
    profile: MeasureProfile,
    *,
    weights: ScoreWeights | None = None,
    multihop_budget_tokens: int = 180,
) -> dict:
    weights = weights or ScoreWeights()
    documents = load_documents(root / "data/documents.json")
    queries = load_queries(root / "data/golden.json")
    validate_dataset(documents, queries)
    multihop_cases = load_multihop_cases(root / "data/multihop_golden.json")
    validate_multihop_dataset(documents, multihop_cases)
    retriever = BM25Retriever(documents)
    search = lambda query, limit: retriever.search(query, limit)

    config = PipelineConfig(
        candidate_k=profile.candidate_k,
        budget_tokens=profile.budget_tokens,
        apply_heuristic_rerank=profile.apply_heuristic_rerank,
        live_llm=profile.live_llm,
        max_llm_calls=profile.max_llm_calls,
        min_coverage=0.34,
    )
    measure_report = run_measurement_pipeline(
        queries,
        search,
        config,
        weights=weights,
        assemblers=select_assemblers(list(profile.strategies)),
    )
    abstention = evaluate_abstention_on_queries(
        queries,
        search,
        candidate_k=profile.candidate_k,
        budget_tokens=profile.budget_tokens,
        weights=weights,
        min_coverage=0.34,
    )
    multihop = evaluate_multihop(
        multihop_cases,
        search,
        budget_tokens=multihop_budget_tokens,
        weights=weights,
        candidate_k=profile.candidate_k,
    )
    primary = profile.strategies[0] if profile.strategies else "pipeline_acl_dedupe_score_diverse"
    scorecard = build_scorecard(measure_report, abstention, multihop, primary_strategy=primary)
    return {
        "eval_type": "contexting_pipeline",
        "profile": {"name": profile.name, "description": profile.description},
        "scorecard": scorecard,
        **measure_report,
        "scenarios": {"abstention": abstention, "multihop": multihop},
    }
