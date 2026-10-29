import argparse
import json
import platform
import sys
from pathlib import Path

from .assemblers import default_assemblers
from .data import (
    load_documents,
    load_multihop_cases,
    load_queries,
    load_thread_scenarios,
    validate_dataset,
    validate_multihop_dataset,
)
from .evaluation import evaluate
from .models import ScoreWeights
from .retrieval import BM25Retriever
from .full_evaluation import run_full_evaluation
from .llm_factory import build_live_llm_stack, resolve_llm_provider
from .measure_profiles import evaluate_gates, load_measure_profiles
from .measurement_pipeline import run_measurement_pipeline, select_assemblers
from .pipeline_config import PipelineConfig
from .pricing import OPENAI_STANDARD_RATES_PER_MILLION, PRICING_AS_OF, PRICING_SOURCE
from .pipeline_evaluation import run_contexting_pipeline_eval
from .scenario_evaluation import evaluate_abstention_on_queries, evaluate_multihop, evaluate_threads

ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = ROOT / "config" / "measure_profiles.json"


def _resolve_profile(name: str):
    profiles = load_measure_profiles(PROFILES_PATH)
    if name not in profiles:
        raise SystemExit(f"Unknown profile '{name}'. Choose from: {', '.join(sorted(profiles))}")
    return profiles[name]


def _environment_note() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "note": "Synthetic demo corpus; metrics measure assembled prompt quality, not LLM answers.",
    }


def _run_evaluate(args: argparse.Namespace) -> dict:
    documents = load_documents(Path(args.documents))
    queries = load_queries(Path(args.queries))
    validate_dataset(documents, queries)
    weights = ScoreWeights(
        relevance=args.weight_relevance,
        freshness=args.weight_freshness,
        authority=args.weight_authority,
        task=args.weight_task,
    )
    retriever = BM25Retriever(documents)
    report = evaluate(
        queries,
        lambda query, limit: retriever.search(query, limit),
        default_assemblers(),
        candidate_k=args.candidate_k,
        budget_tokens=args.budget_tokens,
        weights=weights,
    )
    report["environment"] = _environment_note()
    return report


def _run_evaluate_scenarios(args: argparse.Namespace) -> dict:
    documents = load_documents(Path(args.documents))
    weights = ScoreWeights(
        relevance=args.weight_relevance,
        freshness=args.weight_freshness,
        authority=args.weight_authority,
        task=args.weight_task,
    )
    retriever = BM25Retriever(documents)

    queries = load_queries(Path(args.queries))
    validate_dataset(documents, queries)
    multihop = load_multihop_cases(Path(args.multihop))
    validate_multihop_dataset(documents, multihop)
    threads = load_thread_scenarios(Path(args.threads))

    report = {
        "abstention": evaluate_abstention_on_queries(
            queries,
            lambda query, limit: retriever.search(query, limit),
            candidate_k=args.candidate_k,
            budget_tokens=max(args.budget_tokens, 200),
            weights=weights,
            min_coverage=0.34,
        ),
        "multihop": evaluate_multihop(
            multihop,
            lambda query, limit: retriever.search(query, limit),
            budget_tokens=args.multihop_budget_tokens,
            weights=weights,
            candidate_k=args.candidate_k,
        ),
        "conversation": evaluate_threads(
            threads,
            lambda query, limit: retriever.search(query, limit),
            weights=weights,
            candidate_k=args.candidate_k,
        ),
        "environment": _environment_note(),
    }
    return report


def _run_measure(args: argparse.Namespace) -> dict:
    documents = load_documents(Path(args.documents))
    queries = load_queries(Path(args.queries))
    validate_dataset(documents, queries)
    env_config = PipelineConfig.from_env()
    profile = _resolve_profile(args.profile) if args.profile else None
    candidate_k = profile.candidate_k if profile else args.candidate_k
    budget_tokens = profile.budget_tokens if profile else args.budget_tokens
    apply_rerank = profile.apply_heuristic_rerank if profile else not args.no_rerank
    live_llm = args.live_llm or env_config.live_llm or (profile.live_llm if profile else False)
    max_llm_calls = profile.max_llm_calls if profile else args.max_llm_calls
    config = PipelineConfig(
        candidate_k=candidate_k,
        budget_tokens=budget_tokens,
        apply_heuristic_rerank=apply_rerank,
        live_llm=live_llm,
        answer_model=args.answer_model or env_config.answer_model,
        judge_model=args.judge_model or env_config.judge_model,
        max_llm_calls=max_llm_calls,
        min_coverage=args.min_coverage,
    )
    weights = ScoreWeights(
        relevance=args.weight_relevance,
        freshness=args.weight_freshness,
        authority=args.weight_authority,
        task=args.weight_task,
    )
    if profile:
        assemblers = select_assemblers(list(profile.strategies))
    else:
        assemblers = select_assemblers([s.strip() for s in args.strategies.split(",") if s.strip()] or None)
    retriever = BM25Retriever(documents)

    context_judge = answer_generator = answer_judge = None
    llm_provider = resolve_llm_provider(getattr(args, "llm_provider", None) or None)
    if config.live_llm:
        context_judge, answer_generator, answer_judge, llm_provider = build_live_llm_stack(
            config,
            provider=getattr(args, "llm_provider", None) or None,
            cwd=ROOT,
        )

    report = run_measurement_pipeline(
        queries,
        lambda query, limit: retriever.search(query, limit),
        config,
        weights=weights,
        assemblers=assemblers,
        context_judge=context_judge,
        answer_generator=answer_generator,
        answer_judge=answer_judge,
    )
    if profile:
        report["profile"] = {"name": profile.name, "description": profile.description}
    report["environment"] = _environment_note()
    if config.live_llm:
        report["environment"]["llm_provider"] = llm_provider
        if llm_provider == "openai":
            report["environment"]["pricing_as_of"] = PRICING_AS_OF
            report["environment"]["pricing_source"] = PRICING_SOURCE
            report["environment"]["pricing_rates_usd_per_million_tokens"] = OPENAI_STANDARD_RATES_PER_MILLION
        else:
            report["environment"]["pricing_note"] = (
                "Cursor local agent; use CURSOR_API_KEY from Cursor Integrations (no OpenAI key required)."
            )
    return report


def _run_evaluate_pipeline(args: argparse.Namespace) -> dict:
    if not args.profile:
        raise SystemExit("evaluate-pipeline requires --profile (e.g. ci-contexting-eval)")
    profile = _resolve_profile(args.profile)
    weights = ScoreWeights(
        relevance=args.weight_relevance,
        freshness=args.weight_freshness,
        authority=args.weight_authority,
        task=args.weight_task,
    )
    report = run_contexting_pipeline_eval(
        ROOT,
        profile,
        weights=weights,
        multihop_budget_tokens=args.multihop_budget_tokens,
    )
    report["environment"] = _environment_note()
    if profile.live_llm:
        report["environment"]["note_live_llm"] = (
            "For live judges use measure --profile with live_llm; evaluate-pipeline focuses on offline scorecard."
        )
    return report


def _run_check_measure(args: argparse.Namespace) -> dict:
    if not args.report:
        raise SystemExit("check-measure requires --report")
    if not args.profile:
        raise SystemExit("check-measure requires --profile")
    report = json.loads(Path(args.report).read_text())
    profile = _resolve_profile(args.profile)
    return evaluate_gates(report, profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate context assembly strategies for RAG prompts")
    parser.add_argument(
        "command",
        choices=[
            "evaluate",
            "evaluate-scenarios",
            "evaluate-full",
            "evaluate-pipeline",
            "measure",
            "check-measure",
        ],
    )
    parser.add_argument("--documents", default=str(ROOT / "data/documents.json"))
    parser.add_argument("--queries", default=str(ROOT / "data/golden.json"))
    parser.add_argument("--multihop", default=str(ROOT / "data/multihop_golden.json"))
    parser.add_argument("--threads", default=str(ROOT / "data/threads.json"))
    parser.add_argument("--output", default="")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--budget-tokens", type=int, default=120)
    parser.add_argument("--multihop-budget-tokens", type=int, default=180)
    parser.add_argument("--weight-relevance", type=float, default=0.45)
    parser.add_argument("--weight-freshness", type=float, default=0.20)
    parser.add_argument("--weight-authority", type=float, default=0.20)
    parser.add_argument("--weight-task", type=float, default=0.15)
    parser.add_argument("--strategies", default="", help="Comma-separated assembler names for measure")
    parser.add_argument("--live-llm", action="store_true", help="Run answer generation and LLM judges")
    parser.add_argument(
        "--llm-provider",
        default="",
        help="openai (API key) or cursor (local Cursor agent via CURSOR_API_KEY); auto if unset",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Skip heuristic rerank before assembly in measure")
    parser.add_argument("--answer-model", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--min-coverage", type=float, default=0.34)
    parser.add_argument("--profile", default="", help="Measure profile from config/measure_profiles.json")
    parser.add_argument("--report", default="", help="Measure report JSON for check-measure")
    args = parser.parse_args()

    if args.command == "evaluate":
        report = _run_evaluate(args)
    elif args.command == "evaluate-scenarios":
        report = _run_evaluate_scenarios(args)
    elif args.command == "measure":
        report = _run_measure(args)
    elif args.command == "evaluate-pipeline":
        report = _run_evaluate_pipeline(args)
    elif args.command == "check-measure":
        report = _run_check_measure(args)
    else:
        weights = ScoreWeights(
            relevance=args.weight_relevance,
            freshness=args.weight_freshness,
            authority=args.weight_authority,
            task=args.weight_task,
        )
        report = run_full_evaluation(
            ROOT,
            candidate_k=args.candidate_k,
            budget_tokens=args.budget_tokens,
            multihop_budget_tokens=args.multihop_budget_tokens,
            weights=weights,
        )
        report["environment"] = _environment_note()

    serialized = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)
    if args.command == "check-measure" and not report.get("passed", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
