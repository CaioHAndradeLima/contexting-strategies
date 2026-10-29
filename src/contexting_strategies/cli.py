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
from .scenario_evaluation import evaluate_abstention_on_queries, evaluate_multihop, evaluate_threads

ROOT = Path(__file__).resolve().parents[2]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate context assembly strategies for RAG prompts")
    parser.add_argument("command", choices=["evaluate", "evaluate-scenarios"])
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
    args = parser.parse_args()

    if args.command == "evaluate":
        report = _run_evaluate(args)
    else:
        report = _run_evaluate_scenarios(args)

    serialized = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
