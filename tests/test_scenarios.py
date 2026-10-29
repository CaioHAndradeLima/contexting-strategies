import unittest
from pathlib import Path

from contexting_strategies.data import load_documents, load_queries, load_thread_scenarios, validate_dataset
from contexting_strategies.models import ScoreWeights
from contexting_strategies.retrieval import BM25Retriever
from contexting_strategies.scenario_evaluation import evaluate_abstention_on_queries, evaluate_threads

ROOT = Path(__file__).resolve().parents[1]


class ScenarioTests(unittest.TestCase):
    def test_abstention_perfect_on_golden_set(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        report = evaluate_abstention_on_queries(
            queries,
            lambda query, limit: retriever.search(query, limit),
            candidate_k=10,
            budget_tokens=200,
            weights=ScoreWeights(),
            min_coverage=0.34,
        )
        self.assertGreaterEqual(report["summary"]["abstain_correct_rate"], 1.0)

    def test_last_n_preserves_more_evidence_than_full_history(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        scenarios = load_thread_scenarios(ROOT / "data/threads.json")
        retriever = BM25Retriever(documents)
        report = evaluate_threads(
            scenarios,
            lambda query, limit: retriever.search(query, limit),
            weights=ScoreWeights(),
        )
        full = report["summary"]["pipeline_with_full_history"]["evidence_coverage"]
        last_n = report["summary"]["pipeline_with_last_n"]["evidence_coverage"]
        self.assertGreaterEqual(last_n, full)


if __name__ == "__main__":
    unittest.main()
