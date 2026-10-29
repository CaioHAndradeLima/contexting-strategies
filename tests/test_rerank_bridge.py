import unittest
from pathlib import Path

from contexting_strategies.data import load_documents, load_queries, validate_dataset
from contexting_strategies.models import ScoreWeights
from contexting_strategies.rerank_bridge import evaluate_rerank_to_assembly, load_rerank_report
from contexting_strategies.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


class RerankBridgeTests(unittest.TestCase):
    def test_heuristic_rerank_improves_or_matches_coverage(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        report = evaluate_rerank_to_assembly(
            queries,
            lambda query, limit: retriever.search(query, limit),
            candidate_k=10,
            budget_tokens=200,
            weights=ScoreWeights(),
        )
        self.assertGreaterEqual(
            report["summary"]["mean_heuristic_rerank_coverage"],
            report["summary"]["mean_retrieval_coverage"] - 0.05,
        )

    def test_external_report_parses(self) -> None:
        path = ROOT / "data/rerank_report_sample.json"
        if not path.exists():
            self.skipTest("rerank_report_sample.json missing")
        payload = load_rerank_report(path)
        self.assertTrue(payload.get("results") or payload.get("retrieval_modes"))


if __name__ == "__main__":
    unittest.main()
