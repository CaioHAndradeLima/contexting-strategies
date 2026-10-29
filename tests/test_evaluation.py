import unittest

from contexting_strategies.assemblers import default_assemblers
from contexting_strategies.data import load_documents, load_queries, validate_dataset
from contexting_strategies.evaluation import evaluate
from contexting_strategies.models import ScoreWeights
from contexting_strategies.retrieval import BM25Retriever
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_end_to_end_report_has_summary(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        report = evaluate(
            queries,
            lambda query, limit: retriever.search(query, limit),
            default_assemblers(),
            candidate_k=8,
            budget_tokens=120,
            weights=ScoreWeights(),
        )
        self.assertIn("pipeline_acl_dedupe_score_diverse", report["summary"])
        pipeline = report["summary"]["pipeline_acl_dedupe_score_diverse"]
        self.assertGreaterEqual(pipeline["evidence_coverage"], 0.0)
        self.assertEqual(pipeline["unauthorized_leak"], 0.0)

    def test_pipeline_beats_naive_on_redundancy_or_coverage(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        retriever = BM25Retriever(documents)
        report = evaluate(
            queries,
            lambda query, limit: retriever.search(query, limit),
            default_assemblers(),
            candidate_k=10,
            budget_tokens=100,
            weights=ScoreWeights(),
        )
        pipeline = report["summary"]["pipeline_acl_dedupe_score_diverse"]
        naive = report["summary"]["naive_retrieval_order"]
        self.assertLessEqual(pipeline["redundancy_rate"], naive["redundancy_rate"] + 0.05)
        self.assertLessEqual(pipeline["unauthorized_leak"], naive["unauthorized_leak"])


if __name__ == "__main__":
    unittest.main()
