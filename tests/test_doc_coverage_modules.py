import unittest
from pathlib import Path

from contexting_strategies.assemblers import SimpleDocumentAssembler
from contexting_strategies.data import load_documents, load_queries, load_thread_scenarios, validate_dataset
from contexting_strategies.full_evaluation import run_full_evaluation
from contexting_strategies.models import ScoreWeights
from contexting_strategies.retrieval import BM25Retriever
from contexting_strategies.scoring_breakdown import breakdown_scoring_signals
from contexting_strategies.tradeoffs import evaluate_tradeoffs

ROOT = Path(__file__).resolve().parents[1]


class DocCoverageModuleTests(unittest.TestCase):
    def test_simple_document_shape(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")[:1]
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        case = queries[0]
        hits = retriever.search(case.query, 8)
        ctx = SimpleDocumentAssembler().assemble(case, hits, 120, ScoreWeights())
        self.assertIn("Question:", ctx.rendered)
        self.assertIn("Sources:", ctx.rendered)
        self.assertNotIn("Conversation history", ctx.rendered)

    def test_scoring_breakdown_has_signals(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")[:1]
        retriever = BM25Retriever(documents)
        rows = breakdown_scoring_signals(retriever.search(queries[0].query, 5), queries[0], ScoreWeights())
        self.assertTrue(rows)
        self.assertIn("S_combined", rows[0])

    def test_full_eval_includes_all_doc_tracks(self) -> None:
        report = run_full_evaluation(ROOT, budget_tokens=120)
        for key in (
            "tradeoffs",
            "scoring_math",
            "simple_document",
            "query_resolution",
            "assembly_flow",
            "maturity",
        ):
            self.assertIn(key, report, msg=f"missing {key}")
        self.assertIn("recommended_for_coverage", report["tradeoffs"])

    def test_tradeoffs_flags_naive(self) -> None:
        assembly = {"summary": {"naive_retrieval_order": {"evidence_coverage": 0.8, "redundancy_rate": 0.1, "context_utilization": 0.5, "unauthorized_leak": 0.1}}}
        out = evaluate_tradeoffs(assembly)
        naive = next(r for r in out["strategies"] if r["strategy"] == "naive_retrieval_order")
        self.assertIn("similarity_only_baseline", naive["anti_patterns_observed"])


if __name__ == "__main__":
    unittest.main()
