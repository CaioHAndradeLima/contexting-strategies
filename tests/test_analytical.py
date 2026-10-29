import unittest
from pathlib import Path

from contexting_strategies.analytical import assemble_analytical, structured_numeric_support
from contexting_strategies.data import load_analytical_cases, load_documents, load_structured_results, validate_analytical_dataset
from contexting_strategies.models import ScoreWeights
from contexting_strategies.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


class AnalyticalTests(unittest.TestCase):
    def test_structured_block_in_prompt(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        cases = load_analytical_cases(ROOT / "data/analytical_golden.json")
        structured = load_structured_results(ROOT / "data/structured_results.json")
        validate_analytical_dataset(documents, cases, structured)
        retriever = BM25Retriever(documents)
        case = cases[0]
        hits = retriever.search(case.query, 8)
        row = assemble_analytical(case, hits, structured[case.structured_id], ScoreWeights())
        self.assertTrue(row["structured_block_present"])
        self.assertGreater(row["required_metrics_score"], 0.5)
        support = structured_numeric_support(case.reference_answer or "", structured[case.structured_id])
        self.assertGreaterEqual(support, 0.5)


if __name__ == "__main__":
    unittest.main()
