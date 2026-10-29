import unittest
from pathlib import Path

from contexting_strategies.data import load_documents, load_multihop_cases, validate_multihop_dataset
from contexting_strategies.models import ScoreWeights
from contexting_strategies.multihop import assemble_multihop, assemble_single_query_baseline
from contexting_strategies.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


class MultiHopTests(unittest.TestCase):
    def test_multihop_beats_single_query_hop_coverage(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        cases = load_multihop_cases(ROOT / "data/multihop_golden.json")
        validate_multihop_dataset(documents, cases)
        retriever = BM25Retriever(documents)
        search = lambda query, limit: retriever.search(query, limit)
        case = cases[0]
        weights = ScoreWeights()
        single = assemble_single_query_baseline(case, search, budget_tokens=180, weights=weights)
        multi = assemble_multihop(case, search, budget_tokens=180, weights=weights)
        self.assertIsNotNone(single.hop_coverage)
        self.assertIsNotNone(multi.hop_coverage)
        self.assertGreater(multi.hop_coverage, single.hop_coverage)


if __name__ == "__main__":
    unittest.main()
