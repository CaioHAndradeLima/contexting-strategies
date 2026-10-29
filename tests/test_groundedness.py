import unittest
from pathlib import Path

from contexting_strategies.data import load_answer_samples, load_documents, load_queries, validate_dataset
from contexting_strategies.groundedness import evaluate_answer_samples, lexical_groundedness
from contexting_strategies.models import Document

ROOT = Path(__file__).resolve().parents[1]


class GroundednessTests(unittest.TestCase):
    def test_lexical_groundedness_high_when_supported(self) -> None:
        doc = Document("d", "t", "INC-4821 stuck billing worker remediation")
        score = lexical_groundedness("INC-4821 was caused by a stuck billing worker.", [doc])
        self.assertGreaterEqual(score, 0.5)

    def test_evaluate_samples(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)
        samples = load_answer_samples(ROOT / "data/answer_samples.json")
        golden_ids = {case.id for case in queries}
        report = evaluate_answer_samples(queries, [s for s in samples if s.query_id in golden_ids], documents)
        self.assertGreater(report["summary"]["mean_lexical_groundedness"], 0.0)


if __name__ == "__main__":
    unittest.main()
