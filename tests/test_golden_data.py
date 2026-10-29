import unittest
from pathlib import Path

from contexting_strategies.data import load_documents, load_queries, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


class GoldenDataTests(unittest.TestCase):
    def test_golden_references_known_documents(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)


if __name__ == "__main__":
    unittest.main()
