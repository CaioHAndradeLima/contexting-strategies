import unittest
from pathlib import Path

from contexting_strategies.data import load_documents, load_queries, validate_dataset
from contexting_strategies.models import ScoreWeights
from contexting_strategies.retrieval import BM25Retriever
from contexting_strategies.workflow import run_context_workflow

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_workflow_runs_all_stages(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        trace = run_context_workflow(
            queries[0],
            lambda query, limit: retriever.search(query, limit),
            weights=ScoreWeights(),
        )
        self.assertIn("retrieve", trace.stages_ms)
        self.assertIn("assemble", trace.stages_ms)
        self.assertTrue(trace.assembled_ids)


if __name__ == "__main__":
    unittest.main()
