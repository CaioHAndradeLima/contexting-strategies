import unittest
from pathlib import Path

from contexting_strategies.data import load_documents, load_queries, validate_dataset
from contexting_strategies.measure_profiles import evaluate_gates, load_measure_profiles
from contexting_strategies.measurement_pipeline import run_measurement_pipeline, select_assemblers
from contexting_strategies.pipeline_config import PipelineConfig
from contexting_strategies.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


class MeasureProfileTests(unittest.TestCase):
    def test_ci_profile_passes_offline_report(self) -> None:
        profiles = load_measure_profiles(ROOT / "config/measure_profiles.json")
        profile = profiles["ci-all-strategies"]
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        report = run_measurement_pipeline(
            queries,
            lambda query, limit: retriever.search(query, limit),
            PipelineConfig(live_llm=False, budget_tokens=profile.budget_tokens, candidate_k=profile.candidate_k),
            assemblers=select_assemblers(list(profile.strategies)),
        )
        result = evaluate_gates(report, profile)
        self.assertTrue(result["passed"], result["failures"])

    def test_gate_fails_on_leak(self) -> None:
        profiles = load_measure_profiles(ROOT / "config/measure_profiles.json")
        profile = profiles["ci-pipeline-only"]
        report = {"summary": {"pipeline_acl_dedupe_score_diverse": {"evidence_coverage": 1.0, "unauthorized_leak": 0.5}}}
        result = evaluate_gates(report, profile)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
