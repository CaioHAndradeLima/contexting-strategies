import unittest
from pathlib import Path

from contexting_strategies.data import load_documents, load_queries, validate_dataset
from contexting_strategies.measurement_pipeline import run_measurement_pipeline, select_assemblers
from contexting_strategies.models import QueryCase
from contexting_strategies.pipeline_config import PipelineConfig
from contexting_strategies.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


class FakeContextJudge:
    def judge(self, case: QueryCase, assembled_prompt: str) -> dict:
        return {
            "evidence_sufficiency": 3,
            "citation_clarity": 4,
            "safety": 4,
            "rationale": "fixture",
            "request_count": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "estimated_cost_usd": 0.0,
        }


class FakeAnswerGenerator:
    def generate(self, case: QueryCase, assembled_prompt: str, *, should_abstain: bool) -> dict:
        if should_abstain:
            return {
                "answer": "Insufficient evidence.",
                "skipped_generation": True,
                "request_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        return {
            "answer": "INC-4821 involved a stuck billing worker.",
            "skipped_generation": False,
            "request_count": 1,
            "input_tokens": 20,
            "output_tokens": 10,
            "estimated_cost_usd": 0.0,
        }


class FakeAnswerJudge:
    def judge(self, case: QueryCase, answer: str, evidence) -> dict:
        return {
            "correctness": 3,
            "groundedness": 4,
            "assertiveness_calibration": 4,
            "abstained": "insufficient" in answer.lower(),
            "unsupported_claims": [],
            "rationale": "fixture",
            "abstention_correct": float(("insufficient" in answer.lower()) == (not case.answerable)),
            "request_count": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "estimated_cost_usd": 0.0,
        }


class MeasurementPipelineTests(unittest.TestCase):
    def test_offline_measure_summarizes_strategies(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        report = run_measurement_pipeline(
            queries,
            lambda query, limit: retriever.search(query, limit),
            PipelineConfig(live_llm=False, budget_tokens=200),
            assemblers=select_assemblers(["pipeline_acl_dedupe_score_diverse", "weighted_score"]),
        )
        self.assertIn("pipeline_acl_dedupe_score_diverse", report["summary"])
        self.assertEqual(len(report["rows"]), len(queries) * 2)

    def test_live_path_with_fakes(self) -> None:
        documents = load_documents(ROOT / "data/documents.json")
        queries = load_queries(ROOT / "data/golden.json")[:2]
        validate_dataset(documents, queries)
        retriever = BM25Retriever(documents)
        report = run_measurement_pipeline(
            queries,
            lambda query, limit: retriever.search(query, limit),
            PipelineConfig(live_llm=True, budget_tokens=200, max_llm_calls=50),
            assemblers=select_assemblers(["pipeline_acl_dedupe_score_diverse"]),
            context_judge=FakeContextJudge(),
            answer_generator=FakeAnswerGenerator(),
            answer_judge=FakeAnswerJudge(),
        )
        self.assertIn("judge_groundedness", report["summary"]["pipeline_acl_dedupe_score_diverse"])
        self.assertIn("llm_usage_total", report["experiment"])


if __name__ == "__main__":
    unittest.main()
