import unittest
from pathlib import Path

from contexting_strategies.measure_profiles import evaluate_gates, load_measure_profiles
from contexting_strategies.pipeline_evaluation import run_contexting_pipeline_eval

ROOT = Path(__file__).resolve().parents[1]


class PipelineEvaluationTests(unittest.TestCase):
    def test_ci_contexting_eval_passes(self) -> None:
        profiles = load_measure_profiles(ROOT / "config/measure_profiles.json")
        profile = profiles["ci-contexting-eval"]
        report = run_contexting_pipeline_eval(ROOT, profile)
        self.assertEqual(report["eval_type"], "contexting_pipeline")
        self.assertIn("scorecard", report)
        self.assertIn("scenarios", report)
        gate = evaluate_gates(report, profile)
        self.assertTrue(gate["passed"], gate["failures"])

    def test_scenario_gates_require_pipeline_report(self) -> None:
        profiles = load_measure_profiles(ROOT / "config/measure_profiles.json")
        profile = profiles["ci-contexting-eval"]
        bare = {"summary": profile.gates and {"pipeline_acl_dedupe_score_diverse": {"evidence_coverage": 1.0, "unauthorized_leak": 0.0}}}
        # fill required assembly metrics for strategy gates that are set
        bare["summary"] = {
            "pipeline_acl_dedupe_score_diverse": {
                "evidence_coverage": 1.0,
                "unauthorized_leak": 0.0,
                "redundancy_rate": 0.1,
                "citation_traceability": 1.0,
                "assembly_ms_p95": 1.0,
                "abstain_correct": 1.0,
            }
        }
        result = evaluate_gates(bare, profile)
        self.assertFalse(result["passed"])
        self.assertTrue(any("scenarios.abstention" in f for f in result["failures"]))


if __name__ == "__main__":
    unittest.main()
