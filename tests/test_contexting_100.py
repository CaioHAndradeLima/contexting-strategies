import unittest
from pathlib import Path

from contexting_strategies.analytical_sql import run_validated_select
from contexting_strategies.assemblers import PipelineAssembler
from contexting_strategies.compacting import compact_document_text
from contexting_strategies.context_pipeline import step3_merge_parent_child
from contexting_strategies.data import load_documents, load_structured_results, load_thread_scenarios
from contexting_strategies.models import Document, Hit, QueryCase, ScoreWeights
from contexting_strategies.query_rewrite import rewrite_follow_up_query

ROOT = Path(__file__).resolve().parents[1]


class Contexting100Tests(unittest.TestCase):
    def test_parent_child_merge(self) -> None:
        parent = Hit(
            Document("doc-006", "P", "parent text", {"tenant": "billing"}),
            0.8,
            "bm25",
        )
        child = Hit(
            Document("doc-006-detail", "C", "child text", {"tenant": "billing", "parent_id": "doc-006"}),
            0.7,
            "bm25",
        )
        merged, count = step3_merge_parent_child([parent, child])
        self.assertEqual(count, 1)
        self.assertEqual(len(merged), 1)
        self.assertIn("child text", merged[0].document.text)

    def test_tombstone_dropped_in_pipeline(self) -> None:
        case = QueryCase(id="t", query="retired faq", relevance={})
        hits = [
            Hit(
                Document("doc-tombstone", "T", "retired", {"tenant": "billing", "tombstone": True}),
                1.0,
                "bm25",
            )
        ]
        ctx = PipelineAssembler().assemble(case, hits, 100, ScoreWeights())
        self.assertEqual(ctx.chunks, ())

    def test_validated_sql(self) -> None:
        structured = load_structured_results(ROOT / "data/structured_results.json")["sql-checkout-conversion"]
        result = run_validated_select(
            "SELECT week_start, conversion_rate, wow_change FROM checkout_conversion_weekly "
            "WHERE week_start = '2026-08-18'",
            structured,
        )
        self.assertEqual(result["row_count"], 1)

    def test_query_rewrite_matches_fixture_intent(self) -> None:
        scenarios = load_thread_scenarios(ROOT / "data/threads.json")
        rewritten = rewrite_follow_up_query(scenarios[0].turns, scenarios[0].query)
        self.assertIn("INC-4821", rewritten)

    def test_compacting_reduces_tokens(self) -> None:
        text = " ".join(
            [
                "Sentence one about billing.",
                "Sentence two with INC-4821 and checkout impact.",
                "Sentence three with retry policy details.",
                "Sentence four with notification service behavior.",
            ]
        )
        out = compact_document_text(text, 8, mode="summarize")
        self.assertLessEqual(out["compact_tokens"], out["original_tokens"])
        self.assertIn("INC-4821", out["text"])


if __name__ == "__main__":
    unittest.main()
