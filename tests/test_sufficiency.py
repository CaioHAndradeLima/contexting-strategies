import unittest

from contexting_strategies.models import AssembledContext, QueryCase
from contexting_strategies.sufficiency import apply_sufficiency_gate


class SufficiencyTests(unittest.TestCase):
    def test_unanswerable_query_abstains(self) -> None:
        case = QueryCase(id="q", query="unknown", relevance={}, answerable=False)
        context = AssembledContext(
            strategy="test",
            chunks=(),
            rendered="Evidence:",
            token_count=5,
            dropped_unauthorized=0,
            dropped_duplicate=0,
            budget_tokens=100,
        )
        gated = apply_sufficiency_gate(context, case)
        self.assertTrue(gated.should_abstain)

    def test_low_coverage_answerable_abstains(self) -> None:
        case = QueryCase(id="q", query="x", relevance={"doc-001": 3}, answerable=True)
        context = AssembledContext(
            strategy="test",
            chunks=(),
            rendered="Evidence:",
            token_count=5,
            dropped_unauthorized=0,
            dropped_duplicate=0,
            budget_tokens=100,
        )
        gated = apply_sufficiency_gate(context, case, min_coverage=0.5)
        self.assertTrue(gated.should_abstain)


if __name__ == "__main__":
    unittest.main()
