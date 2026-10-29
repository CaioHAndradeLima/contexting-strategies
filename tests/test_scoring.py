import unittest
from datetime import datetime, timezone

from contexting_strategies.models import Document, Hit, QueryCase, ScoreWeights
from contexting_strategies.scoring import combined_score, score_hits


class ScoringTests(unittest.TestCase):
    def test_draft_loses_to_official_when_other_signals_equal(self) -> None:
        case = QueryCase(id="q", query="invoice retry", relevance={})
        official = Hit(
            Document("a", "t", "invoice retry policy", {"authority_tier": "official", "updated_at": "2026-01-01T00:00:00Z"}),
            1.0,
            "bm25",
        )
        draft = Hit(
            Document("b", "t", "invoice retry policy", {"authority_tier": "draft", "updated_at": "2026-01-01T00:00:00Z"}),
            1.0,
            "bm25",
        )
        weights = ScoreWeights()
        ref = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.assertGreater(
            combined_score(official, case, 1.0, ref, weights),
            combined_score(draft, case, 1.0, ref, weights),
        )

    def test_entity_boost_for_incident_query(self) -> None:
        case = QueryCase(id="q", query="What happened in INC-4821?", relevance={"doc-006": 3})
        incident = Hit(
            Document("doc-006", "Incident", "INC-4821 stuck worker", {"authority_tier": "runbook", "updated_at": "2026-08-01T00:00:00Z"}),
            0.4,
            "bm25",
        )
        other = Hit(
            Document("doc-003", "Other", "notifications only", {"authority_tier": "official", "updated_at": "2026-08-01T00:00:00Z"}),
            1.0,
            "bm25",
        )
        ranked = score_hits([other, incident], case, ScoreWeights(relevance=0.25, freshness=0.15, authority=0.15, task=0.45))
        self.assertEqual(ranked[0][0].document.id, "doc-006")


if __name__ == "__main__":
    unittest.main()
