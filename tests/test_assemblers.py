import unittest

from contexting_strategies.assemblers import NaiveRetrievalOrderAssembler, PipelineAssembler
from contexting_strategies.models import Document, Hit, QueryCase, ScoreWeights


class AssemblerTests(unittest.TestCase):
    def test_pipeline_drops_unauthorized_tenant(self) -> None:
        case = QueryCase(
            id="q-007",
            query="database credentials",
            relevance={"doc-011": 3},
            allowed_tenants=("billing",),
        )
        hits = [
            Hit(
                Document("doc-011", "Secret", "DATABASE_URL secret", {"tenant": "platform"}),
                1.0,
                "bm25",
            )
        ]
        context = PipelineAssembler().assemble(case, hits, budget_tokens=200, weights=ScoreWeights())
        self.assertEqual(context.chunks, ())
        self.assertEqual(context.dropped_unauthorized, 1)

    def test_naive_can_leak_unauthorized_chunk(self) -> None:
        case = QueryCase(
            id="q-007",
            query="database credentials",
            relevance={"doc-011": 3},
            allowed_tenants=("billing",),
        )
        hits = [
            Hit(
                Document("doc-011", "Secret", "DATABASE_URL secret", {"tenant": "platform"}),
                1.0,
                "bm25",
            )
        ]
        context = NaiveRetrievalOrderAssembler().assemble(case, hits, budget_tokens=200, weights=ScoreWeights())
        self.assertEqual(len(context.chunks), 1)

    def test_pipeline_deduplicates_near_duplicate_chunks(self) -> None:
        case = QueryCase(id="q-001", query="invoice retry", relevance={"doc-002": 3})
        text = "Invoice retry messages are published to the billing queue."
        hits = [
            Hit(Document("doc-002", "A", text, {"tenant": "billing", "authority_tier": "official"}), 0.9, "bm25"),
            Hit(Document("doc-002b", "B", text, {"tenant": "billing", "authority_tier": "draft"}), 0.95, "bm25"),
        ]
        context = PipelineAssembler().assemble(case, hits, budget_tokens=400, weights=ScoreWeights())
        self.assertEqual(len(context.chunks), 1)
        self.assertGreaterEqual(context.dropped_duplicate, 1)


if __name__ == "__main__":
    unittest.main()
