from collections.abc import Sequence

from .models import Hit
from .retrieval import tokenize


class HeuristicReranker:
    """Lexical/title reranker aligned with reranking-strategies for bridge demos."""

    name = "heuristic"

    def rerank(self, query: str, candidates: Sequence[Hit], limit: int = 10) -> list[Hit]:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if not candidates or limit == 0:
            return []
        query_terms = set(tokenize(query))
        rescored = []
        for hit in candidates:
            title_terms = set(tokenize(hit.document.title))
            body_terms = set(tokenize(hit.document.text))
            title_match = len(query_terms & title_terms) / max(1, len(query_terms))
            body_match = len(query_terms & body_terms) / max(1, len(query_terms))
            phrase = float(query.lower() in hit.document.searchable_text.lower())
            score = 0.55 * title_match + 0.35 * body_match + 0.10 * phrase
            rescored.append(Hit(hit.document, score, self.name))
        return sorted(rescored, key=lambda hit: (-hit.score, hit.document.id))[:limit]
