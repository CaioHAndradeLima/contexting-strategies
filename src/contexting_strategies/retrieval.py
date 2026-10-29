import math
import re
from collections import Counter
from collections.abc import Iterable

from .models import Document, Hit

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Dependency-free BM25 for the demo corpus."""

    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        document_ids = [doc.id for doc in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Document IDs must be unique for BM25 retrieval")
        self.k1, self.b = k1, b
        self.tokens = [tokenize(doc.searchable_text) for doc in self.documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequency.update(frequencies.keys())

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        terms, count = set(tokenize(query)), len(self.documents)
        scored = []
        for doc, frequencies, length in zip(self.documents, self.term_frequencies, self.lengths):
            score = 0.0
            for term in terms:
                tf = frequencies.get(term, 0)
                if tf:
                    df = self.document_frequency[term]
                    idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                    norm = tf + self.k1 * (1 - self.b + self.b * length / max(1, self.average_length))
                    score += idf * tf * (self.k1 + 1) / norm
            scored.append(Hit(doc, score, "bm25"))
        scored.sort(key=lambda hit: (-hit.score, hit.document.id))
        return [hit for hit in scored[:limit] if hit.score > 0]
