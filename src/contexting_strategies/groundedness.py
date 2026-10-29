import re
from statistics import mean

from .models import AnswerSample, Document, QueryCase

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def lexical_groundedness(answer: str, evidence: list[Document]) -> float:
    """Fraction of answer sentences with some token overlap with evidence (offline baseline)."""
    if not answer.strip():
        return 0.0
    corpus = " ".join(doc.searchable_text.lower() for doc in evidence)
    corpus_terms = set(re.findall(r"[a-z0-9]+", corpus))
    scores = []
    for sentence in _sentences(answer):
        terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        if not terms:
            continue
        overlap = len(terms & corpus_terms) / len(terms)
        scores.append(1.0 if overlap >= 0.45 else 0.0)
    return mean(scores) if scores else 0.0


def evaluate_answer_samples(
    queries: list[QueryCase],
    samples: list[AnswerSample],
    documents: list[Document],
) -> dict:
    by_id = {doc.id: doc for doc in documents}
    query_by_id = {case.id: case for case in queries}
    rows = []
    groundedness_scores = []
    correctness_proxy = []
    for sample in samples:
        case = query_by_id.get(sample.query_id)
        if case is None:
            raise ValueError(f"unknown query_id {sample.query_id}")
        evidence = [by_id[doc_id] for doc_id in sample.evidence_ids if doc_id in by_id]
        grounded = lexical_groundedness(sample.answer, evidence)
        groundedness_scores.append(grounded)
        if not case.answerable:
            abstained = "insufficient" in sample.answer.lower() or "cannot" in sample.answer.lower()
            correctness_proxy.append(1.0 if abstained else 0.0)
        elif case.reference_answer:
            ref_terms = set(re.findall(r"[a-z0-9]+", case.reference_answer.lower()))
            ans_terms = set(re.findall(r"[a-z0-9]+", sample.answer.lower()))
            overlap = len(ref_terms & ans_terms) / max(1, len(ref_terms))
            correctness_proxy.append(1.0 if overlap >= 0.35 else 0.0)
        else:
            correctness_proxy.append(0.0)
        rows.append(
            {
                "query_id": sample.query_id,
                "lexical_groundedness": grounded,
                "correctness_proxy": correctness_proxy[-1],
                "answerable": case.answerable,
            }
        )
    return {
        "summary": {
            "mean_lexical_groundedness": mean(groundedness_scores) if groundedness_scores else 0.0,
            "mean_correctness_proxy": mean(correctness_proxy) if correctness_proxy else 0.0,
        },
        "rows": rows,
    }
