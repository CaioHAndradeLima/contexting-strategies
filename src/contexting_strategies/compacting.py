import re

from .tokens import estimate_tokens

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def truncate_at_sentence_boundary(text: str, max_tokens: int) -> str:
    """Step 6 — truncate at meaningful boundaries (context-algorithm.md)."""
    if estimate_tokens(text) <= max_tokens:
        return text
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    kept: list[str] = []
    for sentence in sentences:
        candidate = " ".join(kept + [sentence])
        if estimate_tokens(candidate) > max_tokens and kept:
            break
        kept.append(sentence)
    if not kept and sentences:
        words = sentences[0].split()
        partial = []
        for word in words:
            if estimate_tokens(" ".join(partial + [word])) > max_tokens:
                break
            partial.append(word)
        return " ".join(partial)
    return " ".join(kept)


def extractive_summarize(text: str, max_tokens: int) -> str:
    """Selective compression preserving provenance-friendly sentences (not abstractive LLM)."""
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return text
    # Prefer sentences with identifiers or numbers, then fill by order.
    def priority(s: str) -> tuple[int, int]:
        has_entity = 1 if re.search(r"\bINC-\d+\b|[A-Z]{2,}-\d+", s) else 0
        has_number = 1 if re.search(r"\d", s) else 0
        return (has_entity + has_number, len(s))

    ranked = sorted(enumerate(sentences), key=lambda pair: (-priority(pair[1])[0], pair[0]))
    kept: list[str] = []
    for _, sentence in ranked:
        candidate = " ".join(kept + [sentence])
        if estimate_tokens(candidate) > max_tokens and kept:
            continue
        if estimate_tokens(candidate) <= max_tokens:
            kept.append(sentence)
    if not kept:
        return truncate_at_sentence_boundary(text, max_tokens)
    return " ".join(kept)


def compact_document_text(text: str, max_tokens: int, mode: str = "truncate") -> dict:
    truncated = truncate_at_sentence_boundary(text, max_tokens)
    summarized = extractive_summarize(text, max_tokens)
    chosen = summarized if mode == "summarize" else truncated
    return {
        "mode": mode,
        "original_tokens": estimate_tokens(text),
        "compact_tokens": estimate_tokens(chosen),
        "text": chosen,
    }
