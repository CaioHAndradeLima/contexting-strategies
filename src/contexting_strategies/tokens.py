import re

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.I)


def estimate_tokens(text: str) -> int:
    """Rough token count for budgeting (same tokenizer as the demo retriever)."""
    if not text:
        return 0
    return max(1, len(TOKEN_RE.findall(text.lower())))


def estimate_document_tokens(document) -> int:
    from .models import Document

    if not isinstance(document, Document):
        raise TypeError("document must be a Document")
    overhead = estimate_tokens(document.title) + 8
    return overhead + estimate_tokens(document.text)
