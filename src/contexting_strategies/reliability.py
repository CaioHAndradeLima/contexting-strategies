import re

INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
)


def sanitize_chunk_text(text: str) -> tuple[str, bool]:
    """Treat retrieved text as untrusted data; flag likely prompt-injection phrases."""
    flagged = any(pattern.search(text) for pattern in INJECTION_PATTERNS)
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered-untrusted-instruction]", cleaned)
    return cleaned, flagged


def any_injection_in_chunks(chunks) -> bool:
    for chunk in chunks:
        _, flagged = sanitize_chunk_text(chunk.document.text)
        if flagged:
            return True
    return False
