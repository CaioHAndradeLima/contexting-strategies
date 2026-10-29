from .compacting import compact_document_text


def evaluate_compacting_demo() -> dict:
    sample = (
        "Incident INC-4821 blocked checkout retries. "
        "Engineering restarted the billing worker. "
        "Retry limit increased from two to three attempts. "
        "Customer notifications continued on a separate service."
    )
    budget = 25
    return {
        "guide": "context-algorithm.md step 6 + retrieval/compacting",
        "truncate": compact_document_text(sample, budget, mode="truncate"),
        "summarize": compact_document_text(sample, budget, mode="summarize"),
    }
