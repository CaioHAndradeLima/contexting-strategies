import json
from pathlib import Path

from .models import Document, QueryCase


def load_documents(path: Path) -> list[Document]:
    payload = json.loads(path.read_text())
    documents = []
    for row in payload:
        documents.append(
            Document(
                id=row["id"],
                title=row["title"],
                text=row["text"],
                metadata=row.get("metadata", {}),
            )
        )
    return documents


def load_queries(path: Path) -> list[QueryCase]:
    payload = json.loads(path.read_text())
    cases = []
    for row in payload:
        allowed = row.get("allowed_tenants")
        cases.append(
            QueryCase(
                id=row["id"],
                query=row["query"],
                relevance={key: int(value) for key, value in row.get("relevance", {}).items()},
                tags=tuple(row.get("tags", [])),
                answerable=row.get("answerable", True),
                reference_answer=row.get("reference_answer"),
                allowed_tenants=tuple(allowed) if allowed else QueryCase.allowed_tenants,
            )
        )
    return cases


def validate_dataset(documents: list[Document], queries: list[QueryCase]) -> None:
    doc_ids = {doc.id for doc in documents}
    if len(doc_ids) != len(documents):
        raise ValueError("duplicate document ids")
    for case in queries:
        unknown = set(case.relevance.keys()) - doc_ids
        if unknown:
            raise ValueError(f"query {case.id} references unknown documents: {sorted(unknown)}")
