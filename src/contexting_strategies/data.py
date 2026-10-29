import json
from pathlib import Path

from .models import DEFAULT_TENANTS, Document, HopSpec, MultiHopCase, QueryCase, ThreadScenario, ConversationTurn


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
                allowed_tenants=tuple(allowed) if allowed else DEFAULT_TENANTS,
            )
        )
    return cases


def load_multihop_cases(path: Path) -> list[MultiHopCase]:
    payload = json.loads(path.read_text())
    cases = []
    for row in payload:
        hops = tuple(
            HopSpec(
                id=hop["id"],
                label=hop.get("label", hop["id"]),
                query=hop["query"],
                relevance={key: int(value) for key, value in hop.get("relevance", {}).items()},
            )
            for hop in row["hops"]
        )
        allowed = row.get("allowed_tenants")
        cases.append(
            MultiHopCase(
                id=row["id"],
                query=row["query"],
                hops=hops,
                tags=tuple(row.get("tags", [])),
                allowed_tenants=tuple(allowed) if allowed else DEFAULT_TENANTS,
            )
        )
    return cases


def load_thread_scenarios(path: Path) -> list[ThreadScenario]:
    payload = json.loads(path.read_text())
    scenarios = []
    for row in payload:
        turns = tuple(ConversationTurn(role=t["role"], content=t["content"]) for t in row.get("turns", []))
        scenarios.append(
            ThreadScenario(
                id=row["id"],
                query=row["query"],
                resolved_query=row["resolved_query"],
                turns=turns,
                relevance={key: int(value) for key, value in row.get("relevance", {}).items()},
                history_strategy=row["history_strategy"],
                history_n=int(row.get("history_n", 4)),
                history_budget_tokens=int(row.get("history_budget_tokens", 60)),
                evidence_budget_tokens=int(row.get("evidence_budget_tokens", 140)),
                tags=tuple(row.get("tags", [])),
                answerable=row.get("answerable", True),
            )
        )
    return scenarios


def validate_multihop_dataset(documents: list[Document], cases: list[MultiHopCase]) -> None:
    doc_ids = {doc.id for doc in documents}
    for case in cases:
        for hop in case.hops:
            unknown = set(hop.relevance.keys()) - doc_ids
            if unknown:
                raise ValueError(f"hop {case.id}/{hop.id} references unknown documents: {sorted(unknown)}")


def validate_dataset(documents: list[Document], queries: list[QueryCase]) -> None:
    doc_ids = {doc.id for doc in documents}
    if len(doc_ids) != len(documents):
        raise ValueError("duplicate document ids")
    for case in queries:
        unknown = set(case.relevance.keys()) - doc_ids
        if unknown:
            raise ValueError(f"query {case.id} references unknown documents: {sorted(unknown)}")
