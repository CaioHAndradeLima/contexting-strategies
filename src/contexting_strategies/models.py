from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float
    source: str


DEFAULT_TENANTS = ("billing", "platform", "catalog", "communications", "payments", "search")


@dataclass(frozen=True)
class QueryCase:
    id: str
    query: str
    relevance: dict[str, int]
    tags: tuple[str, ...] = ()
    answerable: bool = True
    reference_answer: str | None = None
    allowed_tenants: tuple[str, ...] = DEFAULT_TENANTS


@dataclass(frozen=True)
class HopSpec:
    id: str
    label: str
    query: str
    relevance: dict[str, int]


@dataclass(frozen=True)
class MultiHopCase:
    id: str
    query: str
    hops: tuple[HopSpec, ...]
    tags: tuple[str, ...] = ()
    allowed_tenants: tuple[str, ...] = DEFAULT_TENANTS


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(frozen=True)
class ThreadScenario:
    id: str
    query: str
    resolved_query: str
    turns: tuple[ConversationTurn, ...]
    relevance: dict[str, int]
    history_strategy: str
    history_n: int = 4
    history_budget_tokens: int = 60
    evidence_budget_tokens: int = 140
    tags: tuple[str, ...] = ()
    answerable: bool = True


@dataclass(frozen=True)
class ContextChunk:
    document: Document
    retrieval_score: float
    assembly_score: float
    citation_id: str


@dataclass(frozen=True)
class AssembledContext:
    strategy: str
    chunks: tuple[ContextChunk, ...]
    rendered: str
    token_count: int
    dropped_unauthorized: int
    dropped_duplicate: int
    budget_tokens: int
    should_abstain: bool = False
    abstain_reason: str | None = None
    history_token_count: int = 0
    hop_coverage: float | None = None
    evidence_coverage: float | None = None


@dataclass(frozen=True)
class ScoreWeights:
    relevance: float = 0.45
    freshness: float = 0.20
    authority: float = 0.20
    task: float = 0.15

    def validate(self) -> None:
        total = self.relevance + self.freshness + self.authority + self.task
        if abs(total - 1.0) > 1e-6:
            raise ValueError("score weights must sum to 1.0")
