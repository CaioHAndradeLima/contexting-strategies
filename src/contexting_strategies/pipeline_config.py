import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    candidate_k: int = 10
    budget_tokens: int = 200
    apply_heuristic_rerank: bool = True
    live_llm: bool = False
    answer_model: str = "gpt-6-luna"
    judge_model: str = "gpt-6-luna"
    max_llm_calls: int = 100
    min_coverage: float = 0.34

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            live_llm=os.getenv("CONTEXT_LAB_LIVE_LLM", "").lower() in {"1", "true", "yes"},
            answer_model=os.getenv("CONTEXT_LAB_ANSWER_MODEL", "gpt-6-luna"),
            judge_model=os.getenv("CONTEXT_LAB_JUDGE_MODEL", "gpt-6-luna"),
            max_llm_calls=int(os.getenv("CONTEXT_LAB_MAX_LLM_CALLS", "100")),
        )
