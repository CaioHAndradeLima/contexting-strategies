import os
from pathlib import Path

from .cursor_llm import (
    DEFAULT_CURSOR_MODEL,
    CursorAgentClient,
    CursorAnswerGenerator,
    CursorAnswerJudge,
    CursorContextJudge,
)
from .llm_judge import OpenAIAnswerGenerator, OpenAIAnswerJudge, OpenAIContextJudge
from .pipeline_config import PipelineConfig
from .pricing import CallBudget


def resolve_llm_provider(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        choice = explicit.strip().lower()
        if choice not in {"openai", "cursor"}:
            raise ValueError(f"unknown llm provider '{explicit}'; use openai or cursor")
        return choice
    env = os.getenv("CONTEXT_LAB_LLM_PROVIDER", "").strip().lower()
    if env in {"openai", "cursor"}:
        return env
    if os.getenv("CURSOR_API_KEY", "").strip():
        return "cursor"
    return "openai"


def _effective_models(config: PipelineConfig, provider: str) -> tuple[str, str]:
    answer = config.answer_model
    judge = config.judge_model
    if provider == "cursor":
        if not answer or answer.startswith("gpt-"):
            answer = os.getenv("CONTEXT_LAB_CURSOR_MODEL", DEFAULT_CURSOR_MODEL)
        if not judge or judge.startswith("gpt-"):
            judge = os.getenv("CONTEXT_LAB_CURSOR_MODEL", DEFAULT_CURSOR_MODEL)
    return answer, judge


def build_live_llm_stack(
    config: PipelineConfig,
    *,
    provider: str | None = None,
    cwd: Path | None = None,
) -> tuple[object, object, object, str]:
    """Return (context_judge, answer_generator, answer_judge, resolved_provider)."""
    resolved = resolve_llm_provider(provider)
    budget = CallBudget(config.max_llm_calls)
    answer_model, judge_model = _effective_models(config, resolved)
    workdir = cwd or Path.cwd()

    if resolved == "cursor":
        judge_client = CursorAgentClient(judge_model, cwd=workdir, call_budget=budget)
        gen_client = CursorAgentClient(answer_model, cwd=workdir, call_budget=budget)
        return (
            CursorContextJudge(judge_model, judge_client),
            CursorAnswerGenerator(answer_model, gen_client),
            CursorAnswerJudge(judge_model, judge_client),
            resolved,
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[llm-judge]'") from exc
    client = OpenAI()
    return (
        OpenAIContextJudge(judge_model, client, budget),
        OpenAIAnswerGenerator(answer_model, client, budget),
        OpenAIAnswerJudge(judge_model, client, budget),
        resolved,
    )
