import json
import os
from collections.abc import Sequence
from pathlib import Path

from .json_extract import extract_json_object
from .llm_judge import ANSWER_RUBRIC, CONTEXT_RUBRIC
from .models import Document, QueryCase
from .pricing import CallBudget

DEFAULT_CURSOR_MODEL = "composer-2.5"


def _usage_stub() -> dict:
    return {
        "request_count": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "billing_note": "Cursor subscription; token cost not estimated in this lab",
    }


class CursorAgentClient:
    """One-shot local Cursor agent calls via cursor-sdk."""

    def __init__(
        self,
        model: str = DEFAULT_CURSOR_MODEL,
        *,
        api_key: str | None = None,
        cwd: str | Path | None = None,
        call_budget: CallBudget | None = None,
    ):
        self.model = model
        self.api_key = (api_key or os.getenv("CURSOR_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is required for --llm-provider cursor. "
                "Create one at https://cursor.com/dashboard/integrations"
            )
        self.cwd = str(cwd or Path.cwd())
        self.call_budget = call_budget

    def complete(self, instructions: str, user_input: str) -> str:
        if self.call_budget is not None:
            self.call_budget.consume()
        try:
            from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[llm-cursor]'") from exc

        prompt = f"{instructions}\n\n---\n\n{user_input}\n\nReply with a single JSON object only, no markdown."
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=self.api_key,
                model=self.model,
                local=LocalAgentOptions(cwd=self.cwd, setting_sources=[]),
            ),
        )
        if result.status != "finished":
            raise RuntimeError(f"Cursor agent run failed: status={result.status}")
        text = (result.result or "").strip()
        if not text:
            raise RuntimeError("Cursor agent returned empty text")
        return text


class CursorContextJudge:
    def __init__(self, model: str = DEFAULT_CURSOR_MODEL, client: CursorAgentClient | None = None):
        self.client = client or CursorAgentClient(model=model)

    def judge(self, case: QueryCase, assembled_prompt: str) -> dict:
        raw = self.client.complete(
            CONTEXT_RUBRIC,
            json.dumps({"query": case.query, "answerable": case.answerable, "context": assembled_prompt}),
        )
        result = extract_json_object(raw)
        result.update(_usage_stub())
        return result


class CursorAnswerJudge:
    def __init__(self, model: str = DEFAULT_CURSOR_MODEL, client: CursorAgentClient | None = None):
        self.client = client or CursorAgentClient(model=model)

    def judge(self, case: QueryCase, answer: str, evidence: Sequence[Document]) -> dict:
        if not answer.strip():
            raise ValueError("answer must be non-empty")
        raw = self.client.complete(
            ANSWER_RUBRIC,
            json.dumps(
                {
                    "query": case.query,
                    "answerable": case.answerable,
                    "reference_answer": case.reference_answer,
                    "answer": answer,
                    "evidence": [{"id": d.id, "title": d.title, "text": d.text} for d in evidence],
                }
            ),
        )
        result = extract_json_object(raw)
        result["abstention_correct"] = float(result.get("abstained") == (not case.answerable))
        result.update(_usage_stub())
        return result


class CursorAnswerGenerator:
    def __init__(self, model: str = DEFAULT_CURSOR_MODEL, client: CursorAgentClient | None = None):
        self.client = client or CursorAgentClient(model=model)

    def generate(self, case: QueryCase, assembled_prompt: str, *, should_abstain: bool) -> dict:
        if should_abstain:
            return {
                "answer": "I do not have sufficient evidence in the provided context to answer reliably.",
                "request_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
                "skipped_generation": True,
            }
        text = self.client.complete(
            (
                "Answer only from the supplied context. Cite evidence IDs when possible. "
                "If context is insufficient, say so explicitly. Reply with plain text only."
            ),
            assembled_prompt,
        )
        return {"answer": text.strip(), "skipped_generation": False, **_usage_stub()}
