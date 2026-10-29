from enum import Enum

from .models import ConversationTurn
from .tokens import estimate_tokens


class HistoryStrategy(str, Enum):
    FULL = "full_history"
    LAST_N = "last_n"
    SUMMARY_PLUS_RECENT = "summary_plus_recent"


def render_turn(turn: ConversationTurn) -> str:
    role = turn.role.upper()
    return f"{role}: {turn.content}"


def select_history(
    turns: tuple[ConversationTurn, ...],
    strategy: HistoryStrategy,
    budget_tokens: int,
    *,
    last_n: int = 4,
) -> tuple[str, int]:
    if budget_tokens <= 0 or not turns:
        return "", 0
    if strategy is HistoryStrategy.FULL:
        return _pack_turns(turns, budget_tokens)
    if strategy is HistoryStrategy.LAST_N:
        recent = turns[-max(1, last_n) :]
        return _pack_turns(recent, budget_tokens)
    recent = turns[-max(1, last_n) :]
    older = turns[: -len(recent)] if len(turns) > len(recent) else ()
    summary_lines = [f"- {turn.role} mentioned: {_one_line(turn.content)}" for turn in older]
    summary = "Conversation summary:\n" + "\n".join(summary_lines) if summary_lines else ""
    summary_text, summary_tokens = _truncate_to_budget(summary, budget_tokens // 2 if older else budget_tokens)
    remaining = max(0, budget_tokens - summary_tokens)
    recent_text, recent_tokens = _pack_turns(recent, remaining)
    if summary_text and recent_text:
        combined = summary_text + "\n\nRecent messages:\n" + recent_text
    else:
        combined = summary_text or recent_text
    return combined, estimate_tokens(combined)


def _one_line(text: str, limit: int = 80) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def _pack_turns(turns: tuple[ConversationTurn, ...], budget_tokens: int) -> tuple[str, int]:
    blocks: list[str] = []
    used = 0
    for turn in turns:
        block = render_turn(turn)
        tokens = estimate_tokens(block)
        if used + tokens > budget_tokens and blocks:
            break
        if tokens > budget_tokens and not blocks:
            truncated = _one_line(block, max(20, budget_tokens * 4))
            blocks.append(truncated)
            used = estimate_tokens(truncated)
            break
        blocks.append(block)
        used += tokens
    rendered = "\n".join(blocks)
    return rendered, estimate_tokens(rendered)


def _truncate_to_budget(text: str, budget_tokens: int) -> tuple[str, int]:
    if not text:
        return "", 0
    if estimate_tokens(text) <= budget_tokens:
        return text, estimate_tokens(text)
    words = text.split()
    lo, hi = 0, len(words)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[:mid])
        if estimate_tokens(candidate) <= budget_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best, estimate_tokens(best)
