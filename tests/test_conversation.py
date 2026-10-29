import unittest

from contexting_strategies.conversation import HistoryStrategy, select_history
from contexting_strategies.models import ConversationTurn


class ConversationTests(unittest.TestCase):
    def test_last_n_uses_fewer_tokens_than_full_history(self) -> None:
        turns = tuple(
            ConversationTurn(role="user", content=f"message number {index} about catalog metadata and SKU rules")
            for index in range(12)
        )
        _, full_tokens = select_history(turns, HistoryStrategy.FULL, budget_tokens=80)
        _, last_tokens = select_history(turns, HistoryStrategy.LAST_N, budget_tokens=80, last_n=2)
        self.assertLessEqual(last_tokens, full_tokens)

    def test_summary_plus_recent_fits_budget(self) -> None:
        turns = tuple(
            ConversationTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i} " + "detail " * 10)
            for i in range(8)
        )
        text, tokens = select_history(
            turns,
            HistoryStrategy.SUMMARY_PLUS_RECENT,
            budget_tokens=60,
            last_n=2,
        )
        self.assertLessEqual(tokens, 60)
        self.assertIn("Recent messages", text)


if __name__ == "__main__":
    unittest.main()
