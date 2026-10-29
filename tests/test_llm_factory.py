import os
import unittest
from unittest.mock import patch

from contexting_strategies.llm_factory import resolve_llm_provider


class LlmFactoryTests(unittest.TestCase):
    def test_explicit_provider(self) -> None:
        self.assertEqual(resolve_llm_provider("cursor"), "cursor")
        self.assertEqual(resolve_llm_provider("openai"), "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_openai_without_keys(self) -> None:
        self.assertEqual(resolve_llm_provider(None), "openai")

    @patch.dict(os.environ, {"CURSOR_API_KEY": "cursor_test"}, clear=True)
    def test_auto_prefers_cursor_when_key_present(self) -> None:
        self.assertEqual(resolve_llm_provider(None), "cursor")

    @patch.dict(os.environ, {"CONTEXT_LAB_LLM_PROVIDER": "openai", "CURSOR_API_KEY": "x"}, clear=True)
    def test_env_overrides_auto(self) -> None:
        self.assertEqual(resolve_llm_provider(None), "openai")


if __name__ == "__main__":
    unittest.main()
