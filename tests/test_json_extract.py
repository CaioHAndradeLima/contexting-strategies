import unittest

from contexting_strategies.json_extract import extract_json_object


class JsonExtractTests(unittest.TestCase):
    def test_raw_json(self) -> None:
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced_json(self) -> None:
        text = 'Here is the score:\n```json\n{"evidence_sufficiency": 3}\n```'
        self.assertEqual(extract_json_object(text)["evidence_sufficiency"], 3)

    def test_embedded_object(self) -> None:
        text = 'Score: {"correctness": 4, "groundedness": 3} thanks'
        parsed = extract_json_object(text)
        self.assertEqual(parsed["correctness"], 4)


if __name__ == "__main__":
    unittest.main()
