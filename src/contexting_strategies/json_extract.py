import json
import re


def extract_json_object(text: str) -> dict:
    """Parse the first JSON object from model output (raw JSON or fenced block)."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model output")
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError("no JSON object found in model output")
