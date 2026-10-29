#!/usr/bin/env bash
# Live measure using the local Cursor agent (CURSOR_API_KEY), not OpenAI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CURSOR_API_KEY:-}" ]]; then
  echo "Set CURSOR_API_KEY from https://cursor.com/dashboard/integrations"
  exit 1
fi

PROFILE="${1:-nightly-live}"
OUT="${2:-reports/measure-live-cursor.json}"
mkdir -p "$(dirname "$OUT")"

export CONTEXT_LAB_LLM_PROVIDER=cursor

python -m contexting_strategies.cli measure \
  --profile "$PROFILE" \
  --live-llm \
  --llm-provider cursor \
  --max-llm-calls "${CONTEXT_LAB_MAX_LLM_CALLS:-80}" \
  --output "$OUT"

python -m contexting_strategies.cli check-measure \
  --profile "$PROFILE" \
  --report "$OUT"

echo "Wrote $OUT"
