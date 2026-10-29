FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY config ./config

ARG INSTALL_LLM_JUDGE=0
RUN pip install -e . \
    && if [ "$INSTALL_LLM_JUDGE" = "1" ]; then pip install -e '.[llm-judge]'; fi

ENTRYPOINT ["context-lab"]
