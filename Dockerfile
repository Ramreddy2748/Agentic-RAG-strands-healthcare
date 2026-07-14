FROM python:3.12-slim

ARG INSTALL_STRANDS=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    RAG_INDEX_DIR=/app/.rag_index \
    PRELOAD_MODELS=true \
    ENABLE_VERIFICATION=false \
    MAX_CONCURRENT_REQUESTS=1 \
    LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY rag_chatbot ./rag_chatbot
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3" \
    && if [ "$INSTALL_STRANDS" = "true" ]; then \
        python -m pip install ".[strands,mongodb]"; \
    else \
        python -m pip install ".[mongodb]"; \
    fi

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.rag_index "${HF_HOME}" \
    && chown -R appuser:appuser /app "${HF_HOME}"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "rag_chatbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
