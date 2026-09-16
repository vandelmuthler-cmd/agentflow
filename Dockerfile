FROM python:3.11-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENTFLOW_EMBED_MODEL_PATH=/opt/agentflow-model \
    AGENTFLOW_RERANKER_MODEL_PATH=/opt/agentflow-reranker \
    AGENTFLOW_CHECKPOINT_DB_PATH=/app/runtime/agentflow.sqlite3 \
    AGENTFLOW_TRACE_DIR=/app/runtime/traces \
    AGENTFLOW_RUN_WORKSPACE_DIR=/app/runtime/runs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=8

ARG TORCH_VERSION=2.7.1
COPY requirements.lock.txt .
RUN pip install "torch==${TORCH_VERSION}+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.lock.txt

FROM runtime-base AS model-builder

ENV HF_HUB_DISABLE_XET=1

ARG EMBED_MODEL=BAAI/bge-m3
ARG RERANKER_MODEL=BAAI/bge-reranker-base

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}').save('/opt/agentflow-model')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL}').save('/opt/agentflow-reranker')"

FROM runtime-base AS final

ENV PYTHONPATH=/app \
    AGENTFLOW_EMBED_MODEL_PATH=/opt/agentflow-model \
    AGENTFLOW_RERANKER_MODEL_PATH=/opt/agentflow-reranker \
    AGENTFLOW_CHECKPOINT_DB_PATH=/app/runtime/agentflow.sqlite3 \
    AGENTFLOW_TRACE_DIR=/app/runtime/traces \
    AGENTFLOW_RUN_WORKSPACE_DIR=/app/runtime/runs

COPY --from=model-builder --chmod=0755 /opt/agentflow-model /opt/agentflow-model
COPY --from=model-builder --chmod=0755 /opt/agentflow-reranker /opt/agentflow-reranker

COPY . .
RUN useradd --create-home --uid 10001 agentflow \
    && mkdir -p /app/runtime /app/data/index \
    && chown -R agentflow:agentflow /app/runtime /app/data/index

USER agentflow
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "agentflow.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
