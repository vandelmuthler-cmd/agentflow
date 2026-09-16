from __future__ import annotations

import json
import secrets
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Path
from fastapi.responses import StreamingResponse

from agentflow.agents.state import AgentState
from agentflow.agents.langgraph_workflow import LangGraphResearchWorkflow
from agentflow.config import (
    ADMIN_API_KEY,
    CHECKPOINT_DB_PATH,
    RAW_DIR,
    RETRIEVAL_BACKEND,
    TRACE_DIR,
    TOOL_CALLING_ENABLED,
)
from agentflow.retrieval.management import RetrievalIndexManager
from agentflow.retrieval.serving_index import IndexConflictError
from agentflow.retrieval.index import build_retriever
from agentflow.schemas import (
    DocumentIndexRequest,
    DocumentMutationResponse,
    ResearchRequest,
    ResearchReport,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    TraceSummary,
)
from agentflow.storage.traces import SQLiteTraceStore
from agentflow.storage.checkpoints import SQLiteCheckpointBackend
from agentflow.tools.document_search import DocumentSearchTool
from agentflow.tools.document_context import DocumentContextTool
from agentflow.tools.registry import ToolRegistry


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    close_workflow()


app = FastAPI(title="AgentFlow", version="0.1.0", lifespan=lifespan)


@lru_cache(maxsize=1)
def create_retriever():
    return build_retriever(RAW_DIR)


@lru_cache(maxsize=1)
def create_workflow() -> LangGraphResearchWorkflow:
    registry = ToolRegistry()
    retriever = create_retriever()
    registry.register(DocumentSearchTool(retriever))
    registry.register(DocumentContextTool(retriever.keyword_retriever.documents))
    checkpoint_backend = SQLiteCheckpointBackend(CHECKPOINT_DB_PATH)
    trace_store = SQLiteTraceStore(CHECKPOINT_DB_PATH, mirror_jsonl_dir=TRACE_DIR)
    return LangGraphResearchWorkflow(
        registry,
        trace_store,
        checkpoint_backend=checkpoint_backend,
        enable_tool_calling=TOOL_CALLING_ENABLED,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "retrieval_backend": RETRIEVAL_BACKEND}


def require_admin_key(
    x_agentflow_admin_key: str | None = Header(default=None),
) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="document management is disabled")
    if not x_agentflow_admin_key or not secrets.compare_digest(
        x_agentflow_admin_key, ADMIN_API_KEY
    ):
        raise HTTPException(status_code=401, detail="invalid admin API key")


@app.post("/retrieval/search", response_model=RetrievalSearchResponse)
def retrieval_search(request: RetrievalSearchRequest) -> RetrievalSearchResponse:
    started = time.perf_counter()
    try:
        evidence = create_retriever().search(request.query, top_k=request.top_k)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return RetrievalSearchResponse(
        query=request.query,
        backend=RETRIEVAL_BACKEND,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        evidence=evidence,
    )


@app.post(
    "/retrieval/index",
    response_model=DocumentMutationResponse,
    dependencies=[Depends(require_admin_key)],
)
def index_document(request: DocumentIndexRequest) -> DocumentMutationResponse:
    try:
        count = RetrievalIndexManager().index_text(
            request.document_id,
            request.source,
            request.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            language=request.language,
        )
    except IndexConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    refresh_retrieval_runtime()
    return DocumentMutationResponse(
        document_id=request.document_id,
        backend=RETRIEVAL_BACKEND,
        affected_chunks=count,
    )


@app.delete(
    "/retrieval/documents/{document_id}",
    response_model=DocumentMutationResponse,
    dependencies=[Depends(require_admin_key)],
)
def delete_document(
    document_id: str = Path(pattern=r"^[A-Za-z0-9._-]{1,128}$"),
) -> DocumentMutationResponse:
    try:
        count = RetrievalIndexManager().delete_document(document_id)
    except IndexConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    refresh_retrieval_runtime()
    return DocumentMutationResponse(
        document_id=document_id,
        backend=RETRIEVAL_BACKEND,
        affected_chunks=count,
    )


@app.post("/research", response_model=ResearchReport)
def research(request: ResearchRequest) -> ResearchReport:
    state = AgentState(run_id=str(uuid.uuid4()), question=request.question, top_k=request.top_k)
    return create_workflow().run(state)


@app.post("/research/stream")
def research_stream(request: ResearchRequest) -> StreamingResponse:
    state = AgentState(run_id=str(uuid.uuid4()), question=request.question, top_k=request.top_k)
    workflow = create_workflow()

    def events() -> Iterator[str]:
        for event in workflow.stream(state):
            payload = event.model_dump(mode="json")
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/runs/{run_id}/trace")
def get_trace(run_id: str) -> list[dict]:
    return create_workflow().trace_store.read(run_id)


@app.get("/runs/{run_id}/metrics", response_model=TraceSummary)
def get_trace_metrics(run_id: str) -> TraceSummary:
    summary = create_workflow().trace_store.summarize(run_id)
    if summary.event_count == 0:
        raise HTTPException(status_code=404, detail="run trace not found")
    return summary


@app.get("/runs/{run_id}", response_model=ResearchReport)
def get_run(run_id: str) -> ResearchReport:
    report = create_workflow().get_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="run not found")
    return report


@app.post("/runs/{run_id}/resume", response_model=ResearchReport)
def resume_run(run_id: str) -> ResearchReport:
    report = create_workflow().resume(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="run not found")
    return report


def close_workflow() -> None:
    if create_workflow.cache_info().currsize:
        create_workflow().close()
        create_workflow.cache_clear()


def refresh_retrieval_runtime() -> None:
    close_workflow()
    create_retriever.cache_clear()
