# Retrieval Service

AgentFlow exposes the retrieval subsystem independently from the research workflow. The same retriever is used by the HTTP search endpoint and the LangGraph `document_search` tool. The optional `document_context` tool reads neighboring chunks from the active document index; see [bounded native tool calling](tool_calling.md).

## Backends

### Local

The local backend stores chunk records in JSONL and normalized embeddings in a NumPy array. It is used to reproduce the published frozen benchmark.

```env
AGENTFLOW_RETRIEVAL_BACKEND=local
```

### PostgreSQL and pgvector

The pgvector backend stores document chunks, source metadata, and normalized BGE-M3 embeddings in PostgreSQL. It independently retrieves BM25 and vector candidates, fuses their ranks with Reciprocal Rank Fusion, and reranks the top candidates with a Cross-Encoder.

```env
AGENTFLOW_RETRIEVAL_BACKEND=pgvector
AGENTFLOW_DATABASE_URL=postgresql://agentflow:agentflow@localhost:5432/agentflow
AGENTFLOW_ADMIN_API_KEY=replace_with_a_random_admin_key
```

Docker Compose starts PostgreSQL with the pgvector extension and selects this backend by default.

### Local benchmark snapshots

The 16-document frozen benchmark index is immutable. With `AGENTFLOW_CORPUS_VERSION=v2`, text documents added through the management API are indexed into a separate serving snapshot under `data/index/v2/_serving/<strategy>/<model>/`. The service reads the published snapshot if present, otherwise the benchmark index. A restart reads the same published pointer.

```env
AGENTFLOW_CORPUS_VERSION=v2
AGENTFLOW_RETRIEVAL_BACKEND=local
AGENTFLOW_V2_CHUNKING_STRATEGY=section_aware
AGENTFLOW_V2_EMBED_MODEL=bge-m3
AGENTFLOW_V2_RETRIEVAL_METHOD=vector
AGENTFLOW_ADMIN_API_KEY=replace_with_a_random_admin_key
```

An update encodes only the changed document, combines it with the unchanged vectors, writes a new document/vector snapshot, validates alignment and hashes, then atomically replaces the active pointer. A failed build leaves the previous pointer intact. Mutations of the original benchmark documents return HTTP 409. The selected chunking and embedding configuration has its own serving pointer. Text requests may specify `language` (`en` or `zh`); when omitted, the language is inferred from the presence of Chinese characters. Token-based chunking uses 300-token chunks with 50-token overlap; `chunk_size` and `overlap` apply only to `fixed_char`.

## Index a text document

Document mutation endpoints are disabled when `AGENTFLOW_ADMIN_API_KEY` is empty. Send the configured value through `X-AgentFlow-Admin-Key`.

```bash
curl -X POST http://127.0.0.1:8000/retrieval/index \
  -H "Content-Type: application/json" \
  -H "X-AgentFlow-Admin-Key: your-admin-key" \
  -d '{"document_id":"operations-manual","source":"operations.md","text":"Document content..."}'
```

Re-indexing the same `document_id` replaces its existing chunks. The local benchmark backend publishes a validated snapshot atomically. PostgreSQL performs replacement in one transaction.

## Search

```bash
curl -X POST http://127.0.0.1:8000/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How are failed tasks retried?","top_k":5}'
```

The response includes the active backend, request latency, ranked evidence, source names, and chunk identifiers.

## Delete a document

```bash
curl -X DELETE http://127.0.0.1:8000/retrieval/documents/operations-manual \
  -H "X-AgentFlow-Admin-Key: your-admin-key"
```

## Scope

Local snapshots protect readers from partial index publication and serialize writers through a SQLite lock. The Compose deployment runs one API process; a document mutation refreshes that process's retriever. The service does not implement user accounts, role-based access control, background ingestion workers, or object storage.
