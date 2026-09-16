# Docker deployment

## What is packaged

The image installs Python dependencies, BGE-M3, and the Cross-Encoder reranker.
Docker Compose starts AgentFlow together with PostgreSQL and pgvector. Documents
are indexed at runtime rather than copied into the image.

## Build and run

```bash
export AGENTFLOW_ADMIN_API_KEY=replace_with_a_random_value
docker compose up --build -d
curl http://localhost:8000/health
```

On Windows PowerShell, configure secrets in the untracked `.env` file and run:

```powershell
docker compose build
docker compose up -d
Invoke-RestMethod http://127.0.0.1:8000/health
```

The first build needs network access for Python packages and the embedding model.
Later image builds reuse Docker layers where available; database indexes remain
in the PostgreSQL volume.
The pgvector database is empty on first startup. Use the protected
`POST /retrieval/index` endpoint to add text documents. The default online path
uses section-aware chunking, independent BM25 and vector recall, Reciprocal Rank
Fusion, query expansion, and Cross-Encoder reranking.

Run a temporary end-to-end retrieval smoke test inside the container:

```powershell
docker compose exec -T agentflow python -B scripts/smoke_docker_retrieval.py
```

The script indexes two temporary documents, verifies that the relevant document
is ranked first, prints a compact result, and removes both documents in `finally`.

## LLM configuration

The API works in deterministic offline-writer mode without a key. For a real
OpenAI-compatible model, set variables in the shell before starting Compose:

```bash
export AGENTFLOW_LLM_API_KEY=your-key
export AGENTFLOW_LLM_BASE_URL=https://api.deepseek.com
export AGENTFLOW_LLM_MODEL=deepseek-chat
docker compose up -d
```

The Compose file also accepts the legacy `DEEPSEEK_API_KEY`,
`DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL` names used by earlier local setups.
The `AGENTFLOW_LLM_*` names are preferred for provider-neutral deployments.

Do not put a real key in `compose.yaml` or commit a `.env` file.

## Persistent files

Persistent volumes store:

- SQLite LangGraph checkpoints
- SQLite and JSONL trace events
- context-overflow artifacts
- PostgreSQL document chunks and embeddings

Remove the volume only when the saved runs are no longer needed.

## Daily lifecycle

```bash
docker compose ps
docker compose logs -f agentflow
docker compose restart
docker compose down
```

`docker compose down` removes containers and the Compose network but preserves
named volumes. `docker compose down -v` also deletes persisted database and run
state and should only be used for an intentional clean reset.

## Retrieval backend

Compose selects `AGENTFLOW_RETRIEVAL_BACKEND=pgvector` by default. The Agent and
the standalone `/retrieval/search` endpoint use the same retriever instance.
Set `AGENTFLOW_RETRIEVAL_BACKEND=local` only when reproducing the published
JSONL/NumPy benchmark. The Docker image is intended for dynamic user documents
and uses the same selected chunking, embedding, fusion, and reranking components.

An existing empty pgvector table created with another embedding dimension is
recreated automatically. A non-empty incompatible table is rejected so that
stored vectors are never silently interpreted with the wrong model; export or
delete those documents before rebuilding the index.

Document indexing and deletion are disabled unless
`AGENTFLOW_ADMIN_API_KEY` is configured. Clients must send the same value in
the `X-AgentFlow-Admin-Key` header.

## Verification boundary

The Compose deployment was validated on Docker Desktop 4.91.0 with Docker
Engine 29.8.0. The check covered image construction, non-root model loading,
protected HTTP indexing/search/deletion, PostgreSQL and pgvector health checks,
an 80-query frozen bilingual retrieval benchmark, SSE node events, and SQLite
run persistence across a container restart. The pgvector benchmark reproduced
the selected local metrics exactly: All-gold Hit@5 was 95.0% and MRR@10 was
0.914. CPU query-pipeline latency averaged 10.03 s (P95 13.46 s).

External HTTPS handshakes failed in the validation environment, so the Docker
run exercised the traced model-failure path instead of a successful DeepSeek
response. This is not evidence of online model success. See the
[dated validation report](../reports/docker_validation_2026-09-16.md) and the
[machine-readable pgvector report](../reports/v2/frozen/section_aware_bge-m3/pgvector-report.json).
Load, backup and restore, multi-host deployment, and security testing are not
covered. These checks are not currently executed by an automated
continuous-integration workflow, so they should be repeated in each target
environment.
