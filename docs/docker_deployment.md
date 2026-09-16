# Docker deployment

## What is packaged

The image installs Python dependencies and downloads the configured embedding
model. Docker Compose starts AgentFlow together with PostgreSQL and pgvector.
Documents are indexed at runtime rather than copied into the image.

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
`POST /retrieval/index` endpoint to add text documents.

## LLM configuration

The API works in deterministic offline-writer mode without a key. For a real
OpenAI-compatible model, set variables in the shell before starting Compose:

```bash
export AGENTFLOW_LLM_API_KEY=your-key
export AGENTFLOW_LLM_BASE_URL=https://api.deepseek.com
export AGENTFLOW_LLM_MODEL=deepseek-chat
docker compose up -d
```

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
Set `AGENTFLOW_RETRIEVAL_BACKEND=local` to use the JSONL and NumPy indexes used
by the published benchmark.

The V2 serving snapshot path requires a mounted V2 index and a matching V2
embedding model. The default Compose image only prepares the V1 model and
starts with the pgvector backend; it is not a ready-to-run V2 deployment.

Document indexing and deletion are disabled unless
`AGENTFLOW_ADMIN_API_KEY` is configured. Clients must send the same value in
the `X-AgentFlow-Admin-Key` header.

## Verification boundary

The Compose deployment was smoke-tested on Docker Desktop 4.91.0 with Docker
Engine 29.8.0. The check covered image construction, non-root model loading,
API and PostgreSQL health checks, a real pgvector-backed retrieval request,
pgvector extension initialization, and restart persistence. Load, backup and
restore, multi-host deployment, and security testing are not covered. These
checks are not currently executed by an automated continuous-integration
workflow, so they should be repeated in each target environment.
