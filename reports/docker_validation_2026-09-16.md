# Docker validation report

Environment: Docker Desktop 4.91.0, Docker Engine 29.8.0, CPU inference.

## HTTP document lifecycle

The public FastAPI surface was exercised from the host rather than from inside
the application container.

- An unauthenticated `POST /retrieval/index` request returned HTTP 401.
- An authenticated index request stored one temporary document in pgvector.
- `POST /retrieval/search` ranked that document first with a Cross-Encoder score
  of 0.9551.
- The authenticated delete endpoint removed the document.
- A second search returned no result, confirming cache refresh and deletion.

The temporary document was removed after the test.

## Frozen pgvector retrieval

The 16-document, 934-chunk section-aware BGE-M3 index was imported into the
Compose PostgreSQL/pgvector service. All 40 frozen answerable intents were run
in Chinese and English, for 80 query variants. The online service used the same
BM25/vector candidate generation, Reciprocal Rank Fusion, query expansion and
Cross-Encoder reranking as the selected local pipeline.

| Metric | Local index | Docker pgvector | Delta |
| --- | ---: | ---: | ---: |
| All-gold Hit@1 | 85.0% | 85.0% | 0.0 pp |
| All-gold Hit@3 | 95.0% | 95.0% | 0.0 pp |
| All-gold Hit@5 | 95.0% | 95.0% | 0.0 pp |
| MRR@10 | 0.914 | 0.914 | 0.000 |

Candidate all-gold recall at 30 was 97.5%. Mean query-pipeline latency was
10.03 s and P95 latency was 13.46 s on CPU. The benchmark documents were removed
from pgvector after the report was copied out. Full machine-readable results are
stored in
[`v2/frozen/section_aware_bge-m3/pgvector-report.json`](v2/frozen/section_aware_bge-m3/pgvector-report.json).

## Workflow, streaming and persistence

A synchronous `/research` request and an SSE `/research/stream` request were
executed against the frozen pgvector corpus. Retrieval succeeded and returned
the source-supported DEFMap evidence at rank 1. SSE emitted ordered node, model,
tool and context-manager events with a shared run and trace identifier.

The container was restarted after the streamed run. `GET /runs/{run_id}` and
`POST /runs/{run_id}/resume` returned the persisted run with the same run ID,
status, answer and five evidence items. This validates completed-run durability
and idempotent resume across a container restart. Pending-node continuation is
covered separately by the Harness resilience suite.

The DeepSeek calls in this validation run did not complete: TLS handshakes to
external HTTPS endpoints were terminated in both the host and container
environment. Planner and Verifier failures were recorded in trace events and
the workflow followed its explicit fallback/abstention path. Consequently this
run validates failure handling, not successful online DeepSeek generation. The
published 60-case DeepSeek result remains the separate frozen end-to-end
experiment documented in `docs/evaluation.md`.

## Scope

This validation does not cover concurrent load, database backup/restore,
multi-host deployment or a production security audit.
