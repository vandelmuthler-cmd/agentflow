# Architecture and Runtime

## Runtime flow

1. `StructuredPlanner` receives the user question and returns one to three validated retrieval steps. If the model call fails, a deterministic plan is used and the failure is recorded.
2. `document_search` executes the configured retrieval pipeline. The V2 quality path expands the query, retrieves BM25 and BGE-M3 candidates, applies Reciprocal Rank Fusion, and reranks the top 30 with a Cross-Encoder.
3. The optional native tool router can request another search or call `document_context` for adjacent chunks. Its arguments are validated and it can only reference evidence returned in the current run.
4. `ContextManager` deduplicates evidence, reserves output tokens, truncates oversized evidence, and enforces the total context budget.
5. `StructuredEvidenceVerifier` maps required answer aspects to evidence IDs. Missing aspects are converted into targeted follow-up queries.
6. `AdaptiveRetrievalController` admits only novel queries while enforcing total-query, follow-up, time, and no-progress limits.
7. `LLMWriter` produces an evidence-grounded answer. A deterministic writer is available when no model is configured.
8. `CitationVerifier` checks citation identifiers, coverage, groundedness proxies, and abstention state.

## State and persistence

`AgentState` is the shared typed state for the graph. It contains the question, plan, evidence, verification history, context statistics, model usage, tool history, retry counters, stop reason, and final answer.

LangGraph checkpoints are stored in SQLite after graph transitions. A checkpoint is a durable state snapshot, not a continuously running monitor. Resume loads the latest state and continues from pending work; a completed run is idempotent and is not executed again.

Trace events are separate from checkpoints. They record node and tool spans, parent-child relationships, status, wall time, errors, token usage, and component counters. Checkpoints answer "where can the workflow resume?" while traces answer "what happened during execution?".

## Agent boundary

AgentFlow is a single-agent system. Planner, Verifier, and Writer are specialized nodes in one shared workflow, not independently owned agents with separate goals, memories, or delegation contracts. Model autonomy is bounded by the Harness:

- Pydantic validates plans, verifier output, and tool arguments.
- The tool registry controls which actions can execute.
- Conditional edges and hard budgets prevent unbounded loops.
- Context management controls what reaches the model.
- Failure policy determines retry, fallback, abstention, or termination.
- Checkpoint and trace stores make runs recoverable and auditable.

## Retrieval modes

| Mode | Pipeline | Intended use |
| --- | --- | --- |
| `vector` | BGE-M3 top-k | Interactive and lower-latency path |
| `rrf_expanded` | Query expansion + BM25/BGE-M3 + RRF | Fusion experiment |
| `cross_encoder` | `rrf_expanded` top-30 + Cross-Encoder | Highest measured retrieval quality |

The Cross-Encoder cannot recover evidence absent from its candidate pool. Candidate all-gold recall is therefore reported separately from reranked Hit@K.
