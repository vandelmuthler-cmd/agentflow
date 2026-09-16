# Bounded native tool calling

AgentFlow has two read-only document tools:

| Tool | Input | Output | Intended use |
| --- | --- | --- | --- |
| `document_search` | `query`, `top_k` | ranked evidence chunks | Find initial or missing evidence |
| `document_context` | `evidence_id`, `window` (0-2) | the matched chunk and nearby chunks | Recover context split across chunk boundaries |

The Planner still produces one to three initial `document_search` queries. It does not choose `document_context`, because no evidence ID exists before search. The optional tool router runs after the initial search inside the Retriever node. It sends the two tool schemas to the configured chat model, reads provider-native `tool_calls`, validates and executes requests through `ToolRegistry`, and returns each observation as a `tool` message with the provider call ID. The model may request another action or stop. Verifier, Writer, and citation checks remain mandatory graph stages.

## Enable

The offline demo and existing evaluation scripts do not enable this feature. For an online demo with an API key already configured through environment variables:

```bash
python -B scripts/run_demo.py --online --tool-calling
```

For the API service, set `AGENTFLOW_TOOL_CALLING_ENABLED=true` before startup. The default is `false` because native tool selection was not enabled in the published frozen V2 benchmark.

## Boundaries

- At most two model decision rounds and two executed tool actions occur in the initial retrieval pass. Later Verifier-directed retrieval rounds keep the existing controller behavior.
- An ID passed to `document_context` must have appeared in this run's search results. The tool itself never reads another document's neighbors.
- Repeated search queries and calls beyond the configured search/duration budget are rejected. Tool inputs and outputs still pass through Pydantic validation, timeout, retry, and tracing.
- A model request failure leaves the original search evidence available to the Verifier and Writer.
- Search top-k and requested context chunks are separate evidence channels. Up to `top_k` unique context chunks can be attached after search ranking, so an observed neighbor is not discarded by search top-k alone. The context manager can still truncate or drop it when the input token budget is exhausted.
- Each executed call records `returned_evidence_ids`; context calls additionally record `retained_evidence_ids` after context budgeting. A returned chunk is not automatically evidence used by the Verifier or Writer.
- The tool call history and decision stop reason are included in State, Checkpoint, and API reports. Search call counters remain search-only; the trace also counts context calls.
- Model-directed tool use is not evidence of better answer quality. No frozen online comparison has yet been run for this optional path.

A two-case development pilot was used only as an implementation check and is not part of the published benchmark. If an online Verifier response fails schema validation, its reported token usage is retained, the run is marked `verifier_error`, and the normal Writer is skipped; the offline no-key path remains unchanged. See [architecture](architecture.md) for the runtime boundary and [evaluation](evaluation.md) for the frozen configuration.

This is still one Agent with a bounded document-research workflow, not a multi-agent system or a general-purpose tool platform. Merely wrapping Planner or Writer as callable functions would not add meaningful tool choice.
