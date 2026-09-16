from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agentflow.agents.context import ContextManager, ContextPolicy
from agentflow.agents.evidence_verifier import StructuredEvidenceVerifier
from agentflow.agents.planner import PlanStep, StructuredPlanner
from agentflow.agents.retrieval_controller import (
    AdaptiveRetrievalController,
    RetrievalBudgetPolicy,
)
from agentflow.agents.state import AgentState
from agentflow.agents.tool_caller import BoundedToolCaller
from agentflow.config import (
    MAX_CONTEXT_TOKENS,
    MAX_FOLLOW_UP_QUERIES,
    MAX_NO_PROGRESS_ROUNDS,
    MAX_RETRIES,
    MAX_RETRIEVAL_DURATION_MS,
    MAX_RETRIEVAL_QUERIES,
    MAX_TOKENS_PER_EVIDENCE,
    QUERY_SIMILARITY_THRESHOLD,
    RESERVED_OUTPUT_TOKENS,
    RUN_WORKSPACE_DIR,
)
from agentflow.generation.citation_verifier import CitationVerifier
from agentflow.generation.evidence_writer import EvidenceWriter
from agentflow.generation.llm_writer import LLMWriter, LLMWriterError
from agentflow.model_usage import ModelUsage
from agentflow.schemas import NodeStatus, ResearchReport, TraceEvent
from agentflow.tools.base import ToolExecutionContext


class GraphState(TypedDict, total=False):
    run_id: str
    question: str
    search_query: str
    top_k: int
    plan: list[str]
    plan_steps: list[PlanStep]
    task_type: str
    planning_mode: str
    planning_error: str
    context_stats: Any
    evidence: list[Any]
    answer: str
    writer_mode: str
    writer_error: str
    generation_quality: Any
    model_usage: list[Any]
    verifier_decision: str
    verifier_mode: str
    verifier_error: str
    verifier_required_aspects: list[str]
    verifier_covered_aspects: list[str]
    verifier_missing_aspects: list[str]
    verifier_follow_up_queries: list[str]
    verifier_evidence_mapping: dict[str, list[str]]
    verifier_history: list[dict[str, Any]]
    retrieval_queries: list[str]
    retrieval_rounds: list[dict[str, Any]]
    retrieval_tool_calls: int
    retrieval_tool_attempts: int
    retrieval_duration_ms: float
    retrieval_no_progress_rounds: int
    retrieval_stop_reason: str
    tool_call_history: list[dict[str, Any]]
    tool_decision_stop_reason: str
    retries: int
    status: str


class LangGraphResearchWorkflow:
    """Research workflow implemented as a LangGraph StateGraph."""

    def __init__(
        self,
        tools,
        trace_store,
        planner: StructuredPlanner | None = None,
        evidence_verifier: StructuredEvidenceVerifier | None = None,
        llm_writer: LLMWriter | None = None,
        context_manager: ContextManager | None = None,
        retrieval_controller: AdaptiveRetrievalController | None = None,
        checkpoint_backend=None,
        interrupt_before: list[str] | None = None,
        tool_caller: BoundedToolCaller | None = None,
        enable_tool_calling: bool = False,
    ) -> None:
        self.tools = tools
        self.trace_store = trace_store
        if hasattr(self.tools, "set_trace_store"):
            self.tools.set_trace_store(trace_store)
        self.checkpoint_backend = checkpoint_backend
        self.checkpointer = checkpoint_backend.saver if checkpoint_backend else InMemorySaver()
        self.interrupt_before = interrupt_before or []
        self.offline_writer = EvidenceWriter()
        self.llm_writer = llm_writer or LLMWriter.from_env()
        self.citation_verifier = CitationVerifier()
        self.planner = planner or StructuredPlanner.from_env()
        self.evidence_verifier = evidence_verifier or StructuredEvidenceVerifier.from_env()
        self.enable_tool_calling = enable_tool_calling
        self.tool_caller = tool_caller or BoundedToolCaller.from_env()
        self.retrieval_controller = retrieval_controller or AdaptiveRetrievalController(
            RetrievalBudgetPolicy(
                max_follow_up_rounds=MAX_RETRIES,
                max_total_queries=MAX_RETRIEVAL_QUERIES,
                max_follow_up_queries=MAX_FOLLOW_UP_QUERIES,
                max_retrieval_duration_ms=MAX_RETRIEVAL_DURATION_MS,
                max_no_progress_rounds=MAX_NO_PROGRESS_ROUNDS,
                query_similarity_threshold=QUERY_SIMILARITY_THRESHOLD,
            )
        )
        self.context_manager = context_manager or ContextManager(
            RUN_WORKSPACE_DIR,
            ContextPolicy(
                max_context_tokens=MAX_CONTEXT_TOKENS,
                reserved_output_tokens=RESERVED_OUTPUT_TOKENS,
                max_tokens_per_evidence=MAX_TOKENS_PER_EVIDENCE,
            ),
        )
        self.graph = self._build_graph()

    def run(self, state: AgentState) -> ResearchReport:
        final_state = AgentState.model_validate(
            self.graph.invoke(state.model_dump(mode="python"), config=self._graph_config(state.run_id))
        )
        return self._to_report(final_state)

    def get_report(self, run_id: str) -> ResearchReport | None:
        state = self.get_checkpoint_state(run_id)
        if state is None:
            return None
        return self._to_report(state)

    def resume(self, run_id: str) -> ResearchReport | None:
        snapshot = self.graph.get_state(self._graph_config(run_id))
        if not snapshot.values:
            return None
        if not snapshot.next:
            return self._to_report(AgentState.model_validate(snapshot.values))

        final_values = self.graph.invoke(None, config=self._graph_config(run_id))
        return self._to_report(AgentState.model_validate(final_values))

    def close(self) -> None:
        if self.checkpoint_backend is not None:
            self.checkpoint_backend.close()
        if hasattr(self.trace_store, "close"):
            self.trace_store.close()

    def stream(self, state: AgentState) -> Iterator[TraceEvent]:
        seen = 0
        final_update: dict[str, Any] | None = None
        for update in self.graph.stream(
            state.model_dump(mode="python"),
            config=self._graph_config(state.run_id),
            stream_mode="updates",
        ):
            final_update = update
            events = self.trace_store.read(state.run_id)
            for raw_event in events[seen:]:
                yield TraceEvent.model_validate(raw_event)
            seen = len(events)

        if final_update is not None:
            events = self.trace_store.read(state.run_id)
            for raw_event in events[seen:]:
                yield TraceEvent.model_validate(raw_event)

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("verifier", self._verifier_node)
        graph.add_node("query_rewriter", self._query_rewriter_node)
        graph.add_node("writer", self._writer_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retriever")
        graph.add_conditional_edges(
            "retriever",
            self._route_after_retriever,
            {
                "verifier": "verifier",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "verifier",
            self._route_after_verifier,
            {
                "query_rewriter": "query_rewriter",
                "writer": "writer",
            },
        )
        graph.add_edge("query_rewriter", "retriever")
        graph.add_edge("writer", END)
        return graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=self.interrupt_before,
        )

    def get_checkpoint_state(self, run_id: str) -> AgentState | None:
        snapshot = self.graph.get_state(self._graph_config(run_id))
        if not snapshot.values:
            return None
        return AgentState.model_validate(snapshot.values)

    def get_checkpoint_history(self, run_id: str) -> list[dict[str, Any]]:
        history = []
        for snapshot in self.graph.get_state_history(self._graph_config(run_id)):
            history.append(
                {
                    "next": list(snapshot.next),
                    "tasks": len(snapshot.tasks),
                    "status": snapshot.values.get("status"),
                    "verifier_decision": snapshot.values.get("verifier_decision"),
                    "retries": snapshot.values.get("retries", 0),
                    "search_query": snapshot.values.get("search_query"),
                    "planning_mode": snapshot.values.get("planning_mode"),
                    "plan_step_count": len(snapshot.values.get("plan_steps") or []),
                    "context_input_tokens": (snapshot.values.get("context_stats") or {}).get(
                        "estimated_input_tokens", 0
                    ),
                    "evidence_count": len(snapshot.values.get("evidence") or []),
                    "has_answer": bool(snapshot.values.get("answer")),
                }
            )
        return history

    def _planner_node(self, graph_state: GraphState) -> GraphState:
        state = self._to_agent_state(graph_state)
        self._emit(state, "planner", NodeStatus.STARTED, "开始拆解研究任务")
        planner_model_enabled = bool(getattr(self.planner, "is_configured", False))
        if planner_model_enabled:
            self._emit(
                state,
                "model.planner",
                NodeStatus.STARTED,
                "规划模型调用开始",
                {"parent_node": "planner", "model": getattr(self.planner, "model", "")},
            )
        outcome = self.planner.create_plan(
            state.question, self.tools.list_tools({"document_search"})
        )
        if outcome.usage.source == "api":
            state.model_usage.append(outcome.usage)
        if planner_model_enabled:
            self._emit(
                state,
                "model.planner",
                NodeStatus.FAILED if outcome.error else NodeStatus.COMPLETED,
                "规划模型调用失败" if outcome.error else "规划模型调用完成",
                {
                    "parent_node": "planner",
                    "model": getattr(self.planner, "model", ""),
                    "planning_mode": outcome.mode,
                    "error": outcome.error,
                    "error_type": "planner_error" if outcome.error else "",
                    **outcome.usage.model_dump(mode="json"),
                },
            )
        state.plan_steps = outcome.plan.steps
        state.task_type = outcome.plan.task_type
        state.planning_mode = outcome.mode
        state.planning_error = outcome.error
        state.plan = [step.objective for step in state.plan_steps]
        state.search_query = state.plan_steps[0].query
        self._emit(
            state,
            "planner",
            NodeStatus.COMPLETED,
            "任务拆解完成",
            {
                "task_type": state.task_type,
                "planning_mode": state.planning_mode,
                "planning_error": state.planning_error,
                "plan_steps": [step.model_dump(mode="json") for step in state.plan_steps],
            },
        )
        return self._to_graph_state(state)

    def _retriever_node(self, graph_state: GraphState) -> GraphState:
        state = self._to_agent_state(graph_state)
        steps = self._retrieval_steps(state)
        if not steps:
            state.status = "failed"
            state.retrieval_stop_reason = state.retrieval_stop_reason or "no_approved_queries"
            self._emit(
                state,
                "retriever",
                NodeStatus.FAILED,
                "没有通过 Harness 审批的检索查询",
                {"stop_reason": state.retrieval_stop_reason},
            )
            return self._to_graph_state(state)
        self._emit(
            state,
            "retriever",
            NodeStatus.STARTED,
            "开始执行检索计划",
            {"queries": [step.query for step in steps]},
        )
        previous_evidence_ids = {item.id for item in state.evidence}
        ranked_lists: list[list[Any]] = []
        if state.retries > 0 and state.evidence:
            ranked_lists.append(state.evidence)
        call_summaries = []
        for step in steps:
            result = self.tools.run(
                step.tool,
                context=ToolExecutionContext(run_id=state.run_id, node="retriever"),
                query=step.query,
                top_k=state.top_k,
            )
            call_summaries.append(
                {
                    "step_id": step.id,
                    "tool": step.tool,
                    "query": step.query,
                    "ok": result.ok,
                    "tool_call_id": result.tool_call_id,
                    "attempts": result.attempts,
                    "duration_ms": round(result.duration_ms, 3),
                    "outcome": result.outcome.value,
                }
            )
            if result.ok and result.data:
                ranked_lists.append(result.data)

        native_search_queries: list[str] = []
        context_evidence_lists: list[tuple[str, list[Any]]] = []
        if self.enable_tool_calling and state.retries == 0:
            def on_model_call(phase, round_number, duration_ms, usage, error):
                status = {
                    "started": NodeStatus.STARTED,
                    "completed": NodeStatus.COMPLETED,
                    "failed": NodeStatus.FAILED,
                }[phase]
                self._emit(
                    state, "model.tool_router", status, f"工具决策第 {round_number} 轮{phase}",
                    {
                        "parent_node": "retriever",
                        "round": round_number,
                        "duration_ms": round(duration_ms, 3),
                        "error": error,
                        "model": self.tool_caller.model,
                        **usage.model_dump(mode="json"),
                    },
                )

            outcome = self.tool_caller.run(
                question=state.question,
                evidence=[item for ranked in ranked_lists for item in ranked],
                registry=self.tools,
                run_id=state.run_id,
                top_k=state.top_k,
                seen_queries={*state.retrieval_queries, *(step.query for step in steps)},
                remaining_searches=max(
                    0, MAX_RETRIEVAL_QUERIES - state.retrieval_tool_calls - len(call_summaries)
                ),
                remaining_duration_ms=max(
                    0.0,
                    MAX_RETRIEVAL_DURATION_MS - state.retrieval_duration_ms
                    - sum(item["duration_ms"] for item in call_summaries),
                ),
                on_model_call=on_model_call,
            )
            ranked_lists.extend(outcome.evidence_lists)
            context_evidence_lists = outcome.context_evidence_lists
            state.model_usage.extend(outcome.usage)
            state.tool_call_history.extend(outcome.calls)
            state.tool_decision_stop_reason = outcome.stop_reason
            for record in outcome.calls:
                if not record.get("executed"):
                    continue
                arguments = record["arguments"]
                if record["tool"] == "document_search":
                    native_search_queries.append(arguments["query"])
                call_summaries.append(
                    {
                        "step_id": f"native_{len(call_summaries) + 1}",
                        "tool": record["tool"],
                        "query": arguments.get("query", ""),
                        "ok": record["ok"],
                        "tool_call_id": record["tool_call_id"],
                        "attempts": record["attempts"],
                        "duration_ms": record["duration_ms"],
                        "outcome": record["outcome"],
                    }
                )

        state.retrieval_queries.extend(step.query for step in steps)
        state.retrieval_queries.extend(native_search_queries)
        search_summaries = [item for item in call_summaries if item["tool"] == "document_search"]
        state.retrieval_tool_calls += len(search_summaries)
        state.retrieval_tool_attempts += sum(item["attempts"] for item in search_summaries)
        round_duration_ms = sum(item["duration_ms"] for item in call_summaries)
        state.retrieval_duration_ms += round_duration_ms

        if call_summaries and all(not item["ok"] for item in call_summaries):
            if state.retries == 0 or not state.evidence:
                state.status = "failed"
            else:
                state.retrieval_no_progress_rounds += 1
                state.retrieval_stop_reason = "follow_up_tools_failed"
            self._emit(
                state,
                "retriever",
                NodeStatus.FAILED,
                "所有检索子任务均失败",
                {
                    "tool_calls": call_summaries,
                    "retained_evidence_ids": [item.id for item in state.evidence],
                    "stop_reason": state.retrieval_stop_reason,
                },
            )
            return self._to_graph_state(state)

        if state.retries > 0 and state.evidence:
            previous_by_id = {item.id: item for item in state.evidence}
            for record in state.tool_call_history:
                if record.get("tool") != "document_context" or not record.get("executed"):
                    continue
                group = [
                    previous_by_id[item_id]
                    for item_id in record.get("returned_evidence_ids", [])
                    if item_id in previous_by_id
                ]
                context_evidence_lists.append((record["arguments"]["evidence_id"], group))
        merged_evidence = self._merge_ranked_evidence(ranked_lists, state.top_k)
        merged_evidence = self._attach_context_evidence(
            merged_evidence, context_evidence_lists, state.top_k
        )
        self._emit(
            state,
            "context_manager",
            NodeStatus.STARTED,
            "开始构建预算内上下文",
            {"parent_node": "retriever", "evidence_count": len(merged_evidence)},
        )
        prepared_context = self.context_manager.prepare(
            run_id=state.run_id,
            question=state.question,
            plan_steps=state.plan_steps,
            evidence=merged_evidence,
        )
        state.evidence = prepared_context.evidence
        state.context_stats = prepared_context.stats
        current_evidence_ids = {item.id for item in state.evidence}
        for record in state.tool_call_history:
            if record.get("tool") == "document_context" and record.get("executed"):
                returned_ids = record.get("returned_evidence_ids", [])
                record["retained_evidence_ids"] = [
                    item_id for item_id in returned_ids if item_id in current_evidence_ids
                ]
        new_evidence_ids = sorted(current_evidence_ids - previous_evidence_ids)
        if state.retries > 0:
            if new_evidence_ids:
                state.retrieval_no_progress_rounds = 0
            else:
                state.retrieval_no_progress_rounds += 1
        state.retrieval_rounds.append(
            {
                "round": state.retries,
                "queries": [step.query for step in steps] + native_search_queries,
                "evidence_ids": [item.id for item in state.evidence],
                "new_evidence_ids": new_evidence_ids,
                "new_evidence_count": len(new_evidence_ids),
                "logical_tool_calls": len(search_summaries),
                "tool_attempts": sum(item["attempts"] for item in search_summaries),
                "duration_ms": round(round_duration_ms, 3),
            }
        )
        self._emit(
            state,
            "context_manager",
            NodeStatus.COMPLETED,
            "上下文构建完成",
            {
                "parent_node": "retriever",
                "context_stats": state.context_stats.model_dump(mode="json"),
            },
        )
        self._emit(
            state,
            "retriever",
            NodeStatus.COMPLETED,
            f"检索完成，召回 {len(state.evidence)} 条证据",
            {
                "queries": [step.query for step in steps] + native_search_queries,
                "evidence_ids": [item.id for item in state.evidence],
                "tool_calls": call_summaries,
                "new_evidence_ids": new_evidence_ids,
                "retrieval_tool_calls": state.retrieval_tool_calls,
                "retrieval_duration_ms": round(state.retrieval_duration_ms, 3),
                "no_progress_rounds": state.retrieval_no_progress_rounds,
                "context_stats": state.context_stats.model_dump(mode="json"),
                "tool_decision_stop_reason": state.tool_decision_stop_reason,
            },
        )
        return self._to_graph_state(state)

    def _route_after_retriever(self, graph_state: GraphState) -> str:
        state = self._to_agent_state(graph_state)
        if state.status == "failed" and not state.evidence:
            return "end"
        return "verifier"

    def _verifier_node(self, graph_state: GraphState) -> GraphState:
        state = self._to_agent_state(graph_state)
        self._emit(state, "verifier", NodeStatus.STARTED, "开始检查证据是否足够")
        model_enabled = bool(state.evidence and self.evidence_verifier.is_configured)
        if model_enabled:
            self._emit(
                state,
                "model.verifier",
                NodeStatus.STARTED,
                "证据评审模型调用开始",
                {"parent_node": "verifier", "model": self.evidence_verifier.model},
            )
        outcome = self.evidence_verifier.verify(state.question, state.plan, state.evidence)
        assessment = outcome.assessment
        if outcome.usage.source == "api":
            state.model_usage.append(outcome.usage)
        if model_enabled:
            self._emit(
                state,
                "model.verifier",
                NodeStatus.FAILED if outcome.error else NodeStatus.COMPLETED,
                "证据评审模型调用失败" if outcome.error else "证据评审模型调用完成",
                {
                    "parent_node": "verifier",
                    "model": self.evidence_verifier.model,
                    "verifier_mode": outcome.mode,
                    "decision": assessment.decision,
                    "error": outcome.error,
                    "error_type": "evidence_verifier_error" if outcome.error else "",
                    **outcome.usage.model_dump(mode="json"),
                },
            )

        previous_missing_aspects = (
            state.verifier_history[-1]["missing_aspects"]
            if state.verifier_history
            else []
        )
        state.verifier_mode = outcome.mode
        state.verifier_error = outcome.error
        state.verifier_required_aspects = assessment.required_aspects
        state.verifier_covered_aspects = assessment.covered_aspects
        state.verifier_missing_aspects = assessment.missing_aspects
        state.verifier_follow_up_queries = assessment.follow_up_queries
        state.verifier_evidence_mapping = assessment.evidence_mapping

        if outcome.error:
            state.verifier_decision = "insufficient"
            state.retrieval_stop_reason = "verifier_error"
            control_reason = "verifier_error"
            rejected_queries: list[str] = []
            message = "证据评审失败，停止生成正式答案"
        elif assessment.decision == "sufficient":
            state.verifier_decision = "sufficient"
            state.retrieval_stop_reason = "sufficient_evidence"
            control_reason = "sufficient_evidence"
            rejected_queries: list[str] = []
            message = "证据覆盖评审通过"
        else:
            control = self.retrieval_controller.decide_follow_up(
                proposed_queries=assessment.follow_up_queries,
                executed_queries=state.retrieval_queries,
                follow_up_rounds=state.retries,
                total_queries=state.retrieval_tool_calls,
                retrieval_duration_ms=state.retrieval_duration_ms,
                no_progress_rounds=state.retrieval_no_progress_rounds,
                missing_aspects=assessment.missing_aspects,
                previous_missing_aspects=previous_missing_aspects,
                verifier_decision=assessment.decision,
            )
            state.verifier_follow_up_queries = control.queries
            control_reason = control.reason
            rejected_queries = control.rejected_queries
            if control.allowed:
                state.verifier_decision = "rewrite_query"
                message = "证据覆盖不足，Harness 批准定向补充检索"
            else:
                state.verifier_decision = "insufficient"
                state.retrieval_stop_reason = control.reason
                message = f"Harness 停止补充检索：{control.reason}"

        state.verifier_history.append(
            {
                "retry_index": state.retries,
                "mode": outcome.mode,
                "decision": assessment.decision,
                "required_aspects": assessment.required_aspects,
                "covered_aspects": assessment.covered_aspects,
                "missing_aspects": assessment.missing_aspects,
                "proposed_follow_up_queries": assessment.follow_up_queries,
                "follow_up_queries": state.verifier_follow_up_queries,
                "rejected_queries": rejected_queries,
                "controller_reason": control_reason,
                "evidence_mapping": assessment.evidence_mapping,
                "evidence_ids": [item.id for item in state.evidence],
                "rationale": assessment.rationale,
                "error": outcome.error,
            }
        )
        self._emit(
            state,
            "verifier",
            NodeStatus.COMPLETED,
            message,
            {
                "decision": state.verifier_decision,
                "assessment_decision": assessment.decision,
                "mode": outcome.mode,
                "retries": state.retries,
                "required_aspects": assessment.required_aspects,
                "covered_aspects": assessment.covered_aspects,
                "missing_aspects": assessment.missing_aspects,
                "follow_up_queries": assessment.follow_up_queries,
                "approved_follow_up_queries": state.verifier_follow_up_queries,
                "rejected_queries": rejected_queries,
                "controller_reason": control_reason,
                "retrieval_tool_calls": state.retrieval_tool_calls,
                "retrieval_duration_ms": round(state.retrieval_duration_ms, 3),
                "no_progress_rounds": state.retrieval_no_progress_rounds,
                "evidence_mapping": assessment.evidence_mapping,
                "error": outcome.error,
            },
        )
        return self._to_graph_state(state)

    def _route_after_verifier(self, graph_state: GraphState) -> str:
        state = self._to_agent_state(graph_state)
        if state.verifier_decision == "rewrite_query":
            return "query_rewriter"
        return "writer"

    def _query_rewriter_node(self, graph_state: GraphState) -> GraphState:
        state = self._to_agent_state(graph_state)
        self._emit(state, "query_rewriter", NodeStatus.STARTED, "开始改写检索问题")
        state.retries += 1
        if state.verifier_follow_up_queries:
            state.search_query = state.verifier_follow_up_queries[0]
        else:
            state.search_query = f"{state.question} 关键事实 证据"
        self._emit(
            state,
            "query_rewriter",
            NodeStatus.COMPLETED,
            f"查询改写完成，准备第 {state.retries} 次重试",
            {
                "search_query": state.search_query,
                "follow_up_queries": state.verifier_follow_up_queries,
                "retries": state.retries,
            },
        )
        return self._to_graph_state(state)

    def _writer_node(self, graph_state: GraphState) -> GraphState:
        state = self._to_agent_state(graph_state)
        self._emit(state, "writer", NodeStatus.STARTED, "开始生成报告")
        if state.verifier_error:
            state.answer = "证据充分性评审失败，无法可靠确认现有证据是否足以回答；请稍后重试。"
            state.writer_mode = "skipped_after_verifier_error"
            state.generation_quality = self.citation_verifier.verify(state.answer, state.evidence)
            state.status = "completed_with_insufficient_evidence"
        elif not state.evidence:
            state.answer = self.offline_writer.write(state.question, state.evidence)
            state.writer_mode = "offline"
            state.generation_quality = self.citation_verifier.verify(state.answer, state.evidence)
            state.status = "completed_with_insufficient_evidence"
        else:
            model_enabled = self.llm_writer.is_configured
            if model_enabled:
                self._emit(
                    state,
                    "model.writer",
                    NodeStatus.STARTED,
                    "回答模型调用开始",
                    {"parent_node": "writer", "model": self.llm_writer.model},
                )
            state.answer, writer_usage = self._generate_answer(state)
            if writer_usage.source == "api":
                state.model_usage.append(writer_usage)
            if model_enabled:
                model_failed = state.writer_mode == "offline_fallback"
                self._emit(
                    state,
                    "model.writer",
                    NodeStatus.FAILED if model_failed else NodeStatus.COMPLETED,
                    "回答模型调用失败" if model_failed else "回答模型调用完成",
                    {
                        "parent_node": "writer",
                        "model": self.llm_writer.model,
                        "writer_mode": state.writer_mode,
                        "error": state.writer_error,
                        "error_type": "llm_writer_error" if model_failed else "",
                        **writer_usage.model_dump(mode="json"),
                    },
                )
            state.generation_quality = self.citation_verifier.verify(state.answer, state.evidence)
            state.status = (
                "completed_with_insufficient_evidence"
                if state.verifier_decision == "insufficient"
                else "completed"
            )

        self._emit(
            state,
            "writer",
            NodeStatus.COMPLETED,
            "报告生成完成",
            {
                "status": state.status,
                "writer_mode": state.writer_mode,
                "writer_error": state.writer_error,
                "citation_correctness": state.generation_quality.citation_correctness,
                "citation_completeness": state.generation_quality.citation_completeness,
                "groundedness": state.generation_quality.groundedness,
                "abstention": state.generation_quality.abstention,
            },
        )
        return self._to_graph_state(state)

    def _generate_answer(self, state: AgentState) -> tuple[str, ModelUsage]:
        if self.llm_writer.is_configured:
            try:
                state.writer_mode = "llm"
                result = self.llm_writer.write_with_usage(state.question, state.evidence)
                return result.content, result.usage
            except LLMWriterError as error:
                state.writer_mode = "offline_fallback"
                state.writer_error = str(error)
                return self.offline_writer.write(state.question, state.evidence), ModelUsage()

        state.writer_mode = "offline"
        return self.offline_writer.write(state.question, state.evidence), ModelUsage()

    def _emit(self, state: AgentState, node: str, status: NodeStatus, message: str, payload=None) -> None:
        self.trace_store.append(
            TraceEvent(
                run_id=state.run_id,
                node=node,
                status=status,
                message=message,
                payload=payload or {},
            )
        )

    def _to_agent_state(self, graph_state: GraphState) -> AgentState:
        return AgentState.model_validate(graph_state)

    def _to_graph_state(self, state: AgentState) -> GraphState:
        return GraphState(**state.model_dump(mode="python"))

    def _to_report(self, state: AgentState) -> ResearchReport:
        return ResearchReport(
            run_id=state.run_id,
            question=state.question,
            plan=state.plan,
            plan_steps=state.plan_steps,
            task_type=state.task_type,
            planning_mode=state.planning_mode,
            planning_error=state.planning_error,
            context_stats=state.context_stats,
            evidence=state.evidence,
            answer=state.answer,
            status=state.status,
            writer_mode=state.writer_mode,
            writer_error=state.writer_error,
            generation_quality=state.generation_quality,
            model_usage=state.model_usage,
            verifier_decision=state.verifier_decision,
            verifier_mode=state.verifier_mode,
            verifier_error=state.verifier_error,
            verifier_required_aspects=state.verifier_required_aspects,
            verifier_covered_aspects=state.verifier_covered_aspects,
            verifier_missing_aspects=state.verifier_missing_aspects,
            verifier_follow_up_queries=state.verifier_follow_up_queries,
            verifier_evidence_mapping=state.verifier_evidence_mapping,
            verifier_history=state.verifier_history,
            retrieval_queries=state.retrieval_queries,
            retrieval_rounds=state.retrieval_rounds,
            retrieval_tool_calls=state.retrieval_tool_calls,
            retrieval_tool_attempts=state.retrieval_tool_attempts,
            retrieval_duration_ms=state.retrieval_duration_ms,
            retrieval_no_progress_rounds=state.retrieval_no_progress_rounds,
            retrieval_stop_reason=state.retrieval_stop_reason,
            tool_call_history=state.tool_call_history,
            tool_decision_stop_reason=state.tool_decision_stop_reason,
        )

    def _graph_config(self, run_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": run_id}}

    def _retrieval_steps(self, state: AgentState) -> list[PlanStep]:
        if state.retries > 0:
            queries = state.verifier_follow_up_queries or [
                state.search_query or state.question
            ]
            return [
                PlanStep(
                    id=f"retry_{state.retries}_{index}",
                    objective="使用改写后的查询补充证据",
                    tool="document_search",
                    query=query,
                    success_criterion="至少返回一条可引用证据",
                )
                for index, query in enumerate(queries[:2], start=1)
            ]
        planned_steps = state.plan_steps or [
            PlanStep(
                id="step_1",
                objective="检索与问题直接相关的本地证据",
                tool="document_search",
                query=state.question,
                success_criterion="至少返回一条可引用证据",
            )
        ]
        selection = self.retrieval_controller.select_initial_queries(
            [step.query for step in planned_steps]
        )
        if not selection.allowed:
            state.retrieval_stop_reason = selection.reason
            return []
        selected_steps = []
        remaining_queries = list(selection.queries)
        for step in planned_steps:
            normalized_query = " ".join(step.query.split()).strip()
            if normalized_query not in remaining_queries:
                continue
            selected_steps.append(step.model_copy(update={"query": normalized_query}))
            remaining_queries.remove(normalized_query)
        return selected_steps

    def _merge_ranked_evidence(self, ranked_lists: list[list[Any]], top_k: int) -> list[Any]:
        if not ranked_lists:
            return []
        scores: dict[str, float] = {}
        evidence_by_id: dict[str, Any] = {}
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, start=1):
                evidence_by_id.setdefault(item.id, item)
                scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (60 + rank)
        ordered_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)[:top_k]
        return [
            evidence_by_id[item_id].model_copy(update={"rank": rank})
            for rank, item_id in enumerate(ordered_ids, start=1)
        ]

    @staticmethod
    def _attach_context_evidence(
        ranked_evidence: list[Any],
        context_lists: list[tuple[str, list[Any]]],
        top_k: int,
    ) -> list[Any]:
        attached = list(ranked_evidence)
        seen_ids = {item.id for item in attached}
        added = 0
        for anchor_id, group in context_lists:
            anchor_position = next((i for i, item in enumerate(group) if item.id == anchor_id), None)
            if anchor_position is None:
                continue
            ordered = sorted(
                enumerate(group), key=lambda pair: (abs(pair[0] - anchor_position), pair[0])
            )
            for _, item in ordered:
                if added >= top_k:
                    return attached
                if item.id in seen_ids:
                    continue
                attached.append(item.model_copy(update={"rank": len(attached) + 1}))
                seen_ids.add(item.id)
                added += 1
        return attached
