"""
The Phase-1 pipeline as a LangGraph, now doing real Content Batching per the spec:

    Research (IDEAS_PER_BATCH candidates)
        -> Dedup (vs. history) + near-duplicate filter (within this batch)
        -> take top BATCH_SIZE by confidence
        -> for each: Produce -> Guardian (bounded per-item retry) -> record result
        -> Persist whole batch to SQLite, index approved ideas in ChromaDB

This replaces the earlier one-idea-per-run version. The graph still has only one
conditional edge doing real work (the per-item retry/advance decision) -- looping over
a queue via state, not a wider agent mesh, per the spec's "linear graph with
conditional edges" standard.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.brand_guardian_agent import evaluate
from app.agents.content_producer_agent import produce_content
from app.agents.research_agent import run_research
from app.core import config
from app.core.schemas import (
    AgentDecisionLog,
    BrandGuardianResult,
    BrandProfile,
    GeneratedContent,
    Idea,
)
from app.providers.llm import get_llm
from app.repositories.db import get_session
from app.repositories.repository import save_decision, save_idea, save_post
from app.repositories.vector_store import get_vector_store
from app.services.batching import build_batch_queue, idea_text


class BatchResult(TypedDict):
    idea: Idea
    content: GeneratedContent
    guardian_result: BrandGuardianResult
    status: str  # "approved" | "rejected"


class PipelineState(TypedDict, total=False):
    brand: BrandProfile
    queue: list[Idea]
    surplus: list[Idea]          # ideas that survived dedup but didn't make this batch's cutoff
    queue_index: int
    item_retries: int
    content: GeneratedContent
    guardian_result: BrandGuardianResult
    batch_results: list[BatchResult]
    decisions: list[AgentDecisionLog]
    batch_summary: str


def _log(state: PipelineState, entry: AgentDecisionLog) -> list[AgentDecisionLog]:
    return [*state.get("decisions", []), entry]


def _with_retry_note(summary: str) -> str:
    retries = get_llm().last_retry_count
    return f"{summary} (after {retries} retr{'y' if retries == 1 else 'ies'})" if retries else summary


def research_node(state: PipelineState) -> dict:
    candidates = run_research(state["brand"], config.IDEAS_PER_BATCH)
    queue, surplus, note = build_batch_queue(
        candidates,
        vector_store=get_vector_store(),
        dedup_threshold=config.DEDUP_SIMILARITY_THRESHOLD,
        batch_size=config.BATCH_SIZE,
    )

    log_entry = AgentDecisionLog(
        agent_name="ResearchAgent",
        input_summary=f"niche={state['brand'].niche}, requested={config.IDEAS_PER_BATCH}",
        output_summary=_with_retry_note(note),
    )
    return {
        "queue": queue,
        "surplus": surplus,
        "queue_index": 0,
        "item_retries": 0,
        "batch_results": [],
        "decisions": _log(state, log_entry),
    }


def produce_node(state: PipelineState) -> dict:
    idea = state["queue"][state["queue_index"]]
    content = produce_content(state["brand"], idea)
    log_entry = AgentDecisionLog(
        agent_name="ContentProducerAgent",
        input_summary=f"idea='{idea.topic}' ({state['queue_index'] + 1}/{len(state['queue'])})",
        output_summary=_with_retry_note(
            f"caption produced ({len(content.caption)} chars), prompt_version={content.prompt_version}"
        ),
    )
    return {"content": content, "decisions": _log(state, log_entry)}


def guardian_node(state: PipelineState) -> dict:
    idea = state["queue"][state["queue_index"]]
    result = evaluate(state["brand"], state["content"])
    log_entry = AgentDecisionLog(
        agent_name="BrandGuardianAgent",
        input_summary=f"idea='{idea.topic}' ({state['queue_index'] + 1}/{len(state['queue'])})",
        output_summary=_with_retry_note(result.reason),
        passed=result.passed,
        scores=result.scores,
    )
    return {"guardian_result": result, "decisions": _log(state, log_entry)}


def bump_item_retry_node(state: PipelineState) -> dict:
    return {"item_retries": state.get("item_retries", 0) + 1}


def record_result_node(state: PipelineState) -> dict:
    idea = state["queue"][state["queue_index"]]
    status = "approved" if state["guardian_result"].passed else "rejected"
    result: BatchResult = {
        "idea": idea,
        "content": state["content"],
        "guardian_result": state["guardian_result"],
        "status": status,
    }
    return {
        "batch_results": [*state.get("batch_results", []), result],
        "item_retries": 0,  # reset for the next item in the queue
    }


def advance_node(state: PipelineState) -> dict:
    return {"queue_index": state["queue_index"] + 1}


def persist_batch_node(state: PipelineState) -> dict:
    """Writes every item in the batch to SQLite and indexes approved ideas in ChromaDB.
    Also saves surplus (backlogged) ideas that survived dedup but didn't make this
    batch's cutoff, so they are preserved in the Idea Library rather than discarded.
    """
    session = get_session()
    try:
        for result in state["batch_results"]:
            idea_record = save_idea(session, result["idea"], status=result["status"])
            save_post(session, idea_record.id, result["content"], result["guardian_result"])
            if result["status"] == "approved":
                get_vector_store().add_idea(str(idea_record.id), idea_text(result["idea"]))
        for surplus_idea in state.get("surplus", []):
            save_idea(session, surplus_idea, status="backlog")
        for entry in state["decisions"]:
            save_decision(session, entry)
        session.commit()
    finally:
        session.close()

    approved = sum(1 for r in state["batch_results"] if r["status"] == "approved")
    rejected = len(state["batch_results"]) - approved
    backlogged = len(state.get("surplus", []))
    backlog_note = f", {backlogged} backlogged" if backlogged else ""
    return {"batch_summary": f"{len(state['batch_results'])} processed, {approved} approved, {rejected} rejected{backlog_note}"}


def route_after_guardian(state: PipelineState) -> str:
    if state["guardian_result"].passed:
        return "approved"
    if state.get("item_retries", 0) < config.MAX_GUARDIAN_RETRIES:
        return "retry"
    return "rejected"


def route_after_record(state: PipelineState) -> str:
    if state["queue_index"] + 1 < len(state["queue"]):
        return "next"
    return "finalize"


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("research", research_node)
    graph.add_node("produce", produce_node)
    graph.add_node("guardian", guardian_node)
    graph.add_node("bump_item_retry", bump_item_retry_node)
    graph.add_node("record_result", record_result_node)
    graph.add_node("advance", advance_node)
    graph.add_node("persist_batch", persist_batch_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "produce")
    graph.add_edge("produce", "guardian")
    graph.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {"approved": "record_result", "retry": "bump_item_retry", "rejected": "record_result"},
    )
    graph.add_edge("bump_item_retry", "produce")
    graph.add_conditional_edges(
        "record_result",
        route_after_record,
        {"next": "advance", "finalize": "persist_batch"},
    )
    graph.add_edge("advance", "produce")
    graph.add_edge("persist_batch", END)

    return graph.compile()
