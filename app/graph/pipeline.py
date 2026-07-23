"""
The Phase-1 pipeline as a LangGraph, now doing real Content Batching per the spec:

    Research: pull backlog first (Idea Library top-up), then top up with only as
              many fresh candidates as needed to reach IDEAS_PER_BATCH
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
from app.repositories.repository import (
    get_backlog_ideas,
    get_recent_approved_topics,
    save_decision,
    save_idea,
    save_post,
    update_idea_status,
)
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
    stale: list[Idea]            # backlog-sourced ideas filtered out this cycle -- archive, don't re-pull
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
    session = get_session()
    try:
        backlog_ideas = get_backlog_ideas(session, config.IDEAS_PER_BATCH)
        # Same Content Diversity Check query the Guardian uses for strategic_fit,
        # fetched here too so Research steers away from recently covered themes up
        # front instead of relying solely on the post-hoc dedup filter to catch
        # collisions -- see the module docstring in research_agent.py.
        recent_topics = get_recent_approved_topics(session, limit=20)
    finally:
        session.close()

    # Idea Library top-up: only ask the Research Agent for what the backlog didn't
    # already cover -- this is the whole point of persisting surplus ideas instead
    # of discarding them, and it directly saves LLM calls on the free-tier budget.
    deficit = max(0, config.IDEAS_PER_BATCH - len(backlog_ideas))
    fresh_ideas = run_research(state["brand"], deficit, recent_topics=recent_topics) if deficit > 0 else []
    candidates = [*backlog_ideas, *fresh_ideas]

    queue, surplus, stale, note = build_batch_queue(
        candidates,
        vector_store=get_vector_store(),
        dedup_threshold=config.DEDUP_SIMILARITY_THRESHOLD,
        batch_size=config.BATCH_SIZE,
        same_topic_threshold=config.DEDUP_SAME_TOPIC_THRESHOLD,
    )

    backlog_summary = f", {len(backlog_ideas)} pulled from backlog" if backlog_ideas else ""
    history_note = f", diversity_context={len(recent_topics)} recent posts" if recent_topics else ""
    log_entry = AgentDecisionLog(
        agent_name="ResearchAgent",
        input_summary=f"niche={state['brand'].niche}, requested={deficit} fresh{backlog_summary}{history_note}",
        output_summary=_with_retry_note(note),
    )
    return {
        "queue": queue,
        "surplus": surplus,
        "stale": stale,
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

    # Content Diversity Check: pull the last 20 approved topics from SQLite so the
    # Guardian can score strategic_fit against real calendar history, not just a
    # qualitative guess. This is a cheap read-only query done once per item.
    session = get_session()
    try:
        recent_topics = get_recent_approved_topics(session, limit=20)
    finally:
        session.close()

    result = evaluate(state["brand"], state["content"], recent_topics=recent_topics)
    history_note = f", diversity_context={len(recent_topics)} recent posts" if recent_topics else ""
    log_entry = AgentDecisionLog(
        agent_name="BrandGuardianAgent",
        input_summary=f"idea='{idea.topic}' ({state['queue_index'] + 1}/{len(state['queue'])}){history_note}",
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

    Three kinds of ideas get handled differently here, all part of closing the loop
    on the Idea Library backlog:
    - Freshly-researched ideas (no source_backlog_id) get a new row, as before.
    - Backlog-sourced ideas that were processed this cycle get their *existing* row
      updated in place (approved/rejected) rather than a duplicate inserted.
    - Backlog-sourced ideas filtered out this cycle as stale (build_batch_queue's
      `stale` list) get archived so they stop being re-pulled and re-filtered
      every cycle indefinitely.
    Freshly-researched surplus ideas are still saved as new 'backlog' rows;
    backlog-sourced surplus ideas need no change, they're already 'backlog'.
    """
    session = get_session()
    try:
        for result in state["batch_results"]:
            idea = result["idea"]
            if idea.source_backlog_id is not None:
                update_idea_status(session, idea.source_backlog_id, result["status"], dedup_note=idea.dedup_note)
                idea_id = idea.source_backlog_id
            else:
                idea_record = save_idea(session, idea, status=result["status"], dedup_note=idea.dedup_note)
                idea_id = idea_record.id
            save_post(session, idea_id, result["content"], result["guardian_result"])
            if result["status"] == "approved":
                get_vector_store().add_idea(str(idea_id), idea_text(idea), topic=idea.topic)

        for surplus_idea in state.get("surplus", []):
            if surplus_idea.source_backlog_id is None:
                save_idea(session, surplus_idea, status="backlog", dedup_note=surplus_idea.dedup_note)
            # else: backlog-sourced and still surplus -- row is already 'backlog', no-op.

        for stale_idea in state.get("stale", []):
            update_idea_status(session, stale_idea.source_backlog_id, "archived", dedup_note=stale_idea.dedup_note)

        for entry in state["decisions"]:
            save_decision(session, entry)
        session.commit()
    finally:
        session.close()

    approved = sum(1 for r in state["batch_results"] if r["status"] == "approved")
    rejected = len(state["batch_results"]) - approved
    backlogged = sum(1 for i in state.get("surplus", []) if i.source_backlog_id is None)
    archived = len(state.get("stale", []))
    backlog_note = f", {backlogged} backlogged" if backlogged else ""
    archive_note = f", {archived} stale backlog archived" if archived else ""
    return {
        "batch_summary": (
            f"{len(state['batch_results'])} processed, {approved} approved, "
            f"{rejected} rejected{backlog_note}{archive_note}"
        )
    }


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