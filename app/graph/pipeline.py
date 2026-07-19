"""
The Phase-1 pipeline as a linear LangGraph, per the spec: a small graph with one
conditional edge for the regenerate loop -- not a full multi-agent mesh. This is the
smallest slice that proves the design: Research -> Dedup -> Content Producer ->
Brand Guardian -> Persist, with a bounded retry if the Guardian rejects the content.

Scheduling, publishing, and batching across multiple ideas per run are deliberately
not here yet -- they're additive once this slice runs end-to-end. Dedup and SQLite
persistence, which used to be in that "not yet" list, are now wired in: this is what
closes the gap the Guardian itself flagged in an early run ("the brand may need to
vary its content to avoid repetition") -- nothing was checking that before.
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


class PipelineState(TypedDict, total=False):
    brand: BrandProfile
    ideas: list[Idea]
    idea: Idea
    dedup_note: str
    content: GeneratedContent
    guardian_result: BrandGuardianResult
    retries: int
    decisions: list[AgentDecisionLog]
    status: str  # "approved" | "rejected"


def _log(state: PipelineState, entry: AgentDecisionLog) -> list[AgentDecisionLog]:
    return [*state.get("decisions", []), entry]


def _with_retry_note(summary: str) -> str:
    """Append '(after N retries)' to a log summary if the last LLM call needed retries."""
    retries = get_llm().last_retry_count
    return f"{summary} (after {retries} retr{'y' if retries == 1 else 'ies'})" if retries else summary


def _idea_text(idea: Idea) -> str:
    return f"{idea.topic}: {idea.angle}"


def research_node(state: PipelineState) -> dict:
    ideas = run_research(state["brand"], config.IDEAS_PER_RUN)

    # Dedup filter: drop any idea too similar to a previously *approved* idea.
    vector_store = get_vector_store()
    survivors = []
    filtered_notes = []
    for idea in ideas:
        matches = vector_store.find_similar(_idea_text(idea), config.DEDUP_SIMILARITY_THRESHOLD)
        if matches:
            filtered_notes.append(
                f"'{idea.topic}' filtered (similarity={matches[0][1]:.2f} to a prior approved idea)"
            )
        else:
            survivors.append(idea)

    if not survivors:
        # Every candidate matched something already approved. Rather than fail the
        # run, fall back to the original list so there's still something to review --
        # but the note makes it obvious in the log that this run produced nothing
        # genuinely fresh, which is itself useful signal.
        survivors = ideas
        dedup_note = "; ".join(filtered_notes) + " — all candidates were repeats; proceeding without a fresh alternative"
    else:
        dedup_note = "; ".join(filtered_notes) if filtered_notes else "no duplicates found"

    best = max(survivors, key=lambda i: i.confidence_score)
    log_entry = AgentDecisionLog(
        agent_name="ResearchAgent",
        input_summary=f"niche={state['brand'].niche}, count={config.IDEAS_PER_RUN}",
        output_summary=_with_retry_note(
            f"selected '{best.topic}' (confidence={best.confidence_score}) — {best.reasoning} | dedup: {dedup_note}"
        ),
    )
    return {
        "ideas": ideas,
        "idea": best,
        "dedup_note": dedup_note,
        "decisions": _log(state, log_entry),
    }


def produce_node(state: PipelineState) -> dict:
    content = produce_content(state["brand"], state["idea"])
    log_entry = AgentDecisionLog(
        agent_name="ContentProducerAgent",
        input_summary=f"idea='{state['idea'].topic}'",
        output_summary=_with_retry_note(
            f"caption produced ({len(content.caption)} chars), prompt_version={content.prompt_version}"
        ),
    )
    return {"content": content, "decisions": _log(state, log_entry)}


def guardian_node(state: PipelineState) -> dict:
    result = evaluate(state["brand"], state["content"])
    log_entry = AgentDecisionLog(
        agent_name="BrandGuardianAgent",
        input_summary=f"idea='{state['idea'].topic}'",
        output_summary=_with_retry_note(result.reason),
        passed=result.passed,
        scores=result.scores,
    )
    return {"guardian_result": result, "decisions": _log(state, log_entry)}


def bump_retry_node(state: PipelineState) -> dict:
    return {"retries": state.get("retries", 0) + 1}


def finalize_node(state: PipelineState) -> dict:
    status = "approved" if state["guardian_result"].passed else "rejected"
    return {"status": status}


def persist_node(state: PipelineState) -> dict:
    """
    Writes the idea, post, and full decision log to SQLite -- the console print in
    main.py is a convenience, this is the actual `agent_decisions` / `Post` / `Idea`
    persistence the spec calls for. Only an approved idea gets added to the vector
    store, so a rejected idea doesn't block a genuinely different future attempt at
    a similar angle.
    """
    session = get_session()
    try:
        idea_record = save_idea(session, state["idea"], status=state["status"], dedup_note=state.get("dedup_note", ""))
        save_post(session, idea_record.id, state["content"], state["guardian_result"])
        for entry in state["decisions"]:
            save_decision(session, entry)
        session.commit()

        if state["status"] == "approved":
            get_vector_store().add_idea(str(idea_record.id), _idea_text(state["idea"]))
    finally:
        session.close()

    return {}


def route_after_guardian(state: PipelineState) -> str:
    if state["guardian_result"].passed:
        return "approved"
    if state.get("retries", 0) < config.MAX_GUARDIAN_RETRIES:
        return "retry"
    return "rejected"


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("research", research_node)
    graph.add_node("produce", produce_node)
    graph.add_node("guardian", guardian_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("persist", persist_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "produce")
    graph.add_edge("produce", "guardian")
    graph.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {"approved": "finalize", "retry": "bump_retry", "rejected": "finalize"},
    )
    graph.add_edge("bump_retry", "produce")
    graph.add_edge("finalize", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
