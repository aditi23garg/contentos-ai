"""
The Phase-1 pipeline as a linear LangGraph, per the spec: a small graph with one
conditional edge for the regenerate loop -- not a full multi-agent mesh. This is the
smallest slice that proves the design: Research -> Content Producer -> Brand Guardian,
with a bounded retry if the Guardian rejects the content.

Scheduling, publishing, batching across multiple ideas, and the dedup/vector-DB step
are deliberately not here yet -- they're additive once this slice runs end-to-end.
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


class PipelineState(TypedDict, total=False):
    brand: BrandProfile
    ideas: list[Idea]
    idea: Idea
    content: GeneratedContent
    guardian_result: BrandGuardianResult
    retries: int
    decisions: list[AgentDecisionLog]
    status: str  # "approved" | "rejected"


def _log(state: PipelineState, entry: AgentDecisionLog) -> list[AgentDecisionLog]:
    return [*state.get("decisions", []), entry]


def research_node(state: PipelineState) -> dict:
    ideas = run_research(state["brand"], config.IDEAS_PER_RUN)
    best = max(ideas, key=lambda i: i.confidence_score)
    log_entry = AgentDecisionLog(
        agent_name="ResearchAgent",
        input_summary=f"niche={state['brand'].niche}, count={config.IDEAS_PER_RUN}",
        output_summary=f"selected '{best.topic}' (confidence={best.confidence_score}) — {best.reasoning}",
    )
    return {"ideas": ideas, "idea": best, "decisions": _log(state, log_entry)}


def produce_node(state: PipelineState) -> dict:
    content = produce_content(state["brand"], state["idea"])
    log_entry = AgentDecisionLog(
        agent_name="ContentProducerAgent",
        input_summary=f"idea='{state['idea'].topic}'",
        output_summary=f"caption produced ({len(content.caption)} chars), prompt_version={content.prompt_version}",
    )
    return {"content": content, "decisions": _log(state, log_entry)}


def guardian_node(state: PipelineState) -> dict:
    result = evaluate(state["brand"], state["content"])
    log_entry = AgentDecisionLog(
        agent_name="BrandGuardianAgent",
        input_summary=f"idea='{state['idea'].topic}'",
        output_summary=result.reason,
        passed=result.passed,
        scores=result.scores,
    )
    return {"guardian_result": result, "decisions": _log(state, log_entry)}


def bump_retry_node(state: PipelineState) -> dict:
    return {"retries": state.get("retries", 0) + 1}


def finalize_node(state: PipelineState) -> dict:
    status = "approved" if state["guardian_result"].passed else "rejected"
    return {"status": status}


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

    graph.set_entry_point("research")
    graph.add_edge("research", "produce")
    graph.add_edge("produce", "guardian")
    graph.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {"approved": "finalize", "retry": "bump_retry", "rejected": "finalize"},
    )
    graph.add_edge("bump_retry", "produce")
    graph.add_edge("finalize", END)

    return graph.compile()
