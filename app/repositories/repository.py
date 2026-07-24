"""
Repository functions: the only place that translates between the Pydantic schemas
the agents/graph work with (app/core/schemas.py) and the SQLAlchemy records that get
persisted (app/repositories/db.py). Nothing outside this module should import the
*Record classes directly -- that's the point of the Repository Pattern.
"""

from __future__ import annotations

from app.core.schemas import AgentDecisionLog, BrandGuardianResult, GeneratedContent, Idea
from app.repositories.db import (
    AgentDecisionRecord,
    FeedbackEditRecord,
    IdeaRecord,
    PostRecord,
    Session,
    from_json,
    to_json,
)


def save_idea(session: Session, idea: Idea, status: str, dedup_note: str = "") -> IdeaRecord:
    record = IdeaRecord(
        topic=idea.topic,
        angle=idea.angle,
        reasoning=idea.reasoning,
        confidence_score=idea.confidence_score,
        knowledge_sources_used=to_json(idea.knowledge_sources_used),
        status=status,
        dedup_note=dedup_note,
    )
    session.add(record)
    session.flush()  # populates record.id without a full commit
    return record


def get_backlog_ideas(session: Session, limit: int) -> list[Idea]:
    """
    Idea Library top-up: read back ideas saved as status='backlog' by a previous
    cycle (see build_batch_queue's `surplus`) as real Idea objects the pipeline can
    re-run through dedup/rank/production, instead of leaving them to rot unused.

    Ranked by confidence first, then oldest-first as a tiebreak so a backlog idea
    that's tied on confidence doesn't sit forever behind newer arrivals.
    """
    rows = (
        session.query(IdeaRecord)
        .filter(IdeaRecord.status == "backlog")
        .order_by(IdeaRecord.confidence_score.desc(), IdeaRecord.created_at.asc())
        .limit(limit)
        .all()
    )
    return [
        Idea(
            topic=r.topic,
            angle=r.angle,
            reasoning=r.reasoning,
            confidence_score=r.confidence_score,
            knowledge_sources_used=from_json(r.knowledge_sources_used) or [],
            source_backlog_id=r.id,
        )
        for r in rows
    ]


def update_idea_status(session: Session, idea_id: int, status: str, dedup_note: str | None = None) -> None:
    """
    Update an existing ideas row in place -- used for backlog-sourced ideas so a
    second cycle updates the original row (to approved/rejected/archived) instead
    of save_idea() inserting a duplicate row for the same idea.

    dedup_note is optional here (unlike save_idea) because not every caller has a
    fresh one -- e.g. persist_batch_node's stale-archiving pass reuses the note
    dedup_filter already set on the Idea object earlier in the same cycle.
    """
    record = session.query(IdeaRecord).filter(IdeaRecord.id == idea_id).one_or_none()
    if record is not None:
        record.status = status
        if dedup_note is not None:
            record.dedup_note = dedup_note


def save_post(
    session: Session,
    idea_id: int,
    content: GeneratedContent,
    guardian_result: BrandGuardianResult,
    brand_version: int | None = None,
) -> PostRecord:
    record = PostRecord(
        idea_id=idea_id,
        caption=content.caption,
        image_prompt=content.image_prompt,
        hashtags=to_json(content.hashtags),
        cta=content.cta,
        platform_variants=to_json(content.platform_variants),
        prompt_version=content.prompt_version,
        passed=guardian_result.passed,
        rubric_scores=to_json(guardian_result.scores.model_dump()),
        guardian_reason=guardian_result.reason,
        brand_version=brand_version,
    )
    session.add(record)
    session.flush()
    return record


def save_decision(session: Session, entry: AgentDecisionLog) -> AgentDecisionRecord:
    record = AgentDecisionRecord(
        agent_name=entry.agent_name,
        input_summary=entry.input_summary,
        output_summary=entry.output_summary,
        passed=entry.passed,
        scores=to_json(entry.scores.model_dump()) if entry.scores else None,
        timestamp=entry.timestamp,
    )
    session.add(record)
    return record


def recent_approved_idea_texts(session: Session, limit: int = 100) -> list[str]:
    """
    Used as a lightweight fallback/debug view of dedup context -- the real dedup
    decision is made against the vector store (app/repositories/vector_store.py),
    not this SQL query. Kept here because "what did we already publish" is useful
    to be able to ask the database directly, independent of the vector index.
    """
    rows = (
        session.query(IdeaRecord)
        .filter(IdeaRecord.status == "approved")
        .order_by(IdeaRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    return [f"{r.topic}: {r.angle}" for r in rows]


def get_recent_approved_topics(session: Session, limit: int = 20) -> list[str]:
    """
    Content Diversity Check: returns the last `limit` approved post topics+angles
    as compact strings (newest-first) to be passed to the Brand Guardian so it can
    score `strategic_fit` against real calendar history, not just a qualitative guess.

    Deliberately capped at 20 (roughly 400-500 tokens when injected into the prompt)
    to stay comfortably within LLM_MAX_TOKENS. Raise the cap if you increase
    LLM_MAX_TOKENS and want a deeper history window.

    Note: as of the human-approval-gate change, "approved" here means a human
    actually clicked Approve on the dashboard -- not just that the Brand Guardian
    passed it (that intermediate state is "pending_review"). This is intentional:
    dedup/diversity context should reflect genuinely finalized decisions, not
    content still sitting in the review queue.
    """
    rows = (
        session.query(IdeaRecord)
        .filter(IdeaRecord.status == "approved")
        .order_by(IdeaRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    return [f"{r.topic}: {r.angle}" for r in rows]


# --- Dashboard support: human approval gate, Idea Library browsing, logs, feedback ---


def get_pending_review(session: Session) -> list[tuple[IdeaRecord, PostRecord]]:
    """
    The dashboard's main review queue: every idea currently sitting at
    status='pending_review' (Brand Guardian passed it, no human decision yet),
    paired with its latest post. Oldest first, so the queue drains in order.
    """
    ideas = (
        session.query(IdeaRecord)
        .filter(IdeaRecord.status == "pending_review")
        .order_by(IdeaRecord.created_at.asc())
        .all()
    )
    results = []
    for idea in ideas:
        post = (
            session.query(PostRecord)
            .filter(PostRecord.idea_id == idea.id)
            .order_by(PostRecord.created_at.desc())
            .first()
        )
        if post is not None:
            results.append((idea, post))
    return results


def get_idea_library(session: Session, status: str | None = None, limit: int = 200) -> list[IdeaRecord]:
    """Idea Library browse view -- optionally filtered to one status, newest first."""
    query = session.query(IdeaRecord)
    if status:
        query = query.filter(IdeaRecord.status == status)
    return query.order_by(IdeaRecord.created_at.desc()).limit(limit).all()


def get_latest_post_for_idea(session: Session, idea_id: int) -> PostRecord | None:
    return (
        session.query(PostRecord)
        .filter(PostRecord.idea_id == idea_id)
        .order_by(PostRecord.created_at.desc())
        .first()
    )


def get_agent_logs(session: Session, agent_name: str | None = None, limit: int = 200) -> list[AgentDecisionRecord]:
    query = session.query(AgentDecisionRecord)
    if agent_name:
        query = query.filter(AgentDecisionRecord.agent_name == agent_name)
    return query.order_by(AgentDecisionRecord.timestamp.desc()).limit(limit).all()


def get_feedback_history(session: Session, limit: int = 100) -> list[FeedbackEditRecord]:
    return (
        session.query(FeedbackEditRecord)
        .order_by(FeedbackEditRecord.timestamp.desc())
        .limit(limit)
        .all()
    )


def update_post_content(
    session: Session,
    post_id: int,
    caption: str,
    image_prompt: str,
    hashtags: list[str],
    cta: str,
    platform_variants: dict[str, str],
) -> PostRecord:
    """Overwrite a post's editable fields in place -- used by the dashboard's Edit
    action. The pre-edit values should be captured via save_feedback_edit BEFORE
    calling this, since this overwrites them."""
    record = session.query(PostRecord).filter(PostRecord.id == post_id).one()
    record.caption = caption
    record.image_prompt = image_prompt
    record.hashtags = to_json(hashtags)
    record.cta = cta
    record.platform_variants = to_json(platform_variants)
    return record


def save_feedback_edit(
    session: Session,
    idea_id: int,
    post_id: int,
    topic: str,
    edit_type: str,
    original_output: dict,
    edited_output: dict,
    brand_version: int | None = None,
    platform: str = "primary",
    reason: str | None = None,
) -> FeedbackEditRecord:
    """
    Human Feedback Learning, Phase 1 capture. Call this BEFORE update_post_content
    overwrites the post row, passing the full pre-edit and post-edit content dicts
    (e.g. GeneratedContent.model_dump()) so both snapshots are preserved -- Phase 3's
    Performance Reviewer Agent needs the full before/after to mine style patterns
    later, not just a diff of one field.
    """
    record = FeedbackEditRecord(
        idea_id=idea_id,
        post_id=post_id,
        topic=topic,
        platform=platform,
        edit_type=edit_type,
        original_output=to_json(original_output),
        edited_output=to_json(edited_output),
        reason=reason,
        brand_version=brand_version,
    )
    session.add(record)
    session.flush()
    return record
