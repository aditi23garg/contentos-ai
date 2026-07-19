"""
Repository functions: the only place that translates between the Pydantic schemas
the agents/graph work with (app/core/schemas.py) and the SQLAlchemy records that get
persisted (app/repositories/db.py). Nothing outside this module should import the
*Record classes directly -- that's the point of the Repository Pattern.
"""

from __future__ import annotations

from app.core.schemas import AgentDecisionLog, BrandGuardianResult, GeneratedContent, Idea
from app.repositories.db import AgentDecisionRecord, IdeaRecord, PostRecord, Session, from_json, to_json


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


def save_post(
    session: Session, idea_id: int, content: GeneratedContent, guardian_result: BrandGuardianResult
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
