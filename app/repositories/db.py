"""
SQLite persistence layer.

Implements the `Idea`, `Post`, and `agent_decisions` tables from the spec's Database
section. This is deliberately plain SQLAlchemy Core-ish models, not a full repository
abstraction with an interface per entity yet -- that's worth adding once there's a
second storage backend to abstract against (PostgreSQL, per the spec's "future"), not
before.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class IdeaRecord(Base):
    __tablename__ = "ideas"

    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    angle = Column(String, nullable=False)
    reasoning = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    knowledge_sources_used = Column(Text)  # JSON-encoded list
    status = Column(String, default="new")  # new -> ... -> approved / rejected (Idea Library lifecycle)
    dedup_note = Column(Text)  # what the dedup check found/did for this idea, if anything
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PostRecord(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    idea_id = Column(Integer, nullable=False)
    caption = Column(Text, nullable=False)
    image_prompt = Column(Text, nullable=False)
    hashtags = Column(Text)  # JSON-encoded list
    cta = Column(Text)
    platform_variants = Column(Text)  # JSON-encoded dict
    prompt_version = Column(String)
    passed = Column(Boolean, nullable=False)
    rubric_scores = Column(Text)  # JSON-encoded dict
    guardian_reason = Column(Text)
    brand_version = Column(Integer, nullable=True)  # BrandProfile.version at time of production
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class AgentDecisionRecord(Base):
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False)
    input_summary = Column(Text)
    output_summary = Column(Text)
    passed = Column(Boolean, nullable=True)
    scores = Column(Text)  # JSON-encoded dict, nullable (Research/Producer entries have none)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class FeedbackEditRecord(Base):
    """
    Human Feedback Learning capture (Phase 1 of the Feedback Loop spec).

    Each row stores the full before/after content snapshot whenever a human edits
    a post in the dashboard before approving it. This gives the Phase 3 Performance
    Reviewer Agent the raw material to mine style preferences without needing to
    reconstruct diffs from individual field changes.

    Columns:
        idea_id / post_id  -- FK references (not enforced at DB level to keep SQLite
                              constraints simple; enforced by the repository layer).
        topic              -- Idea.topic at time of edit, denormalised so the table
                              is readable without a join.
        platform           -- Which platform variant was edited ('primary', 'instagram',
                              'linkedin'). 'primary' means the main caption/prompt/etc.
        edit_type          -- Free-form label set by the dashboard action: 'caption_edit',
                              'image_prompt_edit', 'full_edit', etc.
        original_output    -- JSON snapshot of GeneratedContent BEFORE the edit.
        edited_output      -- JSON snapshot of GeneratedContent AFTER the edit.
        reason             -- Optional free-text note the human entered explaining why.
        brand_version      -- BrandProfile.version active at the time of the edit.
        timestamp          -- When the edit was saved.
    """
    __tablename__ = "feedback_edits"

    id = Column(Integer, primary_key=True)
    idea_id = Column(Integer, nullable=False)
    post_id = Column(Integer, nullable=False)
    topic = Column(String, nullable=False)
    platform = Column(String, default="primary")
    edit_type = Column(String, nullable=False)
    original_output = Column(Text, nullable=False)  # JSON
    edited_output = Column(Text, nullable=False)    # JSON
    reason = Column(Text, nullable=True)
    brand_version = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)



_engine = None
_SessionFactory = None


def _run_lightweight_migrations(engine) -> None:
    """
    SQLite-only auto-migration for columns added to a Record class *after* an
    existing .db file was first created.

    Base.metadata.create_all() (called right before this) only creates tables
    that don't exist yet -- it never alters an existing table to pick up a new
    column added to the model later (e.g. PostRecord.brand_version). Without
    this, every existing local .db file crashes with
    "table X has no column named Y" the first time a row is inserted, even
    though the code itself is correct. That's a confusing failure mode for a
    single-developer personal build where "just delete the .db and rerun" is
    an easy but data-losing workaround -- this makes it unnecessary.

    Compares each mapped table's expected columns (from Base.metadata) against
    what SQLite actually has (via PRAGMA table_info, through SQLAlchemy's
    inspector) and ALTERs in whatever is missing. Deliberately minimal: it only
    ever ADDs columns, never renames, retypes, or drops one, and every added
    column is created without a NOT NULL constraint regardless of the model's
    nullable=False -- SQLite can't add a NOT NULL column without a constant
    default on existing rows, and guessing one is worse than leaving it
    nullable. Safe for a personal, single-developer SQLite database; not a
    substitute for a real migration tool (Alembic) if this ever moves to
    Postgres/multi-user -- see the module docstring's note on that boundary.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # brand-new table -- create_all() already created it whole

            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                logger.warning(
                    "Auto-migrating %s: adding missing column '%s' (%s). "
                    "Existing rows will have NULL for this column.",
                    table.name,
                    column.name,
                    col_type,
                )
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'))


def get_engine(db_path: str | Path | None = None):
    global _engine
    if _engine is None:
        from app.core import config

        path = Path(db_path) if db_path else config.DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}", future=True)
        Base.metadata.create_all(_engine)
        _run_lightweight_migrations(_engine)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), future=True)
    return _SessionFactory()


def reset_for_testing(db_path: str | Path) -> None:
    """Force a fresh engine bound to a specific path -- used by tests only."""
    global _engine, _SessionFactory
    _engine = None
    _SessionFactory = None
    get_engine(db_path)


def to_json(value) -> str:
    return json.dumps(value)


def from_json(value: str | None):
    return json.loads(value) if value else None
