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
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

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


_engine = None
_SessionFactory = None


def get_engine(db_path: str | Path | None = None):
    global _engine
    if _engine is None:
        from app.core import config

        path = Path(db_path) if db_path else config.DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}", future=True)
        Base.metadata.create_all(_engine)
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
