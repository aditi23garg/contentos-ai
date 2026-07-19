"""
Inspect what's actually stored in SQLite and ChromaDB after running main.py a few
times -- useful for confirming persistence and dedup are doing what the console
output implies, rather than just trusting the printout.

Usage:
    python inspect_history.py                 # list everything stored so far
    python inspect_history.py "some idea text" # test dedup against your real history
"""

from __future__ import annotations

import sys

from app.repositories.db import AgentDecisionRecord, IdeaRecord, PostRecord, get_session
from app.repositories.vector_store import get_vector_store
from app.core import config


def list_history() -> None:
    session = get_session()
    ideas = session.query(IdeaRecord).order_by(IdeaRecord.created_at).all()
    posts = session.query(PostRecord).all()
    decisions = session.query(AgentDecisionRecord).all()

    print(f"SQLite: {len(ideas)} ideas, {len(posts)} posts, {len(decisions)} agent_decisions\n")
    print(f"{'ID':<4} {'Status':<10} {'Topic':<35} {'Dedup note'}")
    print("-" * 100)
    for idea in ideas:
        print(f"{idea.id:<4} {idea.status:<10} {idea.topic[:33]:<35} {idea.dedup_note[:50]}")
    session.close()

    store = get_vector_store()
    count = store._collection.count()
    print(f"\nChromaDB: {count} approved idea(s) indexed for dedup (threshold={config.DEDUP_SIMILARITY_THRESHOLD})")


def test_dedup(text: str) -> None:
    store = get_vector_store()
    matches = store.find_similar(text, config.DEDUP_SIMILARITY_THRESHOLD)
    print(f"Testing: {text!r}\n")
    if matches:
        print(f"WOULD BE FILTERED as a duplicate. Closest match:")
        for doc, score in matches:
            print(f"  similarity={score:.2f}  ->  {doc}")
    else:
        print("No match above threshold -- this would be treated as a fresh idea.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_dedup(" ".join(sys.argv[1:]))
    else:
        list_history()
