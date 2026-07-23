"""
Smoke test for Idea Library backlog top-up: verifies that build_batch_queue's
`stale` tracking, repository's backlog read-back/update-in-place, and the pipeline's
research_node deficit calculation all work together correctly -- without hitting a
live LLM or a real ChromaDB, per the pattern already used in test_json_repair.py.

Run with:  python test_backlog_topup.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".")

PASS, FAIL = "PASS", "FAIL"
errors: list[str] = []
total = 0

def fake_run_research(brand, count, recent_topics=None):
    captured_deficit["requested"] = count
    ...

class FakeVectorStore:
    def find_similar(self, text: str, threshold: float, topic: str | None = None, same_topic_threshold: float | None = None):
        if "ALREADY_APPROVED" in text:
            return [("some prior approved idea", 0.95)]
        return []

def check(name: str, condition: bool, detail: str = "") -> None:
    global total
    total += 1
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" -- {detail}" if detail else ""))
        errors.append(name)


# --- isolated SQLite for this test run --------------------------------------------
TEST_DB = Path("data/_test_backlog.db")
TEST_DB.parent.mkdir(parents=True, exist_ok=True)
if TEST_DB.exists():
    TEST_DB.unlink()

from app.repositories import db as db_module  # noqa: E402

db_module.reset_for_testing(TEST_DB)

from app.core.schemas import Idea  # noqa: E402
from app.repositories.repository import (  # noqa: E402
    get_backlog_ideas,
    save_idea,
    update_idea_status,
)
from app.repositories.db import IdeaRecord, get_session  # noqa: E402
from app.services.batching import build_batch_queue  # noqa: E402


# ==== Part 1: repository round trip ================================================
print("\n=== repository: backlog read-back + in-place update ===\n")

session = get_session()
seeded = Idea(
    topic="Discipline over motivation",
    angle="Motivation fades, systems don't",
    reasoning="Evergreen habits angle, aligns with brand philosophy",
    confidence_score=0.82,
    knowledge_sources_used=["books"],
)
record = save_idea(session, seeded, status="backlog")
session.commit()
seeded_id = record.id
session.close()

session = get_session()
backlog = get_backlog_ideas(session, limit=10)
check("T1 backlog idea read back as Idea with source_backlog_id set",
      len(backlog) == 1 and backlog[0].source_backlog_id == seeded_id)
check("T1b fields round-trip correctly", backlog[0].topic == "Discipline over motivation"
      and backlog[0].knowledge_sources_used == ["books"])
session.close()

session = get_session()
update_idea_status(session, seeded_id, "approved")
session.commit()
row_count = session.query(IdeaRecord).count()
updated_status = session.query(IdeaRecord).filter(IdeaRecord.id == seeded_id).one().status
session.close()
check("T2 update_idea_status updates in place (no duplicate row)", row_count == 1)
check("T2b status actually changed", updated_status == "approved")


# ==== Part 2: build_batch_queue stale detection ====================================
print("\n=== batching: stale backlog detection ===\n")


class FakeVectorStore:
    """Deterministic stand-in for ChromaDB -- flags any idea whose topic contains
    'ALREADY_APPROVED' as matching history, everything else as fresh."""

    def find_similar(self, text: str, threshold: float):
        if "ALREADY_APPROVED" in text:
            return [("some prior approved idea", 0.95)]
        return []


backlog_now_stale = Idea(
    topic="ALREADY_APPROVED topic",
    angle="x",
    reasoning="r",
    confidence_score=0.9,
    source_backlog_id=101,
)
backlog_still_good = Idea(
    topic="Fresh backlog idea",
    angle="y",
    reasoning="r",
    confidence_score=0.7,
    source_backlog_id=102,
)
fresh_candidate = Idea(
    topic="Brand new idea",
    angle="z",
    reasoning="r",
    confidence_score=0.6,
)

queue, surplus, stale, note = build_batch_queue(
    candidates=[backlog_now_stale, backlog_still_good, fresh_candidate],
    vector_store=FakeVectorStore(),
    dedup_threshold=0.85,
    batch_size=5,
)

check("T3 stale backlog idea filtered out and reported in `stale`",
      len(stale) == 1 and stale[0].source_backlog_id == 101)
check("T4 non-stale backlog idea survives into queue/surplus",
      any(i.source_backlog_id == 102 for i in (*queue, *surplus)))
check("T5 fresh candidate unaffected", any(i.topic == "Brand new idea" for i in (*queue, *surplus)))
check("T6 note mentions stale archiving", "stale backlog" in note)


# ==== Part 3: research_node deficit calculation (monkeypatched LLM) ================
print("\n=== pipeline: research_node pulls backlog before asking for fresh ideas ===\n")

# Seed two more backlog ideas so we have 3 backlog rows total (1 already promoted above).
session = get_session()
for i in range(2):
    save_idea(
        session,
        Idea(topic=f"Backlog idea {i}", angle="a", reasoning="r", confidence_score=0.5 + i * 0.1),
        status="backlog",
    )
session.commit()
session.close()

from app.core import config as config_module  # noqa: E402

config_module.IDEAS_PER_BATCH = 5
config_module.BATCH_SIZE = 5
config_module.DEDUP_SIMILARITY_THRESHOLD = 0.85

captured_deficit = {}


def fake_run_research(brand, count, recent_topics=None):
    captured_deficit["requested"] = count
    return [
        Idea(topic=f"Fresh {i}", angle="a", reasoning="r", confidence_score=0.4)
        for i in range(count)
    ]


class FakeLLM:
    last_retry_count = 0


with patch("app.graph.pipeline.run_research", side_effect=fake_run_research), \
     patch("app.graph.pipeline.get_vector_store", return_value=FakeVectorStore()), \
     patch("app.graph.pipeline.get_llm", return_value=FakeLLM()):
    from app.graph.pipeline import research_node
    from app.core.schemas import BrandProfile

    brand = BrandProfile(
        brand_name="Test Brand", mission="m", vision="v", tone=["warm"],
        writing_style=["simple"], audience="18-35", visual_style=["clean"],
        preferred_colors=["green"], content_philosophy=["evergreen"],
        niche="motivation", allowed_topics=["discipline"],
    )
    result = research_node({"brand": brand})

check(
    "T7 only the backlog deficit (5 - 2 backlog rows) is requested from Research",
    captured_deficit["requested"] == 3,
    f"got {captured_deficit.get('requested')}",
)
check("T8 combined pool includes both backlog and fresh ideas", len(result["queue"]) + len(result["surplus"]) >= 2)

# Release the SQLite file handle before deleting it -- on Windows, unlike POSIX,
# a file can't be removed while SQLAlchemy's connection pool still holds it open.
if db_module._engine is not None:
    db_module._engine.dispose()
try:
    TEST_DB.unlink(missing_ok=True)
except PermissionError:
    print(f"  (note: could not delete {TEST_DB} -- still locked by the OS; harmless, ignore)")

print()
if errors:
    print(f"  {len(errors)}/{total} test(s) FAILED: {', '.join(errors)}")
    sys.exit(1)
else:
    print(f"  All {total} tests passed.\n")
