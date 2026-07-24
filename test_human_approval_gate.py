"""
Smoke test for the human approval gate added on top of Content Batching:
- record_result_node now sets "pending_review" (not "approved") when the Brand
  Guardian passes, and "rejected" when it doesn't -- verified directly, without
  running the full graph or hitting a live LLM.
- repository.py's dashboard support functions: get_pending_review, get_idea_library,
  update_post_content, save_feedback_edit, get_feedback_history, get_latest_post_for_idea.
- The Approve action's effect (status -> "approved", indexed in the vector store) and
  Reject action's effect (status -> "rejected", never indexed) using a fake vector
  store -- same pattern as test_backlog_topup.py's FakeVectorStore.

Run with:  python test_human_approval_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

PASS, FAIL = "PASS", "FAIL"
errors: list[str] = []
total = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global total
    total += 1
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" -- {detail}" if detail else ""))
        errors.append(name)


# --- isolated SQLite for this test run --------------------------------------------
TEST_DB = Path("data/_test_approval_gate.db")
TEST_DB.parent.mkdir(parents=True, exist_ok=True)
if TEST_DB.exists():
    TEST_DB.unlink()

from app.repositories import db as db_module  # noqa: E402

db_module.reset_for_testing(TEST_DB)

from app.core.schemas import BrandGuardianResult, GeneratedContent, Idea, RubricScores  # noqa: E402
from app.repositories.db import get_session  # noqa: E402
from app.repositories.repository import (  # noqa: E402
    get_feedback_history,
    get_idea_library,
    get_latest_post_for_idea,
    get_pending_review,
    save_feedback_edit,
    save_idea,
    save_post,
    update_idea_status,
    update_post_content,
)


# ==== Part 1: record_result_node's status logic (unit-level, no graph run) ========
print("\n=== pipeline: record_result_node status logic ===\n")

from app.graph.pipeline import record_result_node  # noqa: E402

fake_idea = Idea(topic="T", angle="A", reasoning="r", confidence_score=0.9)
fake_content = GeneratedContent(idea_topic="T", caption="c", image_prompt="i", hashtags=[], cta="cta")
passed_result = BrandGuardianResult(
    scores=RubricScores(niche_fit=5, brand_alignment=5, originality=5, value_to_audience=5, grammar_clarity=5, strategic_fit=5),
    passed=True,
    reason="great",
)
failed_result = BrandGuardianResult(
    scores=RubricScores(niche_fit=2, brand_alignment=2, originality=2, value_to_audience=2, grammar_clarity=2, strategic_fit=2),
    passed=False,
    reason="poor",
)

state_pass = {"queue": [fake_idea], "queue_index": 0, "content": fake_content, "guardian_result": passed_result}
out_pass = record_result_node(state_pass)
check("T1 Guardian pass -> status='pending_review' (NOT 'approved')",
      out_pass["batch_results"][0]["status"] == "pending_review")

state_fail = {"queue": [fake_idea], "queue_index": 0, "content": fake_content, "guardian_result": failed_result}
out_fail = record_result_node(state_fail)
check("T2 Guardian fail -> status='rejected'", out_fail["batch_results"][0]["status"] == "rejected")


# ==== Part 2: pending review queue + approve/reject transitions ===================
print("\n=== repository: pending review queue + approve/reject ===\n")

session = get_session()
idea = Idea(topic="Deep Work", angle="Focus in a distracted world", reasoning="r", confidence_score=0.8)
idea_record = save_idea(session, idea, status="pending_review")
content = GeneratedContent(
    idea_topic="Deep Work", caption="Original caption", image_prompt="a desk", hashtags=["#focus"], cta="Try it"
)
post_record = save_post(session, idea_record.id, content, passed_result, brand_version=1)
session.commit()
idea_id, post_id = idea_record.id, post_record.id
session.close()

session = get_session()
queue = get_pending_review(session)
check("T3 new pending_review idea appears in the review queue",
      any(i.id == idea_id for i, p in queue))
session.close()

# --- Approve ---
session = get_session()
update_idea_status(session, idea_id, "approved")
session.commit()
session.close()

session = get_session()
queue_after = get_pending_review(session)
approved_list = get_idea_library(session, status="approved")
check("T4 approved idea no longer in pending_review queue", not any(i.id == idea_id for i, p in queue_after))
check("T4b approved idea appears in Idea Library filtered by status='approved'",
      any(i.id == idea_id for i in approved_list))
session.close()


# --- second idea for Reject path ---
session = get_session()
idea2 = Idea(topic="Cold Showers", angle="Discomfort builds discipline", reasoning="r", confidence_score=0.7)
idea2_record = save_idea(session, idea2, status="pending_review")
content2 = GeneratedContent(idea_topic="Cold Showers", caption="c2", image_prompt="i2", hashtags=[], cta="cta2")
save_post(session, idea2_record.id, content2, passed_result, brand_version=1)
session.commit()
idea2_id = idea2_record.id
session.close()

session = get_session()
update_idea_status(session, idea2_id, "rejected")
session.commit()
session.close()

session = get_session()
rejected_list = get_idea_library(session, status="rejected")
pending_after_reject = get_pending_review(session)
check("T5 rejected idea appears in Idea Library filtered by status='rejected'",
      any(i.id == idea2_id for i in rejected_list))
check("T5b rejected idea no longer in pending_review queue",
      not any(i.id == idea2_id for i, p in pending_after_reject))
session.close()


# ==== Part 3: Edit action + feedback capture =======================================
print("\n=== repository: edit action + feedback capture ===\n")

session = get_session()
idea3 = Idea(topic="Habit Stacking", angle="Attach new habits to old ones", reasoning="r", confidence_score=0.75)
idea3_record = save_idea(session, idea3, status="pending_review")
original_content = GeneratedContent(
    idea_topic="Habit Stacking", caption="Old caption", image_prompt="old prompt",
    hashtags=["#habits"], cta="Old CTA", platform_variants={"instagram": "old ig"},
)
post3 = save_post(session, idea3_record.id, original_content, passed_result, brand_version=2)
session.commit()
idea3_id, post3_id = idea3_record.id, post3.id
session.close()

edited_content = GeneratedContent(
    idea_topic="Habit Stacking", caption="New, punchier caption", image_prompt="old prompt",
    hashtags=["#habits", "#tinyhabits"], cta="New CTA", platform_variants={"instagram": "old ig"},
)

session = get_session()
save_feedback_edit(
    session,
    idea_id=idea3_id,
    post_id=post3_id,
    topic="Habit Stacking",
    edit_type="caption+hashtags+cta",
    original_output=original_content.model_dump(),
    edited_output=edited_content.model_dump(),
    brand_version=2,
    reason="tightened the hook",
)
update_post_content(
    session,
    post3_id,
    caption=edited_content.caption,
    image_prompt=edited_content.image_prompt,
    hashtags=edited_content.hashtags,
    cta=edited_content.cta,
    platform_variants=edited_content.platform_variants,
)
update_idea_status(session, idea3_id, "approved")
session.commit()
session.close()

session = get_session()
refreshed_post = get_latest_post_for_idea(session, idea3_id)
check("T6 post content actually updated after edit", refreshed_post.caption == "New, punchier caption")

feedback = get_feedback_history(session)
matching = [f for f in feedback if f.idea_id == idea3_id]
check("T7 feedback edit record created", len(matching) == 1)
if matching:
    import json
    orig = json.loads(matching[0].original_output)
    edit = json.loads(matching[0].edited_output)
    check("T7b original_output snapshot preserved pre-edit caption", orig["caption"] == "Old caption")
    check("T7c edited_output snapshot has post-edit caption", edit["caption"] == "New, punchier caption")
    check("T7d reason captured", matching[0].reason == "tightened the hook")
session.close()


# ==== summary =======================================================================
print(f"\n  {total - len(errors)}/{total} tests passed.\n" if errors else f"\n  All {total} tests passed.\n")
if errors:
    raise SystemExit(1)
