"""
ContentOS AI — Streamlit Dashboard (Phase 1).

The human side of the approval gate added on top of Content Batching: every idea the
Brand Guardian passes lands here as "pending_review", not "approved" -- run this to
actually Approve, Reject, Edit, or Regenerate each item. Only an Approve click here
indexes the idea in ChromaDB, so future dedup/diversity checks reflect real decisions,
not just what the Guardian happened to pass.

Per the spec's Dashboard requirements, this shows: This Week's Batch (-> Pending
Review tab), Idea Library (status filter), Captions, Platform Preview, Agent Logs
(incl. idea-selection reasoning), Feedback History, and Approve/Reject/Edit/Regenerate
controls. Generated Images and the 30-Day Calendar are not shown -- there is no Image
Generation Agent yet (image_prompt is shown as text instead, honestly labeled as such)
and no Scheduler (Phase 2), so there's nothing real to display for either yet.
Analytics is Phase 3+ and likewise omitted rather than faked.

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json

import streamlit as st

from app.agents.brand_guardian_agent import evaluate
from app.agents.content_producer_agent import produce_content
from app.core.config import load_brand_profile
from app.core.schemas import GeneratedContent, Idea
from app.repositories.db import IdeaRecord, PostRecord, from_json, get_session
from app.repositories.repository import (
    get_agent_logs,
    get_feedback_history,
    get_idea_library,
    get_pending_review,
    get_recent_approved_topics,
    save_feedback_edit,
    save_post,
    update_idea_status,
    update_post_content,
)
from app.repositories.vector_store import get_vector_store

st.set_page_config(page_title="ContentOS AI — Dashboard", layout="wide")

IDEA_STATUSES = ["new", "pending", "backlog", "pending_review", "approved", "rejected", "archived"]


# --- shared helpers -----------------------------------------------------------------


@st.cache_resource
def _brand():
    return load_brand_profile()


def _post_content(post: PostRecord) -> GeneratedContent:
    return GeneratedContent(
        idea_topic="",  # not stored on PostRecord itself; not needed for display/edit
        caption=post.caption,
        image_prompt=post.image_prompt,
        hashtags=from_json(post.hashtags) or [],
        cta=post.cta,
        platform_variants=from_json(post.platform_variants) or {},
        prompt_version=post.prompt_version or "v1",
    )


def _approve(idea_id: int, topic: str, angle: str) -> None:
    session = get_session()
    try:
        update_idea_status(session, idea_id, "approved")
        session.commit()
    finally:
        session.close()
    # Only a human Approve indexes the idea -- see pipeline.py's persist_batch_node
    # docstring for why a Guardian pass alone no longer does this.
    get_vector_store().add_idea(str(idea_id), f"{topic}: {angle}", topic=topic)
    st.rerun()


def _reject(idea_id: int) -> None:
    session = get_session()
    try:
        update_idea_status(session, idea_id, "rejected")
        session.commit()
    finally:
        session.close()
    st.rerun()


def _save_edit(
    idea_id: int,
    post_id: int,
    topic: str,
    brand_version: int | None,
    original: GeneratedContent,
    new_caption: str,
    new_image_prompt: str,
    new_hashtags: list[str],
    new_cta: str,
    new_platform_variants: dict[str, str],
    reason: str,
    also_approve: bool,
    angle: str,
) -> None:
    edited = GeneratedContent(
        idea_topic=topic,
        caption=new_caption,
        image_prompt=new_image_prompt,
        hashtags=new_hashtags,
        cta=new_cta,
        platform_variants=new_platform_variants,
        prompt_version=original.prompt_version,
    )
    session = get_session()
    try:
        # Capture the pre-edit snapshot BEFORE overwriting -- see save_feedback_edit's
        # docstring for why both snapshots (not just a diff) are stored.
        save_feedback_edit(
            session,
            idea_id=idea_id,
            post_id=post_id,
            topic=topic,
            edit_type="multi_field",
            original_output=original.model_dump(),
            edited_output=edited.model_dump(),
            brand_version=brand_version,
            reason=reason or None,
        )
        update_post_content(
            session,
            post_id,
            caption=new_caption,
            image_prompt=new_image_prompt,
            hashtags=new_hashtags,
            cta=new_cta,
            platform_variants=new_platform_variants,
        )
        update_idea_status(session, idea_id, "approved" if also_approve else "pending_review")
        session.commit()
    finally:
        session.close()
    if also_approve:
        get_vector_store().add_idea(str(idea_id), f"{topic}: {angle}", topic=topic)
    st.session_state.pop(f"editing_{idea_id}", None)
    st.rerun()


def _regenerate(idea_record: IdeaRecord) -> None:
    brand = _brand()
    idea = Idea(
        topic=idea_record.topic,
        angle=idea_record.angle,
        reasoning=idea_record.reasoning,
        confidence_score=idea_record.confidence_score,
        knowledge_sources_used=from_json(idea_record.knowledge_sources_used) or [],
    )
    with st.spinner(f"Regenerating '{idea_record.topic}' with a fresh LLM call..."):
        content = produce_content(brand, idea)
        session = get_session()
        try:
            recent_topics = get_recent_approved_topics(session, limit=20)
        finally:
            session.close()
        result = evaluate(brand, content, recent_topics=recent_topics)

    session = get_session()
    try:
        save_post(session, idea_record.id, content, result, brand_version=brand.version)
        new_status = "pending_review" if result.passed else "rejected"
        update_idea_status(session, idea_record.id, new_status)
        session.commit()
    finally:
        session.close()
    st.rerun()


# --- sidebar --------------------------------------------------------------------------

brand = _brand()
with st.sidebar:
    st.header(brand.brand_name)
    st.caption(brand.mission)
    st.write(f"**Niche:** {brand.niche}")
    st.write(f"**Tone:** {', '.join(brand.tone)}")
    st.write(f"**Brand version:** v{brand.version}")

    st.divider()
    session = get_session()
    try:
        counts = {
            s: len(get_idea_library(session, status=s, limit=100000)) for s in IDEA_STATUSES
        }
    finally:
        session.close()
    st.subheader("Idea Library counts")
    for s, c in counts.items():
        if c:
            st.write(f"{s}: **{c}**")

    st.divider()
    if st.button("🔄 Run new batch (calls Groq — may take a minute)", width='stretch'):
        from app.graph.pipeline import build_pipeline

        with st.spinner("Running Research → Dedup → Produce → Guardian for a new batch..."):
            final_state = build_pipeline().invoke({"brand": brand})
        st.success(final_state["batch_summary"])
        st.rerun()


# --- tabs -------------------------------------------------------------------------

tab_review, tab_library, tab_logs, tab_feedback = st.tabs(
    ["📋 Pending Review", "📚 Idea Library", "🗒️ Agent Logs", "✏️ Feedback History"]
)


with tab_review:
    session = get_session()
    try:
        queue = get_pending_review(session)
        # Detach the ORM objects' plain values we need before closing the session --
        # avoids DetachedInstanceError when widgets below reference them after close.
        items = [
            (
                idea.id, idea.topic, idea.angle, idea.reasoning, idea.confidence_score, idea.dedup_note,
                post.id, post.brand_version, post,
            )
            for idea, post in queue
        ]
    finally:
        session.close()

    if not items:
        st.info("No items awaiting review. Run a new batch from the sidebar, or with `python main.py`.")

    for idea_id, topic, angle, reasoning, confidence, dedup_note, post_id, brand_version, post in items:
        content = _post_content(post)
        rubric = json.loads(post.rubric_scores) if post.rubric_scores else {}
        avg_score = sum(rubric.values()) / len(rubric) if rubric else 0.0

        with st.container(border=True):
            st.subheader(f"{topic}  ·  confidence {confidence:.2f}  ·  Guardian avg {avg_score:.2f}")
            st.caption(f"Angle: {angle}")
            if dedup_note:
                st.caption(f"Dedup: {dedup_note}")

            col_main, col_meta = st.columns([2, 1])
            with col_main:
                st.markdown("**Caption**")
                st.write(content.caption)
                st.markdown("**Image prompt** *(no Image Generation Agent yet — shown as text)*")
                st.write(content.image_prompt)
                st.markdown("**Hashtags**")
                st.write(" ".join(content.hashtags))
                st.markdown("**CTA**")
                st.write(content.cta)
                if content.platform_variants:
                    st.markdown("**Platform preview**")
                    variant_tabs = st.tabs(list(content.platform_variants.keys()))
                    for vtab, (platform, text) in zip(variant_tabs, content.platform_variants.items()):
                        with vtab:
                            st.write(text)
            with col_meta:
                st.markdown("**Rubric scores**")
                for dim, score in rubric.items():
                    st.write(f"{dim}: {score}")
                st.markdown("**Guardian reasoning**")
                st.caption(post.guardian_reason or "—")
                st.markdown("**Why this idea (Research)**")
                st.caption(reasoning)

            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            if btn_col1.button("✅ Approve", key=f"approve_{idea_id}", width='stretch'):
                _approve(idea_id, topic, angle)
            if btn_col2.button("❌ Reject", key=f"reject_{idea_id}", width='stretch'):
                _reject(idea_id)
            if btn_col3.button("🔁 Regenerate", key=f"regen_{idea_id}", width='stretch'):
                session = get_session()
                try:
                    idea_record = session.query(IdeaRecord).filter(IdeaRecord.id == idea_id).one()
                    # copy plain values out before the session that owns them closes
                    idea_snapshot = IdeaRecord(
                        id=idea_record.id, topic=idea_record.topic, angle=idea_record.angle,
                        reasoning=idea_record.reasoning, confidence_score=idea_record.confidence_score,
                        knowledge_sources_used=idea_record.knowledge_sources_used,
                    )
                finally:
                    session.close()
                _regenerate(idea_snapshot)
            if btn_col4.button("✏️ Edit", key=f"edit_toggle_{idea_id}", width='stretch'):
                st.session_state[f"editing_{idea_id}"] = not st.session_state.get(f"editing_{idea_id}", False)

            if st.session_state.get(f"editing_{idea_id}"):
                with st.form(key=f"edit_form_{idea_id}"):
                    new_caption = st.text_area("Caption", value=content.caption, key=f"cap_{idea_id}")
                    new_image_prompt = st.text_area(
                        "Image prompt", value=content.image_prompt, key=f"img_{idea_id}"
                    )
                    new_hashtags_raw = st.text_input(
                        "Hashtags (space-separated)", value=" ".join(content.hashtags), key=f"tags_{idea_id}"
                    )
                    new_cta = st.text_input("CTA", value=content.cta, key=f"cta_{idea_id}")
                    reason = st.text_input(
                        "Reason for edit (optional, feeds future learning)", key=f"reason_{idea_id}"
                    )
                    save_col, save_approve_col = st.columns(2)
                    save_only = save_col.form_submit_button("Save (keep in review)")
                    save_and_approve = save_approve_col.form_submit_button("Save & Approve")

                    if save_only or save_and_approve:
                        _save_edit(
                            idea_id=idea_id,
                            post_id=post_id,
                            topic=topic,
                            brand_version=brand_version,
                            original=content,
                            new_caption=new_caption,
                            new_image_prompt=new_image_prompt,
                            new_hashtags=new_hashtags_raw.split(),
                            new_cta=new_cta,
                            new_platform_variants=content.platform_variants,
                            reason=reason,
                            also_approve=save_and_approve,
                            angle=angle,
                        )


with tab_library:
    status_filter = st.selectbox("Status filter", ["all"] + IDEA_STATUSES)
    session = get_session()
    try:
        rows = get_idea_library(session, status=None if status_filter == "all" else status_filter)
        table = [
            {
                "id": r.id,
                "topic": r.topic,
                "angle": r.angle,
                "status": r.status,
                "confidence": round(r.confidence_score, 2),
                "dedup_note": r.dedup_note,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    finally:
        session.close()
    st.dataframe(table, width='stretch', hide_index=True)


with tab_logs:
    agent_filter = st.selectbox(
        "Agent", ["all", "ResearchAgent", "ContentProducerAgent", "BrandGuardianAgent"]
    )
    session = get_session()
    try:
        logs = get_agent_logs(session, agent_name=None if agent_filter == "all" else agent_filter)
        table = [
            {
                "timestamp": r.timestamp,
                "agent": r.agent_name,
                "input": r.input_summary,
                "output": r.output_summary,
                "passed": r.passed,
            }
            for r in logs
        ]
    finally:
        session.close()
    st.dataframe(table, width='stretch', hide_index=True)


with tab_feedback:
    session = get_session()
    try:
        edits = get_feedback_history(session)
        snapshot = [
            {
                "timestamp": e.timestamp,
                "topic": e.topic,
                "edit_type": e.edit_type,
                "reason": e.reason,
                "brand_version": e.brand_version,
                "original": e.original_output,
                "edited": e.edited_output,
            }
            for e in edits
        ]
    finally:
        session.close()

    if not snapshot:
        st.info("No edits logged yet.")
    for row in snapshot:
        with st.expander(f"{row['topic']} — {row['edit_type']} ({row['timestamp']:%Y-%m-%d %H:%M})"):
            if row["reason"]:
                st.caption(f"Reason: {row['reason']}")
            col_a, col_b = st.columns(2)
            orig = json.loads(row["original"])
            edit = json.loads(row["edited"])
            with col_a:
                st.markdown("**Before**")
                st.write(orig.get("caption", ""))
            with col_b:
                st.markdown("**After**")
                st.write(edit.get("caption", ""))
